"""
main.py — Astram Guardian API
==============================
Serves the trained resolution-time model behind a FastAPI app, generates
an automated dispatch playbook, and pulls in live OSM/OSRM data for
human-readable display.

KEY DESIGN PRINCIPLE
---------------------
The model's `corridor`, `zone`, and `police_station` features were one-hot
encoded from a FIXED vocabulary seen during training (22 corridors, 10
zones, 54 police stations). If we feed the model live OpenStreetMap road
names or suburb names, they will almost never match that vocabulary, and
the model silently treats them as "unknown" — wasting a real signal it
learned. So this API keeps two separate things distinct:

  1. What the MODEL sees: the nearest known corridor/zone/police_station
     centroid to the reported GPS point (from location_lookup.json,
     built directly from the training data by train_model.py).
  2. What the USER sees: a live, human-readable road/suburb name from
     OpenStreetMap reverse geocoding, purely for display.

Required files (all produced by train_model.py, expected in ./output/):
    resolution_time_model.joblib
    category_vocab.json
    location_lookup.json
    metrics.json   (optional — only used by GET /metadata)
"""

import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ASSET_DIR = "output"

# --------------------------------------------------------------------------
# Default diversion suggestions, one per real named corridor seen in
# training. "Non-corridor" is deliberately excluded here -- it's a
# catch-all label for events that weren't on any major road, so there's
# no real road to suggest a parallel route for.
#
# NOTE: these are illustrative starting points based on general Bengaluru
# road geography, not verified traffic-engineering routes. Review/replace
# with locally-confirmed parallel roads before relying on this for a real
# deployment -- for the hackathon demo it guarantees the playbook always
# names a concrete next step instead of a vague placeholder.
# --------------------------------------------------------------------------
STATIC_DIVERSION_MAP = {
    "Tumkur Road": "Divert inbound traffic via Magadi Road; heavy vehicles further out via Bellary Road.",
    "Mysore Road": "Divert via Magadi Road or Kanakapura Road.",
    "Magadi Road": "Divert via Mysore Road or Tumkur Road.",
    "Bellary Road 1": "Divert via Tumkur Road or Hennur Main Road depending on direction of travel.",
    "Bellary Road 2": "Divert via Hennur Main Road or IRR (Thanisandra Road).",
    "Hosur Road": "Divert outgoing traffic via Bannerghatta Road.",
    "Bannerghata Road": "Divert via Hosur Road or Kanakapura Road.",
    "Airport New South Road": "Divert via Old Airport Road or Varthur Road.",
    "Old Airport Road": "Divert via the ORR East corridor or Varthur Road.",
    "Varthur Road": "Divert via Old Airport Road or the ORR East corridor.",
    "ORR East 1": "Divert through Old Madras Road / Indiranagar.",
    "ORR East 2": "Divert via Old Airport Road or Varthur Road.",
    "ORR North 1": "Divert via Hennur Main Road or Bellary Road.",
    "ORR North 2": "Divert via Bellary Road or Hennur Main Road.",
    "ORR West 1": "Divert via Magadi Road or Mysore Road.",
    "Hennur Main Road": "Divert via Bellary Road or IRR (Thanisandra Road).",
    "IRR(Thanisandra road)": "Divert via Hennur Main Road or the ORR North corridor.",
    "Old Madras Road": "Divert via the ORR East corridor or through the CBD.",
    "West of Chord Road": "Divert via Magadi Road or Tumkur Road.",
    "CBD 1": "Divert via CBD 2 or Old Madras Road.",
    "CBD 2": "Divert via CBD 1 or Old Madras Road.",
}


# --------------------------------------------------------------------------
# Curated, frontend-friendly labels for the messier raw training vocabulary.
# The raw vocab (loaded from category_vocab.json at startup) still has data
# quality quirks from the original dataset -- e.g. "Debris" and "debris"
# exist as two separate trained categories. Validation accepts any raw
# trained value, but the dropdown only needs to offer the clean ones.
# --------------------------------------------------------------------------
EVENT_CAUSE_LABELS = {
    "vehicle_breakdown": "Vehicle Breakdown",
    "accident": "Accident",
    "water_logging": "Water Logging / Flood",
    "pot_holes": "Severe Pothole Damage",
    "construction": "Road Construction",
    "tree_fall": "Tree Fall",
    "road_conditions": "Poor Road Conditions",
    "congestion": "Congestion",
    "public_event": "Planned Public Event (Rally/Match)",
    "procession": "Procession",
    "vip_movement": "VIP Movement",
    "protest": "Protest",
    "others": "Other",
}

VEH_TYPE_LABELS = {
    "bmtc_bus": "BMTC Public Transit Bus",
    "ksrtc_bus": "KSRTC Bus",
    "private_bus": "Private Bus",
    "heavy_vehicle": "Heavy Commercial Vehicle / Truck",
    "truck": "Truck",
    "lcv": "Light Commercial Vehicle",
    "taxi": "Taxi",
    "private_car": "Private Car",
    "auto": "Auto Rickshaw",
    "others": "Other / None",
}

NO_VEHICLE_CAUSES = {"public_event", "water_logging", "pot_holes", "construction",
                      "road_conditions", "congestion", "procession", "protest",
                      "vip_movement", "tree_fall", "others"}

# Beyond this distance from any known training centroid, treat the GPS
# point as outside the area the model has real experience with.
OUT_OF_COVERAGE_KM = 20.0


# --------------------------------------------------------------------------
# Globals populated at startup (see `lifespan` below)
# --------------------------------------------------------------------------
model = None
location_lookup: dict = {}
category_vocab: dict = {}
model_metrics: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, location_lookup, category_vocab, model_metrics

    model_path = os.path.join(ASSET_DIR, "resolution_time_model.joblib")
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"ERROR: could not load model from {model_path}: {e}")
        print("Run train_model.py first to produce this file.")
        model = None

    for name, target in [("location_lookup", "location_lookup.json"),
                          ("category_vocab", "category_vocab.json"),
                          ("model_metrics", "metrics.json")]:
        path = os.path.join(ASSET_DIR, target)
        try:
            with open(path) as f:
                globals()[name] = json.load(f)
            print(f"Loaded {target}")
        except Exception as e:
            print(f"WARNING: could not load {path}: {e}. "
                  f"Re-run train_model.py to regenerate it.")
            globals()[name] = {}

    yield  # app runs here

    # nothing to clean up on shutdown


app = FastAPI(title="Astram Guardian API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Spatial matching: map a live GPS point to the nearest category the model
# actually learned, instead of an arbitrary live place name.
# --------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_known_category(lat: float, lon: float, table: dict) -> tuple:
    """Returns (best_name, distance_km) or (None, None) if table is empty."""
    if not table:
        return None, None
    best_name, best_dist = None, float("inf")
    for name, info in table.items():
        d = haversine_km(lat, lon, info["lat"], info["lon"])
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, round(best_dist, 2)


# --------------------------------------------------------------------------
# Live, human-readable context (display only — never fed to the model)
# --------------------------------------------------------------------------
def get_live_street_data(lat: float, lon: float) -> tuple:
    """Reverse-geocode via OpenStreetMap Nominatim for display purposes."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "json", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1}
        headers = {"User-Agent": "AstramGuardianHackathonApp/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5).json()
        address = resp.get("address", {})
        road = address.get("road", "the nearby road")
        suburb = address.get("suburb", address.get("neighbourhood", "the surrounding area"))
        return road, suburb
    except Exception as e:
        print(f"OSM geocoding failed: {e}")
        return "the affected road", "the local area"


def get_exact_routing_instructions(start_lat, start_lon, end_lat, end_lon) -> list:
    """Turn-by-turn directions via the public OSRM demo server."""
    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{start_lon},{start_lat};{end_lon},{end_lat}")
        resp = requests.get(url, params={"steps": "true", "overview": "false"}, timeout=5).json()
        if resp.get("code") != "Ok":
            return ["Unable to calculate an exact route for these coordinates."]

        steps = resp["routes"][0]["legs"][0]["steps"]
        instructions = []
        for step in steps:
            maneuver = step["maneuver"]["type"]
            modifier = step["maneuver"].get("modifier", "")
            name = step.get("name") or "the road"
            if maneuver == "turn":
                instructions.append(f"Turn {modifier} onto {name}.")
            elif maneuver == "new name":
                instructions.append(f"Continue onto {name}.")
            elif maneuver == "depart":
                instructions.append(f"Start diversion heading {modifier} on {name}.")
            elif maneuver == "arrive":
                instructions.append(f"End of diversion at {name}. Merge traffic back to main corridor.")
        return instructions[:5]
    except Exception as e:
        print(f"OSRM routing failed: {e}")
        return ["Reroute traffic to the nearest parallel arterial road."]


# --------------------------------------------------------------------------
# Dispatch playbook
# --------------------------------------------------------------------------
def generate_playbook(duration_minutes: float, event_cause: str, road_name: str,
                       suburb: str, requires_road_closure: bool) -> list:
    playbook = []

    if duration_minutes > 120:
        playbook.append(f"CRITICAL: Severe bottleneck expected on {road_name}.")
        playbook.append(f"DEPLOYMENT: Send 3 traffic wardens to secure intersections entering {suburb}.")
    elif duration_minutes > 60:
        playbook.append(f"WARNING: Moderate traffic degradation expected on {road_name}.")
        playbook.append("DEPLOYMENT: Dispatch 1 patrol unit to maintain localized flow.")
    else:
        playbook.append(f"MINOR: Localized delay expected in {suburb}. Routine clearing procedures apply.")

    cause_actions = {
        "water_logging": f"ACTION: Alert the municipal drainage department and dispatch pumps to {road_name}.",
        "vehicle_breakdown": "ACTION: Dispatch a heavy-duty tow truck to the exact GPS coordinates.",
        "public_event": f"ACTION: Pre-deploy crowd-control barricades along the {road_name} perimeter.",
        "pot_holes": "ACTION: Log coordinates to the municipal pothole repair queue.",
        "tree_fall": "ACTION: Dispatch a tree-clearing crew with chainsaws and a crane if needed.",
        "accident": "ACTION: Dispatch ambulance and traffic police for scene management.",
    }
    if event_cause in cause_actions:
        playbook.append(cause_actions[event_cause])

    if requires_road_closure:
        playbook.append(f"CLOSURE: Road closure authorized for {road_name} — place barricades at both approach points.")

    return playbook


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class IncidentReport(BaseModel):
    event_cause: str
    veh_type: str = "others"
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    is_planned: bool = False
    reported_by: str = Field("citizen", description="'official' or 'citizen' — drives the model's `authenticated` feature.")
    requires_road_closure: bool = False
    priority: Optional[str] = Field(None, description="'High' or 'Low'. If omitted, inferred from event_cause.")
    detour_start_lat: Optional[float] = None
    detour_start_lon: Optional[float] = None
    detour_end_lat: Optional[float] = None
    detour_end_lon: Optional[float] = None

    @field_validator("event_cause")
    @classmethod
    def validate_event_cause(cls, v):
        valid = category_vocab.get("event_cause", [])
        if valid and v not in valid:
            raise ValueError(f"Unrecognized event_cause '{v}'. Must be one of: {valid}")
        return v

    @field_validator("veh_type")
    @classmethod
    def validate_veh_type(cls, v):
        valid = category_vocab.get("veh_type", [])
        if valid and v not in valid:
            raise ValueError(f"Unrecognized veh_type '{v}'. Must be one of: {valid}")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        if v is None:
            return v
        valid = category_vocab.get("priority", [])
        if valid and v not in valid:
            raise ValueError(f"Unrecognized priority '{v}'. Must be one of: {valid}")
        return v

    @field_validator("reported_by")
    @classmethod
    def validate_reported_by(cls, v):
        if v not in ("official", "citizen"):
            raise ValueError("reported_by must be 'official' or 'citizen'")
        return v


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "message": "Astram Guardian API is live.",
        "model_loaded": model is not None,
        "location_lookup_loaded": bool(location_lookup),
        "category_vocab_loaded": bool(category_vocab),
    }


@app.get("/metadata")
def get_metadata():
    """
    Single source of truth for the frontend: which event causes / vehicle
    types / priorities the model actually understands, plus headline
    model performance, so the dashboard never hardcodes a dropdown list
    that could drift from what the model was trained on.
    """
    return {
        "event_causes": [
            {"value": v, "label": EVENT_CAUSE_LABELS[v]}
            for v in category_vocab.get("event_cause", []) if v in EVENT_CAUSE_LABELS
        ],
        "vehicle_types": [
            {"value": v, "label": VEH_TYPE_LABELS[v]}
            for v in category_vocab.get("veh_type", []) if v in VEH_TYPE_LABELS
        ],
        "no_vehicle_causes": sorted(NO_VEHICLE_CAUSES),
        "priorities": category_vocab.get("priority", []),
        "default_priority_by_cause": category_vocab.get("default_priority_by_cause", {}),
        "model_metrics": model_metrics.get("metrics"),
        "top_feature_importances": model_metrics.get("top_feature_importances"),
    }


@app.post("/predict_impact")
async def predict_impact(incident: IncidentReport):
    if model is None:
        raise HTTPException(status_code=500, detail="AI model not loaded. Run train_model.py first.")

    try:
        # 1. Live, human-readable context (display only)
        road_name, suburb = get_live_street_data(incident.latitude, incident.longitude)

        # 2. Match the GPS point to the nearest category the model actually
        #    learned. This is what gets fed to the model -- NOT road_name/suburb.
        matched_corridor, corridor_dist = nearest_known_category(
            incident.latitude, incident.longitude, location_lookup.get("corridor", {})
        )
        # For suggesting a diversion specifically, exclude "Non-corridor" --
        # it's a catch-all for events not on a named road, so matching to
        # it gives no real road to divert onto. The model still gets the
        # unfiltered `matched_corridor` above; this is only for routing text.
        named_corridors = {k: v for k, v in location_lookup.get("corridor", {}).items() if k != "Non-corridor"}
        nearest_named_corridor, named_corridor_dist = nearest_known_category(
            incident.latitude, incident.longitude, named_corridors
        )
        matched_zone, zone_dist = nearest_known_category(
            incident.latitude, incident.longitude, location_lookup.get("zone", {})
        )
        matched_station, station_dist = nearest_known_category(
            incident.latitude, incident.longitude, location_lookup.get("police_station", {})
        )

        nearest_dist = min(d for d in [corridor_dist, zone_dist, station_dist] if d is not None) \
            if any(d is not None for d in [corridor_dist, zone_dist, station_dist]) else None
        outside_coverage = nearest_dist is not None and nearest_dist > OUT_OF_COVERAGE_KM

        # 3. Resolve priority: explicit override, else the data-driven default for this cause
        priority = incident.priority or category_vocab.get("default_priority_by_cause", {}).get(
            incident.event_cause, "Low"
        )

        # 4. Time features, computed fresh for "right now"
        now = datetime.now()

        # 5. Build the exact feature row the model expects
        input_df = pd.DataFrame([{
            "event_type": "planned" if incident.is_planned else "unplanned",
            "event_cause": incident.event_cause,
            "veh_type": incident.veh_type,
            "priority": priority,
            "corridor": matched_corridor or "unknown",
            "zone": matched_zone or "unknown",
            "police_station": matched_station or "unknown",
            "requires_road_closure": str(incident.requires_road_closure),
            "authenticated": "yes" if incident.reported_by == "official" else "no",
            "latitude": incident.latitude,
            "longitude": incident.longitude,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
        }])

        # 6. Point prediction
        predicted_duration = float(model.predict(input_df)[0])

        # 7. Uncertainty band from the forest's tree-to-tree spread.
        #    (Approximate: each tree's log-scale leaf prediction is
        #    inverse-transformed individually, which is why the median
        #    here can differ slightly from the point prediction above —
        #    that one inverse-transforms the forest's averaged log
        #    prediction instead. Both are legitimate; we surface both.)
        prediction_range = None
        try:
            preprocess = model.named_steps["preprocess"]
            ttr = model.named_steps["model"]
            forest = ttr.regressor_
            X_enc = preprocess.transform(input_df)
            tree_preds_log = np.array([t.predict(X_enc)[0] for t in forest.estimators_])
            tree_preds_minutes = np.expm1(tree_preds_log)
            p10, p50, p90 = np.percentile(tree_preds_minutes, [10, 50, 90])
            prediction_range = {
                "low_estimate_minutes": round(float(p10), 1),
                "median_estimate_minutes": round(float(p50), 1),
                "high_estimate_minutes": round(float(p90), 1),
            }
        except Exception as e:
            print(f"Could not compute prediction interval: {e}")

        # 8. Dispatch playbook
        playbook = generate_playbook(
            predicted_duration, incident.event_cause, road_name, suburb,
            incident.requires_road_closure
        )

        # 9. Routing instruction: exact OSRM turn-by-turn if the user manually
        #    placed detour points, otherwise a concrete default suggestion
        #    from the nearest real named corridor (never the vague
        #    "Non-corridor" catch-all -- see STATIC_DIVERSION_MAP comment).
        if incident.detour_start_lat and incident.detour_end_lat:
            playbook.append("EXACT DIVERSION ROUTE:")
            for step in get_exact_routing_instructions(
                incident.detour_start_lat, incident.detour_start_lon,
                incident.detour_end_lat, incident.detour_end_lon
            ):
                playbook.append(f"   -> {step}")
        elif nearest_named_corridor and nearest_named_corridor in STATIC_DIVERSION_MAP:
            playbook.append(f"SUGGESTED DIVERSION: {STATIC_DIVERSION_MAP[nearest_named_corridor]}")
        else:
            playbook.append(f"ROUTING: No pre-planned diversion on file near {suburb} — map a manual bypass.")

        if outside_coverage:
            playbook.insert(0, f"NOTE: This location is {nearest_dist:.1f} km from the nearest "
                                f"event the model has seen before — treat the prediction with caution.")

        return {
            "status": "success",
            "incident_logged": incident.model_dump(),
            "predicted_duration_minutes": round(predicted_duration, 1),
            "prediction_range": prediction_range,
            "priority_used": priority,
            "context_display": {
                "blocked_road": road_name,
                "affected_area": suburb,
            },
            "context_model_input": {
                "matched_corridor": matched_corridor,
                "matched_corridor_distance_km": corridor_dist,
                "nearest_named_corridor_for_diversion": nearest_named_corridor,
                "nearest_named_corridor_distance_km": named_corridor_dist,
                "matched_zone": matched_zone,
                "matched_zone_distance_km": zone_dist,
                "matched_police_station": matched_station,
                "matched_police_station_distance_km": station_dist,
                "outside_known_coverage_area": outside_coverage,
            },
            "dispatch_playbook": playbook,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Backend error: {e}")
        raise HTTPException(status_code=400, detail=str(e))