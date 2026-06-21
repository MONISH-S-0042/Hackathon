# Astram Guardian — Complete Submission Guide

## What's in this folder

```
Hackathon/
├── train_model.py          # Step 1 — trains the AI model
├── main.py                 # Step 2 — FastAPI backend (serves model + playbook)
├── index.html              # Step 3 — frontend dashboard
├── requirements.txt        # Python dependencies
│
├── output/                 # Auto-created by train_model.py
│   ├── resolution_time_model.joblib
│   ├── location_lookup.json
│   ├── category_vocab.json
│   └── metrics.json
│
└── Astram event data_anonymized - ....csv   ← your dataset
```

---

## How to Run Locally (Windows)

### 1 — Install Python dependencies
```powershell
pip install fastapi uvicorn pydantic joblib pandas numpy requests scikit-learn matplotlib
```

### 2 — Train the model
Put `train_model.py` and the CSV in the same folder, then:
```powershell
python train_model.py
```
This creates the `output/` folder with the model and lookup files.
Takes about 2–3 minutes on first run.

### 3 — Start the backend
```powershell
uvicorn main:app --reload
```
You should see:
```
✅ Astram Guardian AI Model loaded successfully.
INFO:  Uvicorn running on http://127.0.0.1:8000
```

### 4 — Open the dashboard
Open `index.html` directly in your browser (double-click it, or drag it in).
Click anywhere on the Bengaluru map → fill in the form → hit **Run AI Predictive Analysis**.

---
