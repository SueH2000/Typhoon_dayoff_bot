# Typhoon Day-off Bot
A minimal repository to showcase the LINE bot flow and the ML pipeline — with a detailed explanation.

- **UI**: Region → City → Station quick replies in LINE (`src/linebot_typhoon.py`)
- **Crawling**: Fetch live CWB O-A0003-001 station observations by name
- **Processing**: KNN impute (13 cols) → add typhoon meta features → MinMax scale (24 cols, ordered)
- **Predicting**: RandomForest probability, messaged back with graded text

> This repo intentionally stays **minimal** (no Docker, no CI, no extra scaffolding). Ask me if you want the deployment templates.

---

## Repo layout

```
typhoon-dayoff-bot-showcase/
├─ src/
│  ├─ linebot_typhoon.py   # LINE webhook + UI flow
│  ├─ predict.py           # Inference pipeline (impute 13 → scale 24 → RF proba)
│  └─ train_model.py       # Training aligned to the artifact/feature contract
├─ models/                 # put your 3 artifacts here (not committed)
├─ .gitignore
├─ requirements.txt
└─ README.md
```

**Artifacts used at runtime**  
Place these  inside `models/`:
- `kNN_imputer.joblib`
- `MMscaler.joblib`
- `rf_model.joblib` ##too large to upload

`src/predict.py` can read paths from environment variables if you prefer, e.g.:
```
KNN_IMPUTER_PATH=models/kNN_imputer.joblib
MINMAX_SCALER_PATH=models/MMscaler.joblib
MODEL_PATH=models/rf_model.joblib
```

---

## Feature contract (critical)

To reproduce the same behavior between **training** and **inference**, the model pipeline expects these exact columns/orders:

- **KNNImputer (13 columns, specific order)**  
  ```
  ['Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
   'WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y','lat','lon']
  ```

- **MinMaxScaler (24 columns, ORDERED)**  
  ```
  ['Dayoff','Precp','RH','StnHeight','StnPres','T.Max','T.Min','Temperature',
   'TyWS','WDGust_vector_x','WDGust_vector_y','WD_vector_x','WD_vector_y',
   'X10_radius','X7_radius','alert_num','born_spotE','born_spotN','hpa',
   'lat','lon','route_--','route_2','route_3']
  ```

The predictor imputes the 13, reinserts them, scales the 24 **in that order**, then predicts.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to run the LINE bot locally, you’ll need to set environment variables for your secrets (see below).

---

## Run the LINE bot locally (optional)

> This is optional for the showcase. Running a live bot requires LINE credentials and your app reachable over HTTPS.

1) Set environment variables (examples):
```bash
export LINE_CHANNEL_ACCESS_TOKEN=xxx
export LINE_CHANNEL_SECRET=xxx
export CWB_API_KEY=CWB-xxxx
# (optional if not using defaults)
export KNN_IMPUTER_PATH=models/kNN_imputer.joblib
export MINMAX_SCALER_PATH=models/MMscaler.joblib
export MODEL_PATH=models/rf_model.joblib
```

2) Start Flask:
```bash
export FLASK_ENV=development
python -m flask --app src.linebot_typhoon:app run --port 5000
```

3) Expose your local server to the public (choose one):
- `ngrok http 5000`
- `cloudflared tunnel --url http://localhost:5000`

4) In LINE Developers Console, set the webhook to:
```
https://<your-public-url>/callback
```

---

## Training (recreate artifacts)

Put your dataset (e.g., `data_ver_4_DCT.xlsx`) somewhere accessible and run:

```bash
python src/train_model.py   --data /path/to/data_ver_4_DCT.xlsx   --label TmrDayoff   --output-dir models
```

This writes:
- `models/kNN_imputer.joblib`
- `models/MMscaler.joblib`
- `models/rf_model.joblib`

---

## Note on Heroku (free tier)

Heroku’s legacy **free dynos are no longer available**. If your LINE bot used to run on Heroku’s free tier, you’ll need to move to another host or a paid plan.

### Alternatives (no templates included here to stay minimal)
- **Render.com** — Heroku-like UX; run a Python web service and set your env vars.
- **Railway.app** — simple deploy from GitHub with env vars.
- **Google Cloud Run (Docker)** — build a container and deploy; autoscaling and HTTPS.
- **Fly.io (Docker)** — global edge nodes; deploy your container close to users.

If you want, I can add a **Dockerfile** and a one-pager for one of these platforms later.

---

## Security notes

- Do **not** commit LINE tokens or your `CWB_API_KEY`. Use environment variables (or a local `.env` that is `.gitignore`’d).
- The LINE handler should return fast (your logic already does a single CWB fetch + model inference).

---

## What to read first

- `src/linebot_typhoon.py` — user interaction & CWB fetch path (the “crawling” step).
- `src/predict.py` — the exact inference steps and feature alignment.
- `src/train_model.py` — how the artifacts are trained to match inference.



---

## License & Attribution

This repository uses **Apache-2.0** so that attribution is carried in redistributions.

- The full license text is in `LICENSE`.
- An attribution **NOTICE** is included in `NOTICE`. If someone redistributes your software or builds on it,
  Apache-2.0 requires that they **preserve the NOTICE** content in their distribution (e.g., a NOTICE file or docs).
- If you want to add headers to source files, use this template:

```python
# Copyright 2025 [Sue Hsiung]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
```
