# Typhoon Day-off Bot
A minimal repository to showcase the LINE bot flow and the ML pipeline — with a detailed explanation.

- **UI**: Region → City → Station quick replies in LINE (`src/linebot_typhoon.py`)
- **Crawling**: Fetch live CWB O-A0003-001 station observations by name
- **Processing**: KNN impute (13 cols) → add typhoon meta features → MinMax scale (24 cols, ordered)
- **Predicting**: RandomForest probability, messaged back with graded text

> This repo intentionally stays **minimal** (no Docker, no CI, no extra scaffolding).
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

## Training and Artifacts

- Flags (see `python src/train_model.py -h`):
  - `--data PATH` (required)
  - `--label NAME` default: `TmrDayoff`
  - `--test-size FLOAT` default: `0.2`
  - `--output-dir DIR` default: `.`
  - `--seed INT` default: `42`

- Outputs written to `--output-dir`:
  - `kNN_imputer.joblib`
  - `MMscaler.joblib`
  - `rf_model.joblib`
  - `rf_metrics_aligned.json` (holdout metrics)
  - `artifacts_meta.json` (versions, seed, label, feature schema)

- Predictor artifact discovery (`src/predict.py`):
  - Defaults to `models/` if present
  - Env overrides: `KNN_IMPUTER_PATH`, `MINMAX_SCALER_PATH`, `MODEL_PATH`

- Example (write artifacts into `models/`):
```bash
python src/train_model.py \
  --data /path/to/data_ver_4_DCT.xlsx \
  --label TmrDayoff \
  --output-dir models \
  --seed 42
```

---

### Quick score a 1-row CSV

Your CSV should have the predictor's expected columns (see the Feature contract / SCALER_COLS). See `examples/row.csv` for a ready-to-use template. Example usage:

```python
# score_csv.py
import pandas as pd
from src.predict import Predictor

df = pd.read_csv('row.csv')      # 1 row with required columns
proba = Predictor().predict_proba_dayoff(df)
print(f"P(dayoff)= {proba:.3f}")
```

Run it (artifacts discovered under `models/` by default):
```bash
python score_csv.py
```

Or as a one-liner:
```bash
python - <<'PY'
import pandas as pd
from src.predict import Predictor
df = pd.read_csv('row.csv')
print(Predictor().predict_proba_dayoff(df))
PY
```

Note: If your CSV lacks some columns, fill them before scoring or construct the row with helper functions (e.g., fetch station obs and merge typhoon meta, then pass to `Predictor`).


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


## 使用你本機已訓練的 KNN / 模型檔（重點）

如果你已經在電腦本地訓練好模型，這個專案不需要重新訓練，只要把路徑接上即可。

1) 先準備三個 artifact（可放任意路徑）
- `kNN_imputer.joblib`
- `MMscaler.joblib`
- `rf_model.joblib`

2) 啟動前設定環境變數（用你自己的實際路徑）
```bash
export KNN_IMPUTER_PATH=/absolute/path/to/kNN_imputer.joblib
export MINMAX_SCALER_PATH=/absolute/path/to/MMscaler.joblib
export MODEL_PATH=/absolute/path/to/rf_model.joblib
```

3) 啟動 Flask app
```bash
python -m flask --app src.linebot_typhoon:app run --port 5000
```

> 初學者說明：
> - `KNN_IMPUTER_PATH`：缺值補齊模型
> - `MINMAX_SCALER_PATH`：把特徵縮放到訓練時相同範圍
> - `MODEL_PATH`：最後輸出放假機率的分類模型
> 三者順序不能亂，因為推論流程是：補值 → 合併特徵 → 縮放 → 預測。

---

## 手機 App 化（取代 LINE UI）的後端接口


### 3) 手機端 UI（Flask 內建）

如果你想先快速看到「手機風格畫面」，不用先寫 Flutter/React Native，
啟動 Flask 後直接打開：

```text
http://localhost:5000/app/ui
```

這個頁面會：
- 呼叫 `GET /app/locations` 載入測站下拉選單
- 呼叫 `GET /app/predict?locationName=...` 顯示溫度、雨量、天氣、機率與建議

> 初學者理解：
> - 這是一個「可直接跑」的手機 UI 雛形（前端在 HTML/JS，後端在 Flask）。
> - 等你之後改成原生手機 App，只要沿用同兩個 API 即可，邏輯不用重寫。


目前已新增手機 app 可直接呼叫的 API。你可以用 Flutter / React Native / iOS / Android 直接打這兩個端點：

### 1) 取得可選測站清單
```http
GET /app/locations
```
回傳範例：
```json
{
  "ok": true,
  "locations": ["臺北", "淡水", "高雄", "...省略..."]
}
```

### 2) 取得某測站最新天氣 + 放假機率
```http
GET /app/predict?locationName=臺北
```
回傳範例：
```json
{
  "ok": true,
  "locationName": "臺北",
  "weather": {
    "temperature_c": "29.4",
    "rainfall_mm": "0.0",
    "description": "晴"
  },
  "dayoff_probability": 23.1,
  "advice": "明天不太可能放颱風假哦！"
}
```

> 初學者說明：
> - 手機 app 只要負責「選地點 + 顯示 JSON」。
> - 預測邏輯留在後端（Flask），這樣模型不會暴露在手機端，也比較好維護。
> - 你可以先做最小 UI：一個下拉選單 + 一個查詢按鈕 + 三個文字欄位（溫度/雨量/結果）。


### 4) 我完全沒做過手機 App，直接用現成 Flutter 範本

已提供完整範本在 `mobile_app_flutter/`，你可以直接跑起來。

快速步驟：

```bash
cd mobile_app_flutter
flutter pub get
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000
```

詳細說明請看：`mobile_app_flutter/README.md`。


## 新 repo 上線清單

已提供完整上線清單文件：`NEW_REPO_LAUNCH_CHECKLIST.md`。

內容包含：
- repo 名稱建議
- README 首頁模板
- 第一版 tag 命名（SemVer）
- 發佈順序與 release note 範本
