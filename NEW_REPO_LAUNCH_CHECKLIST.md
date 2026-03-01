# 新 Repository 上線清單（Typhoon Dayoff Mobile 版）

> 目標：讓你把目前成果「安全、可維護、可擴充」地發布到新 repo。
> 特色：包含 repo 命名、README 首頁模板、v1 tag 命名、發佈順序。

---

## 0) 先理解：為什麼要拆新 repo？（初學者版）

拆新 repo 的核心價值：

1. **降低複雜度**：LINE Bot 與手機 App/API 分開，維護更清楚。  
2. **發版更穩**：手機前端與後端可各自迭代，不互相綁死。  
3. **協作更容易**：未來找前端同學協作時，專案邊界明確。  

---

## 1) Repo 名稱建議（可直接選一個）

### A. 清楚描述功能（推薦）
- `typhoon-dayoff-mobile-backend`
- `typhoon-dayoff-mobile-app`
- `typhoon-dayoff-platform`

### B. 若你打算前後端分兩個 repo
- 後端：`typhoon-dayoff-api`
- 手機端：`typhoon-dayoff-flutter`

### C. 個人品牌風格（可讀性高）
- `sue-typhoon-dayoff-api`
- `sue-typhoon-dayoff-mobile`

> 命名原則（給初學者）：
> - 全小寫 + `-`，不要空格。
> - 名稱裡要包含「dayoff / typhoon / mobile / api」這類關鍵字，未來搜尋比較好找。

---

## 2) README 首頁模板（可直接貼到新 repo）

你可以把下面整段貼到新 repo 的 `README.md`，再替換 `<...>` 欄位。

```md
# <Project Name>

台灣颱風假機率預測系統（手機端 + Flask API）。

## 功能
- 依測站查詢最新氣象資訊
- 預測明日放颱風假機率
- 提供手機 App（Flutter）與 API 串接範例

## 系統架構
- Backend: Flask (`src/linebot_typhoon.py`)
- Mobile: Flutter (`mobile_app_flutter/`)
- Model artifacts:
  - `kNN_imputer.joblib`
  - `MMscaler.joblib`
  - `rf_model.joblib`

## 快速開始

### 1) 後端啟動
```bash
python -m flask --app src.linebot_typhoon:app run --host 0.0.0.0 --port 5000
```

### 2) 手機端啟動（Flutter）
```bash
cd mobile_app_flutter
flutter pub get
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000
```

## 環境變數
- `KNN_IMPUTER_PATH`
- `MINMAX_SCALER_PATH`
- `MODEL_PATH`
- `CWA_API_KEY`（或 `CWB_API_KEY`）

## API
- `GET /app/locations`
- `GET /app/predict?locationName=臺北`
- `GET /app/ui`

## 版本
- 當前穩定版：`v1.0.0`

## License
Apache-2.0
```

---

## 3) 第一版 Tag 命名建議

### 推薦方案：Semantic Versioning（語意化版本）
- 第一版正式上線：`v1.0.0`
- 修 bug（不破壞介面）：`v1.0.1`
- 新增功能（相容）：`v1.1.0`
- 破壞性變更：`v2.0.0`

### 你的第一版建議
- **Tag**：`v1.0.0`
- **Release title**：`First production release: mobile UI + Flask API + model integration`

> 初學者重點：
> `v1.0.0` 代表你定義了「可用產品基線」。之後所有修改都能對照這個基線追蹤風險。

---

## 4) 發佈順序（一步一步照做）

### Phase 1：建立新 repo
1. 在 GitHub 建立空 repo（不要先加 README，避免衝突）。
2. 本機新增 remote 並推上去。

```bash
git remote add new-origin <你的新repo網址>
git push new-origin work:main
```

### Phase 2：補齊專案資訊
1. 設定 repo Description（例如：`Typhoon dayoff prediction API + Flutter app`）。
2. 設定 Topics：`flask`, `flutter`, `machine-learning`, `weather-api`, `taiwan`。
3. 確認 `LICENSE`、`NOTICE`、`README` 都在。

### Phase 3：設定 Secrets / 環境
1. 將 API key 與模型路徑改用環境變數。
2. 不要把真實金鑰 commit 到 repo。

### Phase 4：建立第一個 Release
1. 建立 tag：
```bash
git tag v1.0.0
git push new-origin v1.0.0
```
2. 到 GitHub Releases 建立 `v1.0.0` release note。  
3. 內容重點：功能、已知限制、下版計畫。

### Phase 5：上線後驗收（最小檢查）
1. `GET /app/locations` 可回 200。
2. `GET /app/predict?locationName=臺北` 可回 200 且包含機率。
3. Flutter app 可顯示測站與預測資訊。

---

## 5) 建議的 GitHub Release Note（v1.0.0）模板

```md
## v1.0.0 - First stable mobile release

### Included
- Flask API for station list and prediction
- Mobile-friendly web UI (`/app/ui`)
- Flutter sample app (`mobile_app_flutter/`)
- Local model artifact integration via env vars

### API Endpoints
- `GET /app/locations`
- `GET /app/predict?locationName=...`
- `GET /app/ui`

### Known limitations
- Need valid weather API key
- Need local model artifacts configured

### Next
- Add auth / rate limit
- Add CI test workflow
- Add Docker deployment docs
```

---

## 6) 初學者常見錯誤清單（避免踩雷）

1. **把模型檔直接 commit**（檔案太大 + 安全性風險）。
2. **Flutter 連不到後端**（base URL 用錯，Android 模擬器不是 localhost）。
3. **只測 happy path**（沒測 API 錯誤回應）。
4. **沒有打 tag 就上線**（日後難追版本問題）。

---

## 7) 一鍵檢查指令（上線前）

```bash
# 確認工作樹乾淨
git status

# 看當前分支與提交
git log --oneline -n 5

# 確認 tag
git tag

# 檢查關鍵檔案是否存在
ls README.md LICENSE NOTICE
ls mobile_app_flutter
```

---

如果你願意，我下一步可以再幫你產出：
1) **新 repo 的最終 README（已填好專案名稱）**  
2) **v1.0.0 Release Note 完整中文稿**  
3) **GitHub 上線操作清單（截圖導向版）**
