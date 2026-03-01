# Flutter 手機 App（最小可執行版）

這個資料夾提供一個可以直接呼叫後端 Flask API 的手機 App 範例。

## 1) 安裝 Flutter
先安裝 Flutter SDK：<https://docs.flutter.dev/get-started/install>

## 2) 啟動後端 Flask
在專案根目錄啟動：

```bash
python -m flask --app src.linebot_typhoon:app run --host 0.0.0.0 --port 5000
```

## 3) 啟動手機 App
進入 `mobile_app_flutter` 後執行：

```bash
flutter pub get
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000
```

### 常見連線位址
- Android 模擬器：`http://10.0.2.2:5000`
- iOS 模擬器：`http://localhost:5000`
- 實機：`http://你的電腦區網IP:5000`（例如 `http://192.168.1.10:5000`）

## 4) 目前功能
- 載入測站清單（`GET /app/locations`）
- 依測站查詢預測（`GET /app/predict?locationName=...`）
- 顯示：溫度、雨量、天氣、放假機率、建議文字

## 初學者重點
- 手機 App 只做畫面與 API 呼叫。
- 預測模型留在 Flask 後端，避免把模型檔放到手機上。
- 這樣維護簡單：模型更新只要改後端，手機 App 不一定要重上架。
