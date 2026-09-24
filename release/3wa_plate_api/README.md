# 3waPlateAI Standalone Release API (Port 1788)

此目錄目前是 source-only API 雛形，**不是可用的車牌辨識發行版**。它不附模型權重或 Bundle；`POST /api/predict` 對有效圖片會回傳 HTTP 503，直到接上真實 Reader。請勿將它的回應當成辨識結果。

原始碼採 MIT 授權；第三方模型、資料、字型與執行環境不因此取得 MIT 授權。詳見 `LICENSE` 與 `THIRD_PARTY_NOTICES.md`。

## 快速啟動
1. 雙擊執行 `run_api_1788.bat`。
2. 服務將於 `http://localhost:1788` 啟動。
3. 真實推論服務尚待整合與驗證；模型須由使用者自行合法取得或訓練，不會由打包器複製。

## API 呼叫範例 (Python)
```python
import requests

url = "http://localhost:1788/api/predict"
files = {"file": open("car.jpg", "rb")}
resp = requests.post(url, files=files)
print(resp.json())
```
