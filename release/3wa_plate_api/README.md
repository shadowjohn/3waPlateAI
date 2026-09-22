# 3waPlateAI Standalone Release API (Port 1788)

此發行包為完全獨立之車牌辨識服務，可直接複製到任何專案目錄中使用。

## 快速啟動
1. 雙擊執行 `run_api_1788.bat`。
2. 服務將於 `http://localhost:1788` 啟動。

## API 呼叫範例 (Python)
```python
import requests

url = "http://localhost:1788/api/predict"
files = {"file": open("car.jpg", "rb")}
resp = requests.post(url, files=files)
print(resp.json())
```
