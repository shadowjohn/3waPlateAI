@echo off
chcp 65001 >nul
echo ========================================================
echo   3waPlateAI 獨立車牌辨識 API 服務
echo   即將在 http://localhost:1788 啟動 (一起發發)
echo ========================================================
python -m uvicorn app:app --host 0.0.0.0 --port 1788 --reload
pause
