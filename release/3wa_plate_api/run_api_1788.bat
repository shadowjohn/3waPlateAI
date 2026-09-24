@echo off
chcp 65001 >nul
echo ========================================================
echo   3waPlateAI 獨立 API 雛形 (尚無真實推論)
echo   即將在 http://localhost:1788 啟動 (一起發發)
echo ========================================================
python -m uvicorn app:app --host 0.0.0.0 --port 1788 --reload
pause
