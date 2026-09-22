"""CLI entry point for starting 3waPlateAI Web Studio."""
import uvicorn

def main():
    print("[3waPlateAI] 正在啟動 Web 工作台 (http://localhost:1688)...")
    uvicorn.run("plateai_web.app:app", host="0.0.0.0", port=1688, reload=True)

if __name__ == "__main__":
    main()
