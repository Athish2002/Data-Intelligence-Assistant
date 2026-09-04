"""
run_fullstack.py
────────────────
Launcher script for the Fullstack Edition of Data Intelligence Assistant.
Runs the FastAPI REST backend + Glassmorphic Web Client on http://localhost:8000
Interactive Swagger Docs at http://localhost:8000/docs
"""

import uvicorn
from dia.config import HOST, PORT

if __name__ == "__main__":
    display_host = "localhost" if HOST in ("0.0.0.0", "127.0.0.1") else HOST
    print(f"[DIA] Starting Data Intelligence Assistant (Fullstack Edition)...")
    print(f"[DIA] Web App:      http://{display_host}:{PORT}")
    print(f"[DIA] Swagger Docs: http://{display_host}:{PORT}/docs")
    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=False)
