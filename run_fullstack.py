"""
run_fullstack.py
────────────────
Launcher script for the Fullstack Edition of Data Intelligence Assistant.
Runs the FastAPI REST backend + Glassmorphic Web Client on http://localhost:8000
Interactive Swagger Docs at http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    print("[DIA] Starting Data Intelligence Assistant (Fullstack Edition)...")
    print("[DIA] Web App:      http://localhost:8000")
    print("[DIA] Swagger Docs: http://localhost:8000/docs")
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
