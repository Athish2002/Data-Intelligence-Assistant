"""
run_streamlit.py
────────────────
Launcher script for the Streamlit Edition of Data Intelligence Assistant.
Runs the interactive dashboard on http://localhost:8501
"""

import subprocess
import sys

if __name__ == "__main__":
    print("[DIA] Starting Data Intelligence Assistant (Streamlit Edition)...")
    print("[DIA] URL: http://localhost:8501")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "false"]
    subprocess.run(cmd)
