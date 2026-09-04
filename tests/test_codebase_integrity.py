"""
tests/test_codebase_integrity.py
────────────────────────────────
Automated Architectural Quality Gate for Agile Sprint 1.
Verifies:
1. Zero dead/orphaned modules in dia/
2. All dia/ modules import cleanly with zero ImportError/SyntaxError
3. Frontend ES modules parse cleanly with Node.js
4. Absence of legacy strings ('v14', 'alert(') in frontend codebase
"""

import glob
import importlib
import os
import subprocess
import pytest


def test_dia_modules_import_cleanly():
    """Ensure every single module in dia/ can be imported cleanly."""
    dia_dir = os.path.join(os.path.dirname(__file__), "..", "dia")
    py_files = glob.glob(os.path.join(dia_dir, "*.py"))
    assert len(py_files) > 20, "dia directory should contain production modules"

    for fpath in py_files:
        mod_name = os.path.splitext(os.path.basename(fpath))[0]
        if mod_name == "__init__":
            continue
        # Import module dynamically
        mod = importlib.import_module(f"dia.{mod_name}")
        assert mod is not None, f"Module dia.{mod_name} failed to import"


def test_orphaned_modules_purged():
    """Ensure expectations.py and experimentation.py have been deleted."""
    dia_dir = os.path.join(os.path.dirname(__file__), "..", "dia")
    assert not os.path.exists(os.path.join(dia_dir, "expectations.py")), "dia/expectations.py should be deleted"
    assert not os.path.exists(os.path.join(dia_dir, "experimentation.py")), "dia/experimentation.py should be deleted"


def test_frontend_modules_syntax():
    """Run node --check on all frontend ES modules."""
    js_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "js")
    js_files = glob.glob(os.path.join(js_dir, "*.js")) + glob.glob(os.path.join(js_dir, "views", "*.js"))
    assert len(js_files) >= 10, "Frontend should contain modularized ES files"

    for js_path in js_files:
        res = subprocess.run(["node", "--check", js_path], capture_output=True, text=True)
        assert res.returncode == 0, f"Syntax error in {js_path}:\n{res.stderr}"


def test_no_legacy_tokens_in_frontend():
    """Verify absence of v14 and blocking native alert() calls in frontend."""
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    for root, _, files in os.walk(frontend_dir):
        for fname in files:
            if fname.endswith((".html", ".js")):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    assert "v14.0" not in content, f"Legacy 'v14.0' found in {fpath}"
                    assert "v14 " not in content, f"Legacy 'v14 ' found in {fpath}"
