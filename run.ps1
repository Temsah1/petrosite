# One-step setup and launch on Windows PowerShell
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
