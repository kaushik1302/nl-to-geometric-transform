## Setup
1. Clone, then from the repo root:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -e .
2. VS Code: Ctrl+Shift+P → Python: Select Interpreter → .venv
3. Verify: python tools\smoke_test.py