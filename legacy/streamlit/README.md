# ClaimGuard AI Streamlit Dashboard

The repository's Next.js application is the primary production UI. This Streamlit dashboard is also supported as an interactive analytics and demo experience. Its Plotly charts provide hover details, zoom, pan, and legend controls.

## Run from VS Code on Windows

Open a PowerShell terminal at the repository root and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r legacy/streamlit/requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

Open <http://localhost:8501>. The root `streamlit_app.py` launches `legacy/streamlit/app.py`, while running `streamlit run legacy/streamlit/app.py` from the repository root also works.

## Dependencies

The dashboard-specific packages are listed in [`requirements.txt`](requirements.txt). They include Streamlit, Plotly, pandas, scikit-learn, XGBoost, LightGBM, and SHAP.
