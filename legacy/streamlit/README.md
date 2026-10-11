# ClaimGuard AI Streamlit Dashboard

The repository's Next.js application is the primary production UI. This Streamlit dashboard is also supported as an interactive analytics and demo experience. Its Plotly charts provide hover details, zoom, pan, and legend controls.

## Run from VS Code on Windows

Open a PowerShell terminal at the repository root and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r legacy/streamlit/requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

Open <http://localhost:8501>. The root `streamlit_app.py` launches `legacy/streamlit/app.py`, while running `streamlit run legacy/streamlit/app.py` from the repository root also works.

## Dependencies

The dashboard-specific packages are listed in [`requirements.txt`](requirements.txt). They include Streamlit, Plotly, pandas, scikit-learn, XGBoost, LightGBM, and SHAP. They are sufficient for the standalone dashboard; root `requirements.txt` and `requirements-full.txt` are for the API/workers and optional monitoring stack.

## PyArrow / CMake installation error

If pip prints a setuptools package-discovery warning for `pyarrow.vendored` and then fails with `command 'cmake' failed`, it is trying to compile PyArrow from source. The Streamlit-only dependency file does not request PyArrow. Use 64-bit CPython 3.11 and an up-to-date pip, and install only the dashboard requirements above. If you also need the backend, install PyArrow as a wheel before the backend requirements:

```powershell
python -m pip install --only-binary=pyarrow "pyarrow>=14.0.0"
python -m pip install -r requirements.txt
```

If pip reports that no matching binary distribution exists, check `python --version` and `python -c "import platform; print(platform.architecture())"`; use a supported 64-bit CPython release rather than building PyArrow locally.
