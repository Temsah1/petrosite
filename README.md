Interactive educational oilfield simulation: 3D rig and wellbore visualisation, mud-circulation hydraulics, a drilling simulator, wireline and coiled-tubing service scenes, and a synthetic-data anomaly-detection demo.
Educational simulation. Simplified physics, illustrative coefficients and synthetic data. Not certified engineering software, not field-calibrated, and not affiliated with or endorsed by any oilfield company.
Requires Python 3.11 or newer (developed and tested on 3.13).
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py

```
or simply .\run.ps1. The app opens at http://localhost:8501.
```
docker build -t petrosite .
docker run -p 8501:8501 petrosite

```
The image uses requirements.lock.txt (exact tested versions), runs as a non-root user and exposes a health check at /_stcore/health. (The Dockerfile was written for this release but not built in the environment where the app was tested.)
```
python -m pip install pytest
python -m pytest -q

```
The suite covers the hydraulics equations and validation, simulator behaviour, ML reproducibility / leakage / baseline comparison, and builds every 3D figure at its edge cases.
- Hydraulics: Newtonian, Bingham plastic or power-law fluid, slot-approximation annulus; no pipe rotation/eccentricity, surge/swab or temperature effects. Cuttings transport uses a single-particle slip velocity (no bed formation).
- SRP: explicit 1-D wave solution with illustrative damping and valve logic; no tubing movement, deviation or fluid inertia; not validated against dynamometer data. Unit ratings are generic.
- ESP: generic normalised pump curves (not a vendor catalogue); gas handling is an illustrative derating; no thermal or shaft-limit checks.
- Reservoir: incompressible two-phase waterflood, no gravity/capillarity, one injector/producer; teaching model, not history-matched.
- Bits and rig equipment are representative schematic geometry, not manufacturer designs.
- Torque & drag: soft-string model (no stiffness); buckling is a threshold screen.
- Well control: static kill-sheet arithmetic; no gas migration or choke dynamics.
- Drilling model: ROP and torque coefficients are illustrative teaching values, not a universal law and not calibrated.
- ML: data and anomalies are synthetic and come from the same generator that defines the features, so scores are optimistic versus field data. Isolation Forest scores are rankings, not failure probabilities. The feature set and 98th-percentile threshold were chosen on a separate validation set, not the displayed test runs.
- Geometry is procedural and schematic (not to scale; underground sections shortened or exaggerated and labelled).
- Wireline / coiled tubing are kinematic teaching models without cable strength, buckling, lock-up physics or pressure-control constraints.
- State is per browser session and in memory; there is no database, authentication or persistence.
- Reservoir module is now an oil reservoir (undersaturated oil, structure, OWC, edge aquifer, depletion or waterflood, oil-saturation maps and 3D structure). Educational; no gravity, capillarity or free gas.
- Realistic animated sucker-rod pump unit with downhole inset and camera presets.
- Graphics quality selector (Standard / High / Ultra); figures are cached and animations run in the browser to keep the app light.
- 'How to read this page' panel on every page.
