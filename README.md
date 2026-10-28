# PetroSite 3D

Interactive educational oilfield simulation: 3D rig and wellbore visualisation, mud-circulation hydraulics,
a drilling simulator, wireline and coiled-tubing service scenes, and a synthetic-data anomaly-detection demo.

> **Educational simulation.** Simplified physics, illustrative coefficients and synthetic data. Not certified
> engineering software, not field-calibrated, and not affiliated with or endorsed by any oilfield company.

## Run on Windows (PowerShell)

Requires Python 3.11 or newer (developed and tested on 3.13).

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    streamlit run app.py

or simply `.\run.ps1`. The app opens at http://localhost:8501.

## Run with Docker

    docker build -t petrosite .
    docker run -p 8501:8501 petrosite

The image uses `requirements.lock.txt` (exact tested versions), runs as a non-root user and exposes a health check at
`/_stcore/health`. *(The Dockerfile was written for this release but not built in the environment where the app was tested.)*

## Modules

| Module | What it does |
|---|---|
| Operations Center | Live summary of the Drilling Simulator state, anomaly status, torque/SPP trends |
| 3D Rig Explorer | Procedural rig with 21 equipment items (power house, solids control, BOP control unit, choke manifold, pipe rack, cabins, tanks, racked stands) and a detailed PDC / tricone bit; select equipment for purpose, components, variables, explanation; camera presets; visibility |
| Wellbore Cutaway | Geology, casing, cement, string, annulus, mud flow; cutaway and labelled vertical exaggeration |
| Mud Circulation Lab | Full circulation loop with animated markers; hydrostatic pressure, annular velocity, friction losses, SPP, ECD |
| Drilling Simulator | Deterministic model with layered lithology, bit wear, MSE, Bingham / power-law rheology, cuttings loading, pore/fracture window, kick/loss flow and pit gain, friction and stick-slip-like disturbances |
| Well Path & Torque-Drag | Build-and-hold minimum-curvature path, Johancsik soft-string torque & drag (pick-up / slack-off / rotating / drilling), buckling screen |
| Well Control | Kill sheet: KMW, ICP/FCP, MAASP, influx height and type, pressure schedule |
| SRP Lab | Sucker-rod pump: four-bar kinematics, damped wave-equation rod string, surface and pump dynamometer cards, torque, loads, Vogel inflow, pump-off, animated 3D unit |
| Reservoir Simulation | 2-D two-phase waterflood (IMPES): heterogeneous permeability, Corey relperm, breakthrough, recovery, saturation/pressure maps, Welge check |
| ESP Lab | Electric submersible pump: generic stage curves, affinity laws, system curve with Vogel inflow, gas derating, motor/cable sizing, VSD sweep, 3D completion |
| Wireline Operations | Unit, reel, lubricator, tool; RIH/POOH kinematics with indicative cable tension |
| Coiled Tubing Lab | Reel, gooseneck, injector, BOP; RIH/POOH with assumed lock-up drag factor |
| AI Diagnostics | Isolation Forest vs static sigma baseline on synthetic labelled runs, with precision/recall/F1/FPR/confusion matrix |
| Scenario Comparison | Two independent runs, overlaid charts, parameter differences, CSV export |
| Engineering Reference | Equations, units, assumptions, limitations, and live self-checks |

## Tests

    python -m pip install pytest
    python -m pytest -q

The suite covers the hydraulics equations and validation, simulator behaviour, ML reproducibility / leakage / baseline
comparison, and builds every 3D figure at its edge cases.

## Assumptions and limitations

- Hydraulics: Newtonian, Bingham plastic or power-law fluid, slot-approximation annulus; no pipe rotation/eccentricity,
  surge/swab or temperature effects. Cuttings transport uses a single-particle slip velocity (no bed formation).
- SRP: explicit 1-D wave solution with illustrative damping and valve logic; no tubing movement, deviation or fluid inertia; not validated against dynamometer data. Unit ratings are generic.
- ESP: generic normalised pump curves (not a vendor catalogue); gas handling is an illustrative derating; no thermal or shaft-limit checks.
- Reservoir: incompressible two-phase waterflood, no gravity/capillarity, one injector/producer; teaching model, not history-matched.
- Bits and rig equipment are representative schematic geometry, not manufacturer designs.
- Torque & drag: soft-string model (no stiffness); buckling is a threshold screen.
- Well control: static kill-sheet arithmetic; no gas migration or choke dynamics.
- Drilling model: ROP and torque coefficients are illustrative teaching values, not a universal law and not calibrated.
- ML: data and anomalies are synthetic and come from the same generator that defines the features, so scores are optimistic
  versus field data. Isolation Forest scores are rankings, not failure probabilities. The feature set and 98th-percentile
  threshold were chosen on a separate validation set, not the displayed test runs.
- Geometry is procedural and schematic (not to scale; underground sections shortened or exaggerated and labelled).
- Wireline / coiled tubing are kinematic teaching models without cable strength, buckling, lock-up physics or pressure-control constraints.
- State is per browser session and in memory; there is no database, authentication or persistence.


## Latest update
- Reservoir module is now an **oil** reservoir (undersaturated oil, structure, OWC, edge aquifer, depletion or waterflood, oil-saturation maps and 3D structure). Educational; no gravity, capillarity or free gas.
- Realistic animated sucker-rod pump unit with downhole inset and camera presets.
- Graphics quality selector (Standard / High / Ultra); figures are cached and animations run in the browser to keep the app light.
- 'How to read this page' panel on every page.
