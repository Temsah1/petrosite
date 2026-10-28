"""
PetroSite 3D - interactive educational oilfield simulation platform.

Single-file Streamlit application.  Sections (in order):
    1. Configuration, theme, constants
    2. Engineering calculations (units documented, input validation)
    3. Drilling simulator model (deterministic, illustrative coefficients)
    4. Machine learning (synthetic data, Isolation Forest, baseline detector)
    5. 3D geometry (procedural Plotly meshes / lines)
    6. UI helpers and session-state management
    7. Pages
    8. Main / navigation

EDUCATIONAL SIMULATION ONLY.  Simplified physics, illustrative coefficients,
synthetic data.  Not certified engineering software and not affiliated with or
endorsed by any oilfield company.
"""
from __future__ import annotations

import bisect
import functools
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

LOG = logging.getLogger("petrosite")

# =============================================================================
# 1. CONFIGURATION, THEME, CONSTANTS
# =============================================================================
PAPER = "#f3f0e8"      # page background (warm engineering paper)
SURFACE = "#fbfaf6"    # cards
PANEL = "#ffffff"      # chart / metric background
EDGE = "#d9d4c7"       # hairlines
WARM = "#c2410c"       # return flow / highlight (rust)
COOL = "#1d4ed8"       # supply flow (cobalt)
PRIMARY = "#0e6b5c"    # brand / primary actions (deep teal)
MUTED = "#6b7280"
INK = "#16212e"
GREEN = "#15803d"
RED = "#b91c1c"
GHOST = "#c9c5b8"      # dimmed geometry
SCENE_BG = "#e9eef2"   # 3D scene background

PAGES = [
    "Operations Center",
    "3D Rig Explorer",
    "Wellbore Cutaway",
    "Mud Circulation Lab",
    "Drilling Simulator",
    "Wireline Operations",
    "Coiled Tubing Lab",
    "AI Diagnostics",
    "Scenario Comparison",
    "SRP Lab",
    "ESP Lab",
    "Reservoir Simulation",
    "Well Path & Torque-Drag",
    "Well Control",
    "Engineering Reference",
]

DISCLAIMER = (
    "Educational simulation. Simplified physics, illustrative coefficients and synthetic data. "
    "Not certified engineering software; not affiliated with or endorsed by any oilfield company."
)

CSS = f"""
<style>
:root {{ --ink:{INK}; --paper:{PAPER}; --edge:{EDGE}; --primary:{PRIMARY}; --muted:{MUTED}; }}
html, body, .stApp {{ background: {PAPER} !important; color: {INK}; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
header[data-testid="stHeader"] {{ background: {PAPER}; border-bottom: 1px solid {EDGE}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ color: {INK}; font-family: Georgia, "Times New Roman", serif; letter-spacing: -.2px; font-weight: 600; }}
h1 {{ font-size: 1.7rem !important; margin: 0 0 .1rem 0; padding: 0; }}
h3 {{ font-size: 1.15rem !important; }}
p, li, label, span {{ color: {INK}; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {EDGE}; }}
div[data-testid="stMetric"] {{ background: transparent; border: 0; border-left: 2px solid {INK}; padding: 0 0 0 10px; border-radius: 0; box-shadow: none; }}
div[data-testid="stMetricValue"] {{ font-size: 1.3rem; font-variant-numeric: tabular-nums; color: {INK}; font-family: Georgia, serif; }}
div[data-testid="stMetricLabel"] p {{ font-size: .8rem; color: {MUTED}; }}
div[data-testid="stExpander"] {{ background: transparent; border: 0; border-top: 1px solid {EDGE}; border-bottom: 1px solid {EDGE}; border-radius: 0; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 3px; border: 1px solid {INK}; background: {PANEL}; color: {INK}; font-weight: 600; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {PRIMARY}; color: {PRIMARY}; }}
.stButton > button[kind="primary"] {{ background: {PRIMARY}; border-color: {PRIMARY}; }}
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span {{ color: #ffffff !important; }}
.stButton > button[kind="primary"]:hover {{ background: #0a5348; border-color: #0a5348; }}
.ps-eyebrow {{ color: {PRIMARY}; font-size: .72rem; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; margin-bottom: .1rem; }}
.ps-sub {{ color: {MUTED}; font-size: .95rem; margin: 0 0 .9rem 0; max-width: 980px; }}
.ps-rule {{ border: 0; border-top: 1px solid {EDGE}; margin: .2rem 0 .8rem 0; }}
.ps-badge {{ display:inline-block; padding: 2px 10px; border-radius: 2px; font-size: .78rem; font-weight: 600;
    border: 1px solid {EDGE}; background: transparent; margin: 0 6px 6px 0; }}
.ps-ok {{ color: {GREEN}; border-color: {GREEN}; }}
.ps-warn {{ color: {WARM}; border-color: {WARM}; }}
.ps-bad {{ color: {RED}; border-color: {RED}; }}
.ps-small {{ color: {MUTED}; font-size: .78rem; line-height: 1.35; }}
.ps-brand {{ font-family: Georgia, serif; font-size: 1.25rem; font-weight: 700; color: {INK}; }}
.ps-brand b {{ color: {PRIMARY}; }}
.ps-footer {{ margin-top: 2rem; padding-top: .6rem; border-top: 1px solid {EDGE}; }}
</style>
"""

# =============================================================================
# 2. ENGINEERING CALCULATIONS
# Field units: ppg, gpm, ft, in, psi, cP, klbf, kft-lbf
# =============================================================================
GAL_TO_FT3 = 0.133680556  # 1 US gal = 0.133680556 ft^3
PSI_PER_PPG_FT = 0.052  # psi / (ppg * ft)


def _ret(x):
    """Return float for 0-d inputs, array otherwise."""
    return float(x) if np.ndim(x) == 0 else x


def gpm_to_ft3s(q_gpm):
    """Convert flow rate gal/min -> ft^3/s."""
    return _ret(np.asarray(q_gpm, dtype=float) * GAL_TO_FT3 / 60.0)


def hydrostatic_psi(mw_ppg, tvd_ft):
    """Ph (psi) = 0.052 * MW (ppg) * TVD (ft)."""
    return _ret(PSI_PER_PPG_FT * np.asarray(mw_ppg, dtype=float) * np.asarray(tvd_ft, dtype=float))


def circle_area_in2(d_in):
    """A (in^2) = pi/4 * D^2."""
    return _ret(np.pi / 4.0 * np.asarray(d_in, dtype=float) ** 2)


def annular_area_in2(dh_in, dp_in):
    """A (in^2) = pi/4 * (Dh^2 - Dp^2)."""
    return _ret(np.pi / 4.0 * (np.asarray(dh_in, dtype=float) ** 2 - np.asarray(dp_in, dtype=float) ** 2))


def velocity_ft_s(q_gpm, area_in2):
    """v (ft/s) = Q (ft^3/s) / A (ft^2); A is converted from in^2 (144 in^2 per ft^2)."""
    area_ft2 = np.asarray(area_in2, dtype=float) / 144.0
    return _ret(gpm_to_ft3s(q_gpm) / area_ft2)


def fanning_factor(re, resistance=1.0):
    """
    Simplified Fanning friction factor (documented simplification):
        f = max(16/Re, 0.046*Re^-0.2) * resistance
    The max() blends laminar and smooth-pipe turbulent branches continuously (crossover near Re ~ 1500).
    `resistance` is a dimensionless multiplier (1.0 = nominal) for roughness / non-Newtonian effects.
    Zero flow (Re <= 1) returns 0 safely.
    """
    re = np.asarray(re, dtype=float)
    safe = np.where(re > 1.0, re, 1.0)
    f = np.maximum(16.0 / safe, 0.046 * safe ** -0.2)
    f = np.where(re > 1.0, f, 0.0)
    return _ret(resistance * f)


def reynolds(mw_ppg, v_fts, d_in, mu_cp):
    """Re = 928 * rho(ppg) * v(ft/s) * d(in) / mu(cP)   (oilfield-unit form, Newtonian)."""
    return _ret(928.0 * np.asarray(mw_ppg, float) * np.asarray(v_fts, float) * np.asarray(d_in, float) / float(mu_cp))


def friction_gradient_psi_ft(q_gpm, mw_ppg, area_in2, d_eq_in, mu_cp, resistance=1.0):
    """
    Frictional pressure gradient (psi/ft) = f * rho * v^2 / (25.8 * d)
    with rho in ppg, v in ft/s, d in inches (Fanning f).  d_eq = ID for pipe, (Dh - Dp) for the annulus
    (simplified slot approximation).
    """
    v = np.asarray(velocity_ft_s(q_gpm, area_in2), dtype=float)
    re = reynolds(mw_ppg, v, d_eq_in, mu_cp)
    f = fanning_factor(re, resistance)
    return _ret(f * np.asarray(mw_ppg, float) * v ** 2 / (25.8 * np.asarray(d_eq_in, float)))


# ---- rheology models (field-unit correlations in the style of API RP 13D / Bourgoyne et al.; simplified) ----
RHEO_MODELS = ["Newtonian", "Bingham plastic", "Power law"]


def make_rheology(model="Newtonian", mu_cp=15.0, t600=60.0, t300=38.0) -> dict:
    """
    Fluid description.  Bingham-plastic and power-law parameters are derived from Fann dial readings:
        PV = theta600 - theta300 [cP],  YP = theta300 - PV [lbf/100 ft^2],
        n = 3.32 log10(theta600/theta300),  K = 5.11 theta300 / 511^n [dyne s^n / cm^2].
    The Newtonian model uses `mu_cp` directly.
    """
    t600, t300 = float(t600), float(t300)
    pv = t600 - t300
    yp = t300 - pv
    if t600 > 0 and t300 > 0:
        n = 3.32 * math.log10(t600 / t300)
        k_dyn = 5.11 * t300 / 511.0 ** n
    else:
        n, k_dyn = float("nan"), float("nan")
    return dict(model=model, mu=float(mu_cp), t600=t600, t300=t300, pv=pv, yp=yp, n=float(n), k=float(k_dyn))


def validate_rheology(rh: dict) -> list[str]:
    errs = []
    m = rh.get("model")
    if m not in RHEO_MODELS:
        return ["Unknown rheology model."]
    if m == "Newtonian":
        if not rh["mu"] > 0:
            errs.append("Effective viscosity must be positive.")
        return errs
    if not (rh["t600"] > rh["t300"] > 0):
        errs.append("Fann readings must satisfy theta600 > theta300 > 0.")
    elif rh["yp"] < 0 or not (0.05 <= rh["n"] <= 1.0):
        errs.append("Fann readings give a negative yield point (theta300 must be at least half of theta600).")
    return errs


def _dpdl_psi_ft(v, mw, d, rh, kind):
    """
    Frictional pressure gradient (psi/ft).  v [ft/s], mw [ppg], d [in] (pipe ID, or Dh - Dp for the annulus).
      Newtonian      : Fanning f = max(16/Re, 0.046 Re^-0.2)
      Bingham plastic: laminar PV v/(1500 d^2) + YP/(225 d) (annulus 1000 / 200); turbulent rho^0.75 v^1.75 PV^0.25/(1800 d^1.25)
                       (annulus 1396); the larger of the two branches is used, which is continuous at the transition.
      Power law      : effective viscosity; laminar below Re_c = 3470 - 1370 n, turbulent f = a / Re^b.
    """
    v = np.asarray(v, dtype=float)
    mw = np.asarray(mw, dtype=float)
    d = np.asarray(d, dtype=float)
    vs = np.where(v > 1e-9, v, 1e-9)
    m = rh["model"]
    if m == "Bingham plastic":
        pv, yp = rh["pv"], rh["yp"]
        c_lam, c_yp, c_tur = (1500.0, 225.0, 1800.0) if kind == "pipe" else (1000.0, 200.0, 1396.0)
        lam = pv * vs / (c_lam * d ** 2) + yp / (c_yp * d)
        tur = mw ** 0.75 * vs ** 1.75 * pv ** 0.25 / (c_tur * d ** 1.25)
        g = np.maximum(lam, tur)
    elif m == "Power law":
        n, kd = rh["n"], rh["k"]
        if kind == "pipe":
            mu_e = 100.0 * kd * ((96.0 * vs / d) * (3.0 * n + 1.0) / (4.0 * n)) ** (n - 1.0)
            re = 928.0 * mw * vs * d / mu_e
            c_lam, c_tur = 1500.0, 25.81
        else:
            mu_e = 100.0 * kd * ((144.0 * vs / d) * (2.0 * n + 1.0) / (3.0 * n)) ** (n - 1.0)
            re = 757.0 * mw * vs * d / mu_e
            c_lam, c_tur = 1000.0, 21.1
        lam = mu_e * vs / (c_lam * d ** 2)
        a = (math.log10(n) + 3.93) / 50.0
        b = (1.75 - math.log10(n)) / 7.0
        f = a / np.maximum(re, 1.0) ** b
        tur = f * mw * vs ** 2 / (c_tur * d)
        g = np.where(re < 3470.0 - 1370.0 * n, lam, tur)
    else:
        re = 928.0 * mw * vs * d / rh["mu"]
        rs = np.maximum(re, 1.0)
        f = np.maximum(16.0 / rs, 0.046 * rs ** -0.2)
        g = f * mw * vs ** 2 / (25.8 * d)
    return np.where(v > 1e-9, g, 0.0)


def pipe_dp_rheo(q_gpm, mw_ppg, id_in, length_ft, rh, resistance=1.0):
    v = velocity_ft_s(q_gpm, circle_area_in2(id_in))
    return _ret(resistance * _dpdl_psi_ft(v, mw_ppg, id_in, rh, "pipe") * np.asarray(length_ft, float))


def ann_dp_rheo(q_gpm, mw_ppg, dh_in, dp_in, length_ft, rh, resistance=1.0):
    v = velocity_ft_s(q_gpm, annular_area_in2(dh_in, dp_in))
    return _ret(resistance * _dpdl_psi_ft(v, mw_ppg, dh_in - dp_in, rh, "annulus") * np.asarray(length_ft, float))


def pipe_pressure_loss_psi(q_gpm, mw_ppg, id_in, length_ft, mu_cp, resistance=1.0):
    """Newtonian pipe friction loss (psi); kept as a thin wrapper over the rheology-aware routine."""
    return pipe_dp_rheo(q_gpm, mw_ppg, id_in, length_ft, make_rheology("Newtonian", mu_cp), resistance)


def annular_pressure_loss_psi(q_gpm, mw_ppg, dh_in, dp_in, length_ft, mu_cp, resistance=1.0):
    """Newtonian annular friction loss (psi); thin wrapper over the rheology-aware routine."""
    return ann_dp_rheo(q_gpm, mw_ppg, dh_in, dp_in, length_ft, make_rheology("Newtonian", mu_cp), resistance)


def annular_flow_state(v_fts, mw_ppg, dh_in, dp_in, rh) -> dict:
    """Apparent annular viscosity (cP), effective Reynolds number and laminar/turbulent flag for the annulus."""
    v = max(float(v_fts), 1e-9)
    d = float(dh_in - dp_in)
    m = rh["model"]
    if m == "Bingham plastic":
        mu_a = rh["pv"] + 5.0 * rh["yp"] * d / v
        re, crit = 928.0 * mw_ppg * v * d / mu_a, 2100.0
    elif m == "Power law":
        n = rh["n"]
        mu_a = 100.0 * rh["k"] * ((144.0 * v / d) * (2.0 * n + 1.0) / (3.0 * n)) ** (n - 1.0)
        re, crit = 757.0 * mw_ppg * v * d / mu_a, 3470.0 - 1370.0 * n
    else:
        mu_a = rh["mu"]
        re, crit = 928.0 * mw_ppg * v * d / mu_a, 1500.0
    return dict(mu_app_cp=float(mu_a), re=float(re), laminar=bool(re < crit), re_crit=float(crit))


# ---- cuttings transport (vertical annulus, simplified) ----
def cuttings_slip_velocity_fts(d_cut_in, rho_cut_ppg, mw_ppg, mu_app_cp) -> float:
    """
    Terminal settling (slip) velocity of a spherical cutting from a force balance with the Schiller-Naumann drag law
    (CD = 24/Re (1 + 0.15 Re^0.687) for Re < 1000, 0.44 above), solved by relaxed fixed-point iteration.
    Returns ft/s.  Zero when the particle is not denser than the mud.
    """
    if rho_cut_ppg <= mw_ppg or mu_app_cp <= 0 or d_cut_in <= 0:
        return 0.0
    d = d_cut_in * 0.0254
    rs, rm = rho_cut_ppg * 119.826, mw_ppg * 119.826  # kg/m^3
    mu = mu_app_cp * 1e-3  # Pa s
    v = 0.05
    for _ in range(80):
        re = max(rm * v * d / mu, 1e-9)
        cd = 24.0 / re * (1.0 + 0.15 * re ** 0.687) if re < 1000.0 else 0.44
        v_new = math.sqrt(4.0 * 9.80665 * d * (rs - rm) / (3.0 * rm * cd))
        v = 0.5 * v + 0.5 * v_new
    return v * 3.28084


def transport_ratio(v_ann_fts, v_slip_fts) -> float:
    """Ft = (Va - Vs) / Va, clipped to [0, 1]."""
    if v_ann_fts <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (v_ann_fts - v_slip_fts) / v_ann_fts)))


def cuttings_concentration(rop_ftph, hole_in, q_gpm, ft, porosity=0.2) -> float:
    """
    Volume fraction of cuttings in the annulus from solids continuity:
        C = (1 - phi) A_hole ROP / (Ft Q)     (capped at 0.5; 0.5 when nothing is transported)
    """
    vol_rate = (1.0 - porosity) * (circle_area_in2(hole_in) / 144.0) * max(rop_ftph, 0.0) / 3600.0  # ft^3/s
    if vol_rate <= 0:
        return 0.0
    carried = ft * gpm_to_ft3s(q_gpm)
    return 0.5 if carried <= 0 else float(min(0.5, vol_rate / carried))


# ---- lithology, pore pressure and fracture gradient (illustrative profiles, NOT real field geology) ----
LITHOLOGY = [
    # name, top ft, bottom ft, relative strength (-), abrasiveness (-), display colour
    ("Overburden / soil", 0, 600, 0.35, 0.3, "#8c6d46"),
    ("Sandstone (aquifer-like)", 600, 2800, 0.60, 1.2, "#d9b36c"),
    ("Shale", 2800, 5200, 0.80, 0.5, "#6b7280"),
    ("Limestone", 5200, 7400, 1.30, 0.8, "#b8c2cc"),
    ("Reservoir sandstone (illustrative)", 7400, 9600, 1.00, 1.1, "#e0a458"),
    ("Hard rock / basement", 9600, 12000, 2.00, 1.6, "#4b4f58"),
]
_LITH_TOPS = np.array([row[1] for row in LITHOLOGY], dtype=float)
_LITH_STRENGTH = np.array([row[3] for row in LITHOLOGY], dtype=float)
_LITH_ABRAS = np.array([row[4] for row in LITHOLOGY], dtype=float)
_LITH_NAMES = np.array([row[0] for row in LITHOLOGY])


def lithology_index(depth_ft):
    idx = np.searchsorted(_LITH_TOPS, np.asarray(depth_ft, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(LITHOLOGY) - 1)


def pore_gradient_ppg(tvd_ft, overpress=0.0, op_top=9500.0):
    """Illustrative pore-pressure gradient (ppg): 8.6 + 4e-5 TVD, plus an overpressure ramp (up to +3 ppg x overpress) over 1,500 ft."""
    tvd = np.asarray(tvd_ft, dtype=float)
    ramp = np.clip((tvd - op_top) / 1500.0, 0.0, 1.0)
    return _ret(8.6 + 4.0e-5 * tvd + 3.0 * overpress * ramp)


def overburden_gradient_ppg(tvd_ft):
    return _ret(17.5 - 6.5 * np.exp(-np.asarray(tvd_ft, dtype=float) / 5000.0))


def fracture_gradient_ppg(tvd_ft, overpress=0.0, op_top=9500.0):
    """Eaton-style illustrative fracture gradient: FG = PP + nu/(1-nu) (OB - PP), nu rising with depth."""
    tvd = np.asarray(tvd_ft, dtype=float)
    pp = np.asarray(pore_gradient_ppg(tvd, overpress, op_top), dtype=float)
    ob = np.asarray(overburden_gradient_ppg(tvd), dtype=float)
    nu = 0.25 + 0.15 * (1.0 - np.exp(-tvd / 6000.0))
    return _ret(pp + nu / (1.0 - nu) * (ob - pp))


def bit_torque_ftlbf(wob_klbf, hole_in, wear, mu_b=0.6):
    """Bit torque (ft-lbf) = mu_b D WOB / 36 (D in, WOB lbf), rising with wear (illustrative)."""
    return mu_b * (1.0 + 0.4 * np.asarray(wear, float)) * hole_in * np.asarray(wob_klbf, float) * 1000.0 / 36.0


def mse_psi(wob_klbf, rpm, torque_ftlbf, rop_ftph, hole_in):
    """Teale mechanical specific energy: MSE = WOB/Ab + 120 pi N T / (Ab ROP)   [psi; WOB lbf, T ft-lbf, ROP ft/h]."""
    ab = circle_area_in2(hole_in)
    rop = np.maximum(np.asarray(rop_ftph, float), 1e-6)
    return _ret(np.asarray(wob_klbf, float) * 1000.0 / ab + 120.0 * math.pi * np.asarray(rpm, float) * np.asarray(torque_ftlbf, float) / (ab * rop))


def bit_pressure_drop_psi(q_gpm, mw_ppg, tfa_in2, cd=0.95):
    """dP_bit (psi) = 8.311e-5 * rho(ppg) * Q^2(gpm) / (Cd^2 * TFA^2(in^2))."""
    return _ret(8.311e-5 * np.asarray(mw_ppg, float) * np.asarray(q_gpm, float) ** 2 / (cd ** 2 * tfa_in2 ** 2))


def surface_loss_psi(q_gpm, k=3.0e-4):
    """Illustrative surface-equipment loss (standpipe, hose, swivel): k * Q^2 (k is an assumed coefficient)."""
    return _ret(k * np.asarray(q_gpm, float) ** 2)


def ecd_ppg(mw_ppg, dp_ann_psi, tvd_ft):
    """ECD (ppg) = MW + dP_annulus / (0.052 * TVD).  Returns MW when TVD <= 0."""
    tvd = np.asarray(tvd_ft, float)
    safe = np.where(tvd > 0, tvd, 1.0)
    out = np.where(tvd > 0, np.asarray(mw_ppg, float) + np.asarray(dp_ann_psi, float) / (PSI_PER_PPG_FT * safe), mw_ppg)
    return _ret(out)


def validate_hydraulics(mw, q, tvd, dh, dp, did=None, tfa=None, mu=None, k=None) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errs = []
    if not mw > 0:
        errs.append("Mud weight must be positive.")
    if q < 0:
        errs.append("Flow rate must be non-negative.")
    if not tvd > 0:
        errs.append("True vertical depth must be positive.")
    if not dh > 0:
        errs.append("Hole diameter must be positive.")
    if not dp > 0:
        errs.append("Drill-pipe outer diameter must be positive.")
    if dh > 0 and dp > 0 and not dh > dp:
        errs.append("Hole diameter must be greater than drill-pipe outer diameter.")
    if did is not None:
        if not did > 0:
            errs.append("Drill-pipe inner diameter must be positive.")
        elif dp > 0 and not did < dp:
            errs.append("Drill-pipe inner diameter must be smaller than its outer diameter.")
    if tfa is not None and not tfa > 0:
        errs.append("Bit total flow area must be positive.")
    if mu is not None and not mu > 0:
        errs.append("Effective viscosity must be positive.")
    if k is not None and k < 0:
        errs.append("Hydraulic-resistance multiplier must be non-negative.")
    return errs


def hydraulics_summary(mw, q, tvd, dh, dp, did=4.276, tfa=0.746, mu=15.0, k=1.0, rheo=None) -> dict:
    """Full simplified hydraulics for a given state.  `rheo` (see make_rheology) defaults to Newtonian(mu). Raises ValueError on invalid inputs."""
    rh = rheo if rheo is not None else make_rheology("Newtonian", mu)
    errs = validate_hydraulics(mw, q, tvd, dh, dp, did, tfa, mu if rh["model"] == "Newtonian" else 1.0, k) + validate_rheology(rh)
    if errs:
        raise ValueError(" ".join(errs))
    a_ann = annular_area_in2(dh, dp)
    a_pipe = circle_area_in2(did)
    v_ann = velocity_ft_s(q, a_ann)
    v_pipe = velocity_ft_s(q, a_pipe)
    dp_pipe = pipe_dp_rheo(q, mw, did, tvd, rh, k)
    dp_ann = ann_dp_rheo(q, mw, dh, dp, tvd, rh, k)
    dp_bit = bit_pressure_drop_psi(q, mw, tfa)
    dp_surf = surface_loss_psi(q)
    spp = dp_pipe + dp_ann + dp_bit + dp_surf
    state = annular_flow_state(v_ann, mw, dh, dp, rh)
    return dict(
        hydrostatic_psi=hydrostatic_psi(mw, tvd),
        annular_area_in2=a_ann,
        pipe_area_in2=a_pipe,
        q_ft3s=gpm_to_ft3s(q),
        v_ann_fts=v_ann,
        v_ann_ftmin=v_ann * 60.0,
        v_pipe_fts=v_pipe,
        dp_pipe_psi=dp_pipe,
        dp_ann_psi=dp_ann,
        dp_bit_psi=dp_bit,
        dp_surface_psi=dp_surf,
        spp_psi=spp,
        ecd_ppg=ecd_ppg(mw, dp_ann, tvd),
        bottomhole_circ_psi=hydrostatic_psi(mw, tvd) + dp_ann,
        pump_hhp=spp * q / 1714.0,
        re_ann=state["re"],
        mu_app_cp=state["mu_app_cp"],
        regime="laminar" if state["laminar"] else "turbulent-like",
        rheology=rh,
    )


# =============================================================================
# 3. DRILLING SIMULATOR MODEL  (deterministic, parameterised, illustrative)
# =============================================================================
SIM_DT = 2.0  # s per step
SS_PERIOD = 8.0  # s, synthetic stick-slip-like period
ROP0 = 60.0  # ft/h reference ROP (illustrative)
BIT_LIFE_H = 80.0  # nominal bit life at reference WOB/RPM in an abrasiveness-1 formation (illustrative)
KICK_GPM_PER_PPG = 60.0  # illustrative influx rate per ppg of underbalance
LOSS_GPM_PER_PPG = 80.0  # illustrative loss rate per ppg of ECD above fracture gradient
GEOM_KEYS = ["hole_in", "pipe_od", "pipe_id", "tfa", "mu_cp", "k_res", "rheo_model", "t600", "t300"]
_LITH_TOPS_LIST = [float(x) for x in _LITH_TOPS]

SIM_DEFAULTS = dict(
    wob=22.0, rpm=110.0, flow=520.0, mw=10.0, fr=1.0, fric=0.0, ss=0.0,
    duration_min=30.0, start_depth=8000.0,
    hole_in=8.5, pipe_od=5.0, pipe_id=4.276, tfa=0.746, mu_cp=15.0, k_res=1.0,
    rheo_model="Newtonian", t600=60.0, t300=38.0,
    lith_on=1.0, bit_wear0=0.0, wear_lapse=20.0, d_cut=0.25, rho_cut=21.0,
    overpress=0.0, op_top=9500.0,
    noise=0.01, wander=0.0, seed=7,
)


def rheology_of(p: dict) -> dict:
    return make_rheology(p.get("rheo_model", "Newtonian"), p["mu_cp"], p.get("t600", 60.0), p.get("t300", 38.0))


def validate_sim_params(p: dict) -> list[str]:
    errs = validate_hydraulics(p["mw"], p["flow"], max(p["start_depth"], 1.0), p["hole_in"], p["pipe_od"],
                               p["pipe_id"], p["tfa"], p["mu_cp"], p["k_res"])
    errs += validate_rheology(rheology_of(p))
    if not p["wob"] > 0:
        errs.append("WOB must be positive.")
    if p["rpm"] < 0:
        errs.append("RPM must be non-negative.")
    if not p["fr"] > 0:
        errs.append("Formation resistance must be positive.")
    if not (0 <= p["fric"] <= 1 and 0 <= p["ss"] <= 1):
        errs.append("Disturbance severities must be within 0-1.")
    if not p["duration_min"] > 0:
        errs.append("Duration must be positive.")
    if not 0.0 <= p["bit_wear0"] <= 0.95:
        errs.append("Initial bit wear must be between 0 and 0.95.")
    if p["wear_lapse"] < 0:
        errs.append("Wear time-lapse factor must be non-negative.")
    if not p["d_cut"] > 0:
        errs.append("Cuttings size must be positive.")
    if not p["overpress"] >= 0:
        errs.append("Overpressure factor must be non-negative.")
    return errs


def rop_factor_terms(p: dict, wob=None, rpm=None) -> dict:
    """
    Multiplicative ROP factors of the illustrative model (all dimensionless).
        ROP = ROP0 * f_wob * f_rpm * f_mw * f_hyd * f_fr * f_ss * f_fric * f_wear * f_load / strength(depth)
    ILLUSTRATIVE COEFFICIENTS - not calibrated to any field data, not a universal ROP law.
    """
    wob = p["wob"] if wob is None else wob
    rpm = p["rpm"] if rpm is None else rpm
    wob = np.asarray(wob, float)
    rpm = np.asarray(rpm, float)
    q_ref = 50.0 * p["hole_in"]  # assumed hole-cleaning reference flow (gpm)
    return dict(
        f_wob=(wob / 20.0) ** 0.8 / (1.0 + (wob / 45.0) ** 4),  # power-law gain with bit-floundering roll-off
        f_rpm=(rpm / 100.0) ** 0.6 / (1.0 + (rpm / 220.0) ** 3),  # diminishing returns at high RPM
        f_mw=1.0 / (1.0 + 0.12 * max(0.0, p["mw"] - 9.0)),  # chip hold-down penalty
        f_hyd=0.55 + 0.45 * min(1.0, (max(p["flow"], 0.0) / q_ref) ** 0.5),  # cuttings removal
        f_fr=1.0 / p["fr"],
        f_ss=1.0 - 0.35 * p["ss"],
        f_fric=1.0 - 0.20 * p["fric"],
    )


def expected_torque(wob, rpm, depth):
    """Nominal (undisturbed, FR=1) torque in kft-lbf used as an ML feature reference."""
    return (0.8 + 0.00015 * np.asarray(depth, float)) + 0.11 * np.asarray(wob, float) + 0.004 * np.asarray(rpm, float)


def expected_spp(flow, mw, depth, geom: dict):
    """Expected standpipe pressure (psi) from the hydraulics model for a given flow/depth."""
    flow = np.asarray(flow, float)
    depth = np.asarray(depth, float)
    rh = make_rheology(geom.get("rheo_model", "Newtonian"), geom["mu_cp"], geom.get("t600", 60.0), geom.get("t300", 38.0))
    return (
        pipe_dp_rheo(flow, mw, geom["pipe_id"], depth, rh, geom["k_res"])
        + ann_dp_rheo(flow, mw, geom["hole_in"], geom["pipe_od"], depth, rh, geom["k_res"])
        + bit_pressure_drop_psi(flow, mw, geom["tfa"])
        + surface_loss_psi(flow)
    )


def steady_rop_mse(p: dict, wob_grid=None, rpm_grid=None) -> dict:
    """Steady-state ROP and MSE of the model at the start depth, swept over WOB and RPM (no wear, clean hole)."""
    wob_grid = np.linspace(2.0, 50.0, 49) if wob_grid is None else np.asarray(wob_grid, float)
    rpm_grid = np.linspace(20.0, 220.0, 41) if rpm_grid is None else np.asarray(rpm_grid, float)
    li = int(lithology_index(p["start_depth"])) if p.get("lith_on", 1.0) >= 0.5 else None
    s_rel = _LITH_STRENGTH[li] if li is not None else 1.0
    out = {}
    for name, grid, fixed in (("wob", wob_grid, "rpm"), ("rpm", rpm_grid, "wob")):
        fac = rop_factor_terms(p, wob=grid if name == "wob" else p["wob"], rpm=grid if name == "rpm" else p["rpm"])
        rop = ROP0 * fac["f_wob"] * fac["f_rpm"] * fac["f_mw"] * fac["f_hyd"] * fac["f_ss"] * fac["f_fric"] / (p["fr"] * s_rel)
        wob_v = grid if name == "wob" else np.full_like(grid, p["wob"])
        rpm_v = grid if name == "rpm" else np.full_like(grid, p["rpm"])
        tb = bit_torque_ftlbf(wob_v, p["hole_in"], p["bit_wear0"])
        out[name] = dict(x=grid, rop=np.asarray(rop, float), mse=np.asarray(mse_psi(wob_v, rpm_v, tb, rop, p["hole_in"]), float))
    return out


def simulate_drilling(params: dict) -> pd.DataFrame:
    """
    Deterministic drilling time series (seeded measurement noise).  Formation strength changes with depth (LITHOLOGY),
    the bit wears, cuttings load the annulus (feeding back on ROP), and ECD is compared with the pore/fracture window.
    Columns: t_s, depth_ft, rop_ftph, wob_klbf, rpm, rpm_bit, torque_kftlbf, flow_gpm, flow_set_gpm, spp_psi, ecd_ppg,
             hyd_psi, mw_ppg, bit_angle_deg, formation, strength_rel, wear, bit_torque_ftlbf, mse_psi, cut_conc,
             ecd_cut_ppg, pore_ppg, frac_ppg, flow_out_gpm, pit_gain_bbl
    """
    p = {**SIM_DEFAULTS, **params}
    errs = validate_sim_params(p)
    if errs:
        raise ValueError(" ".join(errs))
    n = max(2, int(round(p["duration_min"] * 60.0 / SIM_DT)))
    t = np.arange(n) * SIM_DT
    rng = np.random.default_rng(int(p["seed"]))
    phi = rng.uniform(0, 2 * np.pi, 3)

    wob_t = np.clip(p["wob"] + p["wander"] * np.sin(2 * np.pi * t / 300.0 + phi[0]), 0.5, None)
    rpm_t = np.clip(p["rpm"] + 0.5 * p["wander"] * np.sin(2 * np.pi * t / 420.0 + phi[1]), 0.0, None)
    fac = rop_factor_terms(p, wob=wob_t, rpm=rpm_t)
    rh = rheology_of(p)
    q = float(p["flow"])

    # hole cleaning: annular velocity -> slip velocity -> transport ratio (constant for the run)
    if q > 0:
        va = velocity_ft_s(q, annular_area_in2(p["hole_in"], p["pipe_od"]))
        mu_app = annular_flow_state(va, p["mw"], p["hole_in"], p["pipe_od"], rh)["mu_app_cp"]
        ft = transport_ratio(va, cuttings_slip_velocity_fts(p["d_cut"], p["rho_cut"], p["mw"], mu_app))
    else:
        ft = 0.0

    gain_t = ROP0 * fac["f_mw"] * fac["f_hyd"] * fac["f_ss"] * fac["f_fric"] / p["fr"] * fac["f_wob"] * fac["f_rpm"]
    lith_on = p["lith_on"] >= 0.5
    depth = np.empty(n)
    rop_true = np.empty(n)
    s_arr = np.empty(n)
    wear = np.empty(n)
    cc = np.empty(n)
    d = float(p["start_depth"])
    w = float(p["bit_wear0"])
    rop_prev = float(gain_t[0]) * 0.8
    last = len(_LITH_TOPS_LIST) - 1
    for i in range(n):
        li = min(max(bisect.bisect_right(_LITH_TOPS_LIST, d) - 1, 0), last)
        s_rel = _LITH_STRENGTH[li] if lith_on else 1.0
        abr = _LITH_ABRAS[li] if lith_on else 1.0
        c = cuttings_concentration(rop_prev, p["hole_in"], q, ft)
        f_load = 1.0 - min(0.6, 5.0 * max(0.0, c - 0.04))  # hole-loading penalty above ~4 % cuttings
        r = gain_t[i] / s_rel * (1.0 - 0.55 * w) * f_load
        depth[i], rop_true[i], s_arr[i], wear[i], cc[i] = d, r, s_rel, w, c
        d += r * SIM_DT / 3600.0
        w = min(1.0, w + SIM_DT / 3600.0 * p["wear_lapse"] * (rpm_t[i] / 100.0) * (wob_t[i] / 20.0) * abr / BIT_LIFE_H)
        rop_prev = r

    dp_pipe = pipe_dp_rheo(q, p["mw"], p["pipe_id"], depth, rh, p["k_res"])
    dp_ann = ann_dp_rheo(q, p["mw"], p["hole_in"], p["pipe_od"], depth, rh, p["k_res"])
    spp_true = dp_pipe + dp_ann + bit_pressure_drop_psi(q, p["mw"], p["tfa"]) + surface_loss_psi(q)
    ecd_t = ecd_ppg(p["mw"], dp_ann, depth)
    ecd_cut = ecd_t + cc * (p["rho_cut"] - p["mw"])
    pore = pore_gradient_ppg(depth, p["overpress"], p["op_top"])
    frac = fracture_gradient_ppg(depth, p["overpress"], p["op_top"])
    q_in = KICK_GPM_PER_PPG * np.maximum(0.0, pore - ecd_cut)
    q_loss = LOSS_GPM_PER_PPG * np.maximum(0.0, ecd_cut - frac)
    flow_out = q + q_in - q_loss
    pit_gain = np.cumsum((q_in - q_loss) * SIM_DT / 60.0 / 42.0)

    osc = np.sin(2 * np.pi * t / SS_PERIOD)
    osc_t = np.sin(2 * np.pi * t / SS_PERIOD + np.pi / 3)
    osc_r = np.sin(2 * np.pi * t / SS_PERIOD + np.pi / 6)
    torque_true = (
        (0.8 + 0.00015 * depth)
        + 0.11 * wob_t * np.sqrt(p["fr"] * s_arr) * (1.0 + 1.2 * p["fric"])
        + 0.004 * rpm_t
    ) * (1.0 + 0.5 * wear) * (1.0 + 0.45 * p["ss"] * osc_t)
    rpm_bit = rpm_t * np.clip(1.0 + 0.8 * p["ss"] * osc, 0.0, None)
    rpm_surf = rpm_t * (1.0 + 0.15 * p["ss"] * osc_r)
    bit_angle = np.cumsum(rpm_bit * 6.0 * SIM_DT)  # deg (rpm * 360/60 deg/s)
    tb = bit_torque_ftlbf(wob_t, p["hole_in"], wear)
    mse = mse_psi(wob_t, rpm_t, tb, rop_true, p["hole_in"])

    nz = p["noise"]

    def noisy(x, scale=1.0):
        return x * (1.0 + rng.normal(0.0, nz * scale, n))

    formation = _LITH_NAMES[lithology_index(depth)] if lith_on else np.full(n, "(uniform formation)")
    return pd.DataFrame(dict(
        t_s=t, depth_ft=depth, rop_ftph=noisy(rop_true, 2.0), wob_klbf=noisy(wob_t, 0.5),
        rpm=noisy(rpm_surf, 0.5), rpm_bit=rpm_bit, torque_kftlbf=noisy(torque_true),
        flow_gpm=noisy(np.full(n, float(q)), 0.5), flow_set_gpm=np.full(n, float(q)),
        spp_psi=noisy(spp_true, 0.8), ecd_ppg=ecd_t, hyd_psi=hydrostatic_psi(p["mw"], depth),
        mw_ppg=np.full(n, float(p["mw"])), bit_angle_deg=bit_angle,
        formation=formation, strength_rel=s_arr, wear=wear, bit_torque_ftlbf=tb, mse_psi=mse, cut_conc=cc,
        ecd_cut_ppg=ecd_cut, pore_ppg=pore, frac_ppg=frac, flow_out_gpm=flow_out, pit_gain_bbl=pit_gain,
    ))


def freeze(d: dict) -> tuple:
    return tuple(sorted(d.items()))


@st.cache_data(show_spinner=False, max_entries=128, ttl=3600)
def cached_sim(frozen: tuple) -> pd.DataFrame:
    return simulate_drilling(dict(frozen))


# =============================================================================
# 3b. SUCKER-ROD PUMP (SRP) ENGINE  - educational, simplified, NOT validated against field dynamometer data
# =============================================================================
G_FT_S2 = 32.174
ROD_TABLE = {  # nominal rod dia (in): (weight lb/ft in air, area in^2)
    0.75: (1.634, 0.4418), 0.875: (2.224, 0.6013), 1.0: (2.904, 0.7854),
}
E_STEEL_PSI = 30.5e6
ROD_WAVE_FTS = 16300.0
UNIT_SIZES = {  # name: (peak torque rating kin-lbf, structure rating lbf, max stroke in) - generic illustrative sizes
    "C-114D-143-64": (114.0, 14300.0, 64.0), "C-228D-213-86": (228.0, 21300.0, 86.0),
    "C-320D-256-100": (320.0, 25600.0, 100.0), "C-456D-305-144": (456.0, 30500.0, 144.0),
}


def srp_four_bar(phi, r=0.17, pm=1.0, h_over=0.0):
    """
    Beam angle beta(phi) of a conventional (Class I) unit in units of the rear beam arm P = 1.
    Crank shaft sits directly below the rear beam end at (1, -(pm + r + h_over)); crank angle phi from top dead centre (rad).
    Returns beta (rad; front end DOWN positive).
    """
    phi = np.asarray(phi, float)
    px, py = 1.0 + r * np.sin(phi), -(pm + r + h_over) + r * np.cos(phi)
    rho = np.hypot(px, py)
    psi = np.arctan2(py, px)
    kappa = (1.0 + rho ** 2 - pm ** 2) / 2.0
    ac = np.arccos(np.clip(kappa / rho, -1.0, 1.0))
    b1, b2 = psi + ac, psi - ac
    wrap = lambda b: (b + np.pi) % (2 * np.pi) - np.pi
    b1, b2 = wrap(b1), wrap(b2)
    return np.where(np.abs(b1) < np.abs(b2), b1, b2)


def srp_kinematics(stroke_in, n=720, **kw):
    """Polished-rod position y (ft, up positive, 0 at bottom of stroke) and torque factor TF (in/rad) over one crank revolution."""
    phi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    beta = srp_four_bar(phi, **kw)
    # front end goes DOWN with positive beta, so polished rod rises when beta falls
    span = beta.max() - beta.min()
    i_arm = stroke_in / span  # front beam arm (in) that yields the requested stroke
    y_in = i_arm * (beta.max() - beta)
    dphi = 2 * np.pi / n
    tf = np.gradient(np.concatenate([y_in, y_in[:1]]), dphi)[:n]  # in/rad
    return phi, y_in / 12.0, tf


def rod_string(depth_ft, fr_top, fr_mid, fr_bot):
    """Three-section taper (1 in / 7/8 in / 3/4 in), fractions normalised. Returns list of (dia, length_ft, w, A)."""
    f = np.array([fr_top, fr_mid, fr_bot], float)
    f = f / f.sum()
    out = []
    for d, fr in zip((1.0, 0.875, 0.75), f):
        if fr > 1e-6:
            w, a = ROD_TABLE[d]
            out.append((d, depth_ft * float(fr), w, a))
    return out


def validate_srp(p: dict) -> list[str]:
    e = []
    if not 2000 <= p["depth"] <= 12000:
        e.append("Pump depth must be 2,000-12,000 ft.")
    if not 0.5 <= p["pump_d"] <= 4.0:
        e.append("Plunger diameter must be 0.5-4 in.")
    if not 20 <= p["stroke"] <= 200:
        e.append("Surface stroke must be 20-200 in.")
    if not 1.0 <= p["spm"] <= 15.0:
        e.append("SPM must be 1-15.")
    if not 0.6 <= p["sg"] <= 1.3:
        e.append("Fluid specific gravity must be 0.6-1.3.")
    if p["pt"] < 0 or p["pr"] <= 0:
        e.append("Tubing pressure must be >= 0 and reservoir pressure > 0.")
    if p["qmax"] <= 0:
        e.append("Vogel qmax must be positive.")
    if not 0.0 <= p["slip"] < 0.5:
        e.append("Slippage fraction must be 0-0.5.")
    if not 0.0 <= p["damp"] <= 0.6:
        e.append("Damping factor must be 0-0.6.")
    if p["f_top"] + p["f_mid"] + p["f_bot"] <= 0:
        e.append("Rod taper fractions must not all be zero.")
    return e


SRP_DEFAULTS = dict(depth=6000.0, pump_d=1.75, stroke=64.0, spm=6.0, sg=0.85, pt=100.0, pr=1800.0, qmax=600.0,
                    pi_min=60.0, slip=0.03, damp=0.15, f_top=0.33, f_mid=0.34, f_bot=0.33, cbe=-1.0, unit="C-228D-213-86",
                    nodes=60, cycles=4)


def vogel_q(pwf, pr, qmax):
    x = np.clip(np.asarray(pwf, float) / pr, 0.0, 1.0)
    return qmax * (1.0 - 0.2 * x - 0.8 * x ** 2)


def vogel_pwf(q, pr, qmax):
    """Inverse Vogel: flowing bottomhole pressure (psi) for a rate q (bpd); None if q > qmax."""
    if q >= qmax:
        return 0.0
    x = (-0.2 + np.sqrt(0.04 + 3.2 * (1.0 - q / qmax))) / 1.6  # solve 0.8x^2 + 0.2x - (1 - q/qmax) = 0
    return float(x * pr)


def srp_inflow_match(p: dict) -> dict:
    """Match pump capacity to Vogel inflow. Pump set at mid-perforation (assumption) so intake pressure = Pwf."""
    ap = np.pi / 4 * p["pump_d"] ** 2
    q_th = 0.1166 * ap * p["stroke"] * p["spm"]  # API: bpd = 0.1166 Ap Sp N (Sp in in)
    q_cap = q_th * (1.0 - p["slip"])
    q_at_min = float(vogel_q(p["pi_min"], p["pr"], p["qmax"]))
    if q_cap <= q_at_min:  # pump can't drain the well below pi_min: balanced, full pump
        pwf, fill, q = vogel_pwf(q_cap, p["pr"], p["qmax"]), 1.0, q_cap
    else:  # pump-off: inflow limited at pi_min
        pwf, fill, q = p["pi_min"], q_at_min / q_cap, q_at_min
    return dict(ap=float(ap), q_theor=float(q_th), q_cap=float(q_cap), pi=float(max(pwf, p["pi_min"])), pwf=float(pwf), fill=float(fill),
                q_prod=float(q), pump_off=bool(q_cap > q_at_min))


@functools.lru_cache(maxsize=32)
def _srp_cached(items: tuple) -> dict:
    return _srp_run(dict(items))


def srp_simulate(params: dict) -> dict:
    p = {**SRP_DEFAULTS, **params}
    errs = validate_srp(p)
    if errs:
        raise ValueError(" ".join(errs))
    return _srp_cached(tuple(sorted(p.items())))


def _srp_run(p: dict) -> dict:
    inflow = srp_inflow_match(p)
    ap, pi, fill = inflow["ap"], inflow["pi"], inflow["fill"]
    L = float(p["depth"])
    secs = rod_string(L, p["f_top"], p["f_mid"], p["f_bot"])
    n = int(p["nodes"])
    dz = L / n
    # element properties (z downward from surface), node 0 = polished rod, node n = plunger
    z_mid = (np.arange(n) + 0.5) * dz
    edges = np.cumsum([s[1] for s in secs])
    sec_i = np.minimum(np.searchsorted(edges, z_mid), len(secs) - 1)
    w_el = np.array([secs[i][2] for i in sec_i])
    a_el = np.array([secs[i][3] for i in sec_i])
    k_el = E_STEEL_PSI * a_el / dz  # lbf/ft (EA in lbf, strain = delta_u[ft] / dz[ft])
    m_el = w_el * dz / G_FT_S2
    m_node = np.zeros(n + 1)
    m_node[:-1] += 0.5 * m_el
    m_node[1:] += 0.5 * m_el
    bf = 1.0 - p["sg"] * 62.4 / 490.0
    w_buoy_total = float((w_el * dz).sum() * bf)
    dt = 0.95 * dz / ROD_WAVE_FTS
    period = 60.0 / p["spm"]
    steps_cycle = int(np.ceil(period / dt))
    dt = period / steps_cycle
    ncyc = int(p["cycles"])
    nstep = steps_cycle * ncyc
    # kinematics
    phi_g, y_g, tf_g = srp_kinematics(p["stroke"])
    t = np.arange(nstep) * dt
    ph = (2 * np.pi * (t % period) / period)
    y_pr = np.interp(ph, np.append(phi_g, 2 * np.pi), np.append(y_g, y_g[0]))
    tf_t = np.interp(ph, np.append(phi_g, 2 * np.pi), np.append(tf_g, tf_g[0]))
    p_dis = p["pt"] + 0.433 * p["sg"] * L  # pump discharge pressure: wellhead tubing pressure + full liquid column
    fo = max(ap * (p_dis - pi), 0.0)  # fluid load on the plunger (lbf)
    c = p["damp"] * np.pi * ROD_WAVE_FTS / (2 * L)  # 1/s, mass-proportional damping
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    x_top, x_prev = 0.0, 0.0
    sp_est = p["stroke"] / 12.0
    up = True
    vs0 = np.gradient(y_pr, dt)
    f_fluid = 0.0
    tau = max(0.5, 0.02 * period)  # load-transfer time (valve action + gas): a step would ring, a real transfer is gradual
    d_open_prev = 0.0
    load = np.empty(nstep)
    bload = np.empty(nstep)
    plunger = np.empty(nstep)
    pv = np.empty(nstep)
    for s in range(nstep):
        u[0] = y_pr[s]
        v[0] = (y_pr[s] - y_pr[s - 1]) / dt if s else 0.0
        stretch = u[:-1] - u[1:]
        tdyn = k_el * stretch
        acc = np.zeros(n + 1)
        acc[1:-1] = (tdyn[:-1] - tdyn[1:]) / m_node[1:-1] - c * v[1:-1]
        # plunger node: element tension above, fluid load below (reference: static state without fluid load)
        acc[-1] = (tdyn[-1] - f_fluid) / m_node[-1] - c * v[-1]
        v[1:] += acc[1:] * dt
        u[1:] += v[1:] * dt
        xp, vp = u[-1], v[-1]
        # valve state follows the polished-rod direction (load transfers at the surface turn-arounds; the elastic delay then
        # appears naturally in the wave solution).  x_top tracks the plunger's highest point of the current upstroke.
        up_now = vs0[s] > 0.0
        if up_now:
            x_top = max(x_top, xp) if up else xp
        elif up:
            sp_est = max(0.5 * sp_est + 0.5 * max(x_top - x_prev, 1e-3), 1e-3)
        if (not up_now) and up:
            x_top = max(x_top, xp)
        if up_now and not up:
            x_prev = xp
        up = up_now
        if up:
            target = fo
        else:
            d = max(x_top - xp, 0.0)
            lg = (1.0 - fill) * sp_est
            if lg <= 1e-6:
                target = 0.0  # incompressible liquid: TV opens as soon as the plunger starts down
            else:
                pbar = pi / max(1.0 - d / lg, 1e-3)
                target = max(0.0, ap * (p_dis - pbar)) if pbar < p_dis else 0.0
        # smooth fluid-load transfer (gas / valve dynamics)
        f_fluid += (target - f_fluid) * min(1.0, dt / tau)
        load[s] = tdyn[0] + 0.0  # surface dynamic tension; static part added below
        plunger[s], pv[s] = xp, vp
        bload[s] = tdyn[-1]
    # static surface load = buoyed rod weight (dynamic tension excludes weight, it's the deviation about the no-fluid static state)
    f_surf = load + w_buoy_total
    # last cycle only (steady state)
    sl = slice(nstep - steps_cycle, nstep)
    F = f_surf[sl]
    Y = y_pr[sl] * 12.0
    Xp = plunger[sl] * 12.0
    tf = tf_t[sl]
    fmax, fmin = float(F.max()), float(F.min())
    cbe = p["cbe"] if p["cbe"] >= 0 else 0.5 * (fmax + fmin)
    torque = tf * (F - cbe) / 1000.0  # kin-lbf
    area = float(abs(np.trapezoid(F, Y)))  # lbf*in per cycle
    prhp = area * p["spm"] / 396000.0
    q_prod = inflow["q_prod"]
    lnet = (p_dis - pi) / (0.433 * p["sg"])
    hhp = 7.36e-6 * q_prod * lnet * p["sg"]
    a_top = secs[0][3]
    smax, smin = fmax / a_top, fmin / a_top
    allow = (125000.0 / 4.0 + 0.5625 * smin) * 0.9
    unit = UNIT_SIZES[p["unit"]]
    plunger_stroke = float(Xp.max() - Xp.min())
    return dict(
        inflow=inflow, secs=secs, rod_weight_buoyed=w_buoy_total, fo=fo, f_max=fmax, f_min=fmin, cbe=float(cbe),
        torque_peak_kinlb=float(np.abs(torque).max()), torque_rating=unit[0], structure_pct=100.0 * fmax / unit[1],
        gearbox_pct=100.0 * float(np.abs(torque).max()) / unit[0], prhp=float(prhp), hhp=float(hhp), lnet=float(lnet),
        eff_pct=100.0 * hhp / prhp if prhp > 0 else 0.0, goodman_pct=100.0 * smax / allow if allow > 0 else 0.0,
        s_max_psi=float(smax), s_min_psi=float(smin), s_allow_psi=float(allow), plunger_stroke_in=plunger_stroke,
        stroke_loss_in=float(p["stroke"] - plunger_stroke), dt=float(dt), steps_cycle=int(steps_cycle),
        card=pd.DataFrame(dict(pos_in=Y, load_lbf=F)), pump_card=pd.DataFrame(dict(pos_in=Xp - Xp.min(), load_lbf=bload[sl])),
        ts=pd.DataFrame(dict(t_s=t[sl] - t[sl][0], pos_in=Y, load_lbf=F, torque_kinlb=torque, plunger_in=Xp, tf_in_rad=tf)),
        kin=pd.DataFrame(dict(phi_deg=np.degrees(phi_g), y_in=y_g * 12.0, tf=tf_g)),
    )


# =============================================================================
# 3c. WELL PATH, TORQUE & DRAG, WELL CONTROL  (educational; soft-string, no stiffness, no hydraulics-induced loads)
# =============================================================================
PIPE_TABLE = {  # name: (OD in, ID in, adjusted weight lb/ft in air incl. tool joints) - generic illustrative values
    '3-1/2" 13.3 lb/ft': (3.5, 2.764, 14.7), '5" 19.5 lb/ft': (5.0, 4.276, 21.9), '5-1/2" 21.9 lb/ft': (5.5, 4.778, 24.7),
}
BHA_LEN_FT, BHA_OD, BHA_ID, BHA_W = 300.0, 6.5, 2.8125, 100.0  # generic drill-collar section (illustrative)


def well_path(kop: float, build_rate: float, hold_inc: float, azi: float, td_md: float, step: float = 30.0) -> pd.DataFrame:
    """
    Build-and-hold profile on a minimum-curvature basis.  kop (ft MD), build_rate (deg/100 ft), hold_inc (deg), azi (deg), td_md (ft).
    Returns md, inc_deg, azi_deg, tvd, north, east, dls (deg/100 ft).
    """
    md = np.unique(np.append(np.arange(0.0, td_md, step), td_md))
    inc = np.clip((md - kop) * build_rate / 100.0, 0.0, hold_inc) if build_rate > 0 else np.zeros_like(md)
    inc = np.minimum(inc, hold_inc)
    th = np.radians(inc)
    az = np.radians(np.full_like(md, azi))
    tvd = np.zeros_like(md)
    north = np.zeros_like(md)
    east = np.zeros_like(md)
    dls = np.zeros_like(md)
    for i in range(1, len(md)):
        dl = md[i] - md[i - 1]
        cosb = np.cos(th[i] - th[i - 1]) - np.sin(th[i - 1]) * np.sin(th[i]) * (1.0 - np.cos(az[i] - az[i - 1]))
        beta = np.arccos(np.clip(cosb, -1.0, 1.0))
        rf = 1.0 if beta < 1e-9 else (2.0 / beta) * np.tan(beta / 2.0)  # minimum-curvature ratio factor
        tvd[i] = tvd[i - 1] + 0.5 * dl * (np.cos(th[i - 1]) + np.cos(th[i])) * rf
        north[i] = north[i - 1] + 0.5 * dl * (np.sin(th[i - 1]) * np.cos(az[i - 1]) + np.sin(th[i]) * np.cos(az[i])) * rf
        east[i] = east[i - 1] + 0.5 * dl * (np.sin(th[i - 1]) * np.sin(az[i - 1]) + np.sin(th[i]) * np.sin(az[i])) * rf
        dls[i] = np.degrees(beta) / dl * 100.0
    return pd.DataFrame(dict(md=md, inc_deg=inc, azi_deg=np.full_like(md, azi), tvd=tvd, north=north, east=east, dls=dls))


def validate_td(p: dict) -> list[str]:
    e = []
    if not 1000 <= p["td_md"] <= 30000:
        e.append("Total MD must be 1,000-30,000 ft.")
    if not 0 <= p["kop"] < p["td_md"]:
        e.append("Kick-off point must be within the well.")
    if not 0 <= p["hold_inc"] <= 95:
        e.append("Hold inclination must be 0-95 deg.")
    if not 0 <= p["build_rate"] <= 15:
        e.append("Build rate must be 0-15 deg/100 ft.")
    if not 0 <= p["mu"] <= 1:
        e.append("Friction factor must be 0-1.")
    if p["pipe"] not in PIPE_TABLE:
        e.append("Unknown pipe size.")
    if not p["hole_in"] > BHA_OD:
        e.append(f"Hole diameter must exceed the collar OD ({BHA_OD} in).")
    return e


def torque_drag(path: pd.DataFrame, mu: float, mw: float, pipe: str, hole_in: float, wob: float = 0.0, bit_torque: float = 0.0) -> dict:
    """
    Johancsik soft-string model, integrated bottom-up.  Modes: pick-up, slack-off, rotating (off bottom) and drilling (on bottom).
    Tensions in lbf, torque in ft-lbf.  Returns arrays per station plus surface hookloads and buckling indicators.
    """
    od, idd, w_air = PIPE_TABLE[pipe]
    bf = 1.0 - mw / 65.4
    md = path.md.values
    th = np.radians(path.inc_deg.values)
    az = np.radians(path.azi_deg.values)
    n = len(md)
    td = md[-1]

    def w_at(m):  # buoyed weight per ft at measured depth m
        return (BHA_W if m > td - BHA_LEN_FT else w_air) * bf

    ei = lambda o, i: E_STEEL_PSI * np.pi / 64.0 * (o ** 4 - i ** 4)
    out = {}
    for mode in ("pick-up", "slack-off", "rotating", "drilling"):
        t = np.zeros(n)
        tq = np.zeros(n)
        t[-1] = -wob * 1000.0 if mode == "drilling" else 0.0
        tq[-1] = bit_torque if mode == "drilling" else 0.0
        for k in range(n - 2, -1, -1):
            dl = md[k + 1] - md[k]
            thm = 0.5 * (th[k] + th[k + 1])
            dth = th[k + 1] - th[k]
            daz = az[k + 1] - az[k]
            w = w_at(0.5 * (md[k] + md[k + 1]))
            tb = t[k + 1]
            nforce = np.hypot(tb * daz * np.sin(thm), tb * dth + w * dl * np.sin(thm))
            rj = 0.5 * (od * 1.1 if md[k] < td - BHA_LEN_FT else BHA_OD) / 12.0  # contact radius (ft); tool-joint OD ~ 1.1 x pipe
            axial = w * dl * np.cos(thm)
            if mode in ("pick-up",):
                t[k] = tb + axial + mu * nforce
            elif mode == "slack-off":
                t[k] = tb + axial - mu * nforce
            else:
                t[k] = tb + axial
                tq[k] = tq[k + 1] + mu * nforce * rj
        out[mode] = (t, tq)
    # buckling thresholds (pipe body) against the compression in slack-off / drilling
    r_clear = 0.5 * (hole_in - od)
    w_in = w_air * bf / 12.0
    sin_t = np.sin(th)
    fs = np.where(sin_t > 0.05, 2.0 * np.sqrt(ei(od, idd) * w_in * np.maximum(sin_t, 1e-6) / r_clear),
                  1.94 * (ei(od, idd) * (w_in * 12.0) ** 2 / 144.0) ** (1.0 / 3.0))
    comp = {m: np.maximum(-out[m][0], 0.0) for m in ("slack-off", "drilling")}
    res = dict(md=md, tension=dict(pickup=out["pick-up"][0], slackoff=out["slack-off"][0], rotating=out["rotating"][0], drilling=out["drilling"][0]),
               torque=dict(rotating=out["rotating"][1], drilling=out["drilling"][1]),
               hookload=dict(pickup=float(out["pick-up"][0][0]), slackoff=float(out["slack-off"][0][0]), rotating=float(out["rotating"][0][0])),
               surface_torque=dict(rotating=float(out["rotating"][1][0]), drilling=float(out["drilling"][1][0])),
               fs_sin=fs, fs_hel=2.83 * fs, compression=comp,
               sinusoidal=dict((m, bool(np.any(comp[m] > fs))) for m in comp), helical=dict((m, bool(np.any(comp[m] > 2.83 * fs))) for m in comp),
               free_hang=float(sum(w_at(0.5 * (md[k] + md[k + 1])) * (md[k + 1] - md[k]) for k in range(n - 1))))
    return res


def kill_sheet(mw: float, tvd: float, sidpp: float, sicp: float, pit_gain: float, scr_psi: float, scr_spm: float,
               dp_vol_bbl: float, ann_vol_bbl: float, pump_bbl_stk: float, shoe_tvd: float, shoe_fg: float, hole_in: float, pipe_od: float) -> dict:
    """Driller's-method style kill data (field units).  KMW = MW + SIDPP / (0.052 TVD)."""
    kmw = mw + sidpp / (0.052 * tvd)
    icp = sidpp + scr_psi
    fcp = scr_psi * kmw / mw
    maasp = (shoe_fg - mw) * 0.052 * shoe_tvd
    ann_cap = (hole_in ** 2 - pipe_od ** 2) / 1029.4  # bbl/ft
    h_inf = pit_gain / ann_cap if ann_cap > 0 else 0.0
    grad_inf = 0.052 * mw - (sicp - sidpp) / h_inf if h_inf > 0 else np.nan
    kind = "unknown" if np.isnan(grad_inf) else ("gas" if grad_inf < 0.25 else "oil / mixed" if grad_inf < 0.40 else "salt water")
    stk_dp = dp_vol_bbl / pump_bbl_stk
    stk = np.linspace(0, stk_dp, 11)
    sched = np.linspace(icp, fcp, 11)
    return dict(kmw=float(kmw), icp=float(icp), fcp=float(fcp), maasp=float(maasp), influx_height_ft=float(h_inf),
                influx_grad=float(grad_inf), influx_type=kind, strokes_to_bit=float(stk_dp), time_to_bit_min=float(stk_dp / scr_spm),
                time_surface_min=float((dp_vol_bbl + ann_vol_bbl) / pump_bbl_stk / scr_spm), kmw_over_fg=bool(kmw >= shoe_fg),
                schedule=pd.DataFrame(dict(strokes=stk, dp_pressure_psi=sched)), sicp_over_maasp=bool(sicp > maasp))


# =============================================================================
# 3d. ESP ENGINE (electric submersible pump) - generic normalised pump curves, NOT a manufacturer catalogue
# =============================================================================
ESP_PUMPS = {  # name: (q_bep bpd @60 Hz, head per stage @BEP ft, eta_bep, housing OD in)
    "Small  (BEP 800 bpd)": (800.0, 24.0, 0.55, 4.00),
    "Medium (BEP 2,500 bpd)": (2500.0, 31.0, 0.66, 5.13),
    "Large  (BEP 5,000 bpd)": (5000.0, 36.0, 0.70, 5.62),
}
ESP_MOTORS_HP = [30, 45, 60, 90, 120, 150, 200, 250, 300, 400, 500]
ESP_CABLES = {"#6 Cu": 0.395, "#4 Cu": 0.249, "#2 Cu": 0.156, "#1 Cu": 0.124}  # ohm / 1000 ft at 25 C
ESP_DEFAULTS = dict(depth=6000.0, tbg_id=2.992, pt=150.0, sg=0.90, pr=2200.0, qmax=2500.0, pump="Medium (BEP 2,500 bpd)",
                    stages=120, freq=55.0, gvf=0.03, motor_v=2000.0, cable="#4 Cu", bht=200.0, hw_c=120.0, vsd=1)


def esp_stage_head(q, f, pump):
    """Per-stage head (ft) at rate q (bpd) and frequency f (Hz) by the affinity laws; normalised curve H/Hbep = 1.3 - 0.3 x^2."""
    qb, hb, _eta, _od = ESP_PUMPS[pump]
    r = f / 60.0
    x = np.asarray(q, float) / (qb * r)
    return hb * r ** 2 * np.maximum(1.3 - 0.3 * x ** 2, 0.0)


def esp_stage_eff(q, f, pump):
    qb, _hb, eb, _od = ESP_PUMPS[pump]
    x = np.asarray(q, float) / (qb * f / 60.0)
    return eb * np.maximum(1.0 - 1.25 * (x - 1.0) ** 2, 0.02)


def esp_stage_hp(q, f, pump, sg):
    """Brake horsepower per stage: hp = q[bpd] H[ft] SG / (135,770 eta)."""
    return np.asarray(q, float) * esp_stage_head(q, f, pump) * sg / (135770.0 * esp_stage_eff(q, f, pump))


def esp_gas_derate(gvf):
    """Head multiplier vs. free-gas volume fraction at the intake (illustrative; no gas separator / handler)."""
    return np.maximum(1.0 - (np.asarray(gvf, float) / 0.30) ** 1.2, 0.0)


def hazen_williams_ft_per_100(q_bpd, id_in, c=120.0):
    gpm = np.asarray(q_bpd, float) / 34.286
    return 2.083 * (100.0 / c) ** 1.85 * gpm ** 1.85 / id_in ** 4.8655


def validate_esp(p: dict) -> list[str]:
    e = []
    if not 2000 <= p["depth"] <= 12000:
        e.append("Pump depth must be 2,000-12,000 ft.")
    if p["pump"] not in ESP_PUMPS:
        e.append("Unknown pump.")
    elif ESP_PUMPS[p["pump"]][3] + 0.3 > 7.0:
        e.append("Pump housing does not fit the casing.")
    if not 10 <= p["stages"] <= 600:
        e.append("Stages must be 10-600.")
    if not 30.0 <= p["freq"] <= 70.0:
        e.append("Frequency must be 30-70 Hz.")
    if not 0.0 <= p["gvf"] < 1.0:
        e.append("Free-gas fraction must be 0-1.")
    if not 0.6 <= p["sg"] <= 1.2:
        e.append("Fluid specific gravity must be 0.6-1.2.")
    if p["pr"] <= 0 or p["qmax"] <= 0 or p["pt"] < 0 or p["tbg_id"] <= 0:
        e.append("Reservoir pressure, qmax, tubing ID must be positive and tubing pressure non-negative.")
    if p["cable"] not in ESP_CABLES:
        e.append("Unknown cable size.")
    return e


def esp_system_head(q, p):
    """Total dynamic head required (ft) at rate q (bpd): lift to surface + tubing friction + wellhead pressure - intake pressure (Pwf from Vogel)."""
    q = np.asarray(q, float)
    x = np.clip(q / p["qmax"], 0.0, 1.0)
    pwf = p["pr"] * (-0.2 + np.sqrt(0.04 + 3.2 * (1.0 - x))) / 1.6
    return p["depth"] + (p["pt"] - pwf) / (0.433 * p["sg"]) + hazen_williams_ft_per_100(q, p["tbg_id"], p["hw_c"]) * p["depth"] / 100.0


def esp_pump_head(q, p, f=None):
    f = p["freq"] if f is None else f
    return p["stages"] * esp_stage_head(q, f, p["pump"]) * esp_gas_derate(p["gvf"])


def esp_operating_point(p: dict, f=None) -> dict:
    """Intersection of pump head and system head curves by bisection (None rate if the pump cannot lift)."""
    f = p["freq"] if f is None else f
    lo, hi = 1e-6, p["qmax"] * 0.999
    g = lambda q: float(esp_pump_head(q, p, f) - esp_system_head(q, p))  # noqa: E731
    if g(lo) <= 0:
        return dict(q=0.0, head=float(esp_pump_head(lo, p, f)), lifts=False)
    if g(hi) >= 0:
        return dict(q=hi, head=float(esp_system_head(hi, p)), lifts=True, inflow_limited=True)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
    q = 0.5 * (lo + hi)
    return dict(q=float(q), head=float(esp_system_head(q, p)), lifts=True, inflow_limited=False)


def esp_simulate(params: dict) -> dict:
    p = {**ESP_DEFAULTS, **params}
    errs = validate_esp(p)
    if errs:
        raise ValueError(" ".join(errs))
    op = esp_operating_point(p)
    q = op["q"]
    qb, hb, eb, od = ESP_PUMPS[p["pump"]]
    r = p["freq"] / 60.0
    x = q / (qb * r) if q > 0 else 0.0
    eta = float(esp_stage_eff(q, p["freq"], p["pump"])) if q > 0 else 0.0
    shaft_hp = float(p["stages"] * esp_stage_hp(q, p["freq"], p["pump"], p["sg"])) if q > 0 else 0.0
    hhp = q * op["head"] * p["sg"] / 135770.0
    motor_hp = next((m for m in ESP_MOTORS_HP if m >= shaft_hp / 0.95), ESP_MOTORS_HP[-1])
    load = 100.0 * shaft_hp / motor_hp
    amps = shaft_hp * 745.7 / (np.sqrt(3) * p["motor_v"] * 0.85 * 0.85) if shaft_hp > 0 else 0.0  # pf 0.85, motor eff 0.85
    rc = ESP_CABLES[p["cable"]] * (1.0 + 0.00393 * (p["bht"] - 25.0))
    clen = p["depth"] + 100.0
    vdrop = np.sqrt(3) * amps * rc * clen / 1000.0
    kva = np.sqrt(3) * (p["motor_v"] + vdrop) * amps / 1000.0
    sweep = []
    for f in np.arange(30.0, 70.5, 2.5):
        o = esp_operating_point(p, f)
        sh = float(p["stages"] * esp_stage_hp(o["q"], f, p["pump"], p["sg"])) if o["q"] > 0 else 0.0
        sweep.append(dict(freq=float(f), q=o["q"], head=o["head"], shaft_hp=sh))
    flags = []
    if q <= 0:
        flags.append("Pump cannot lift fluid to surface at this frequency / stage count.")
    else:
        if x < 0.6:
            flags.append("Operating below the recommended range (downthrust, recirculation, heating).")
        if x > 1.3:
            flags.append("Operating above the recommended range (upthrust, rapid wear).")
    if p["gvf"] > 0.15:
        flags.append("Free-gas fraction above ~15 %: gas-lock risk. Use a gas separator / handler.")
    if load > 100:
        flags.append("Motor overloaded: select a larger motor or reduce stages / frequency.")
    if 100.0 * vdrop / p["motor_v"] > 15.0:
        flags.append("Cable voltage drop above 15 %: use a larger cable.")
    return dict(op=op, q=float(q), head=float(op["head"]), x_bep=float(x), eta_stage=eta, shaft_hp=shaft_hp, hhp=float(hhp),
                eff_overall=float(100.0 * hhp / max(shaft_hp / 0.85, 1e-9)) if shaft_hp > 0 else 0.0, motor_hp=int(motor_hp), motor_load=float(load),
                amps=float(amps), vdrop=float(vdrop), vdrop_pct=float(100.0 * vdrop / p["motor_v"]), kva=float(kva), od=float(od),
                sweep=pd.DataFrame(sweep), flags=flags, params=p)


def esp_required_stages(q_target: float, p: dict) -> int:
    """Stages needed at the current frequency to deliver q_target against the system head (ignores gas derate if gvf=0)."""
    h1 = float(esp_stage_head(q_target, p["freq"], p["pump"])) * float(esp_gas_derate(p["gvf"]))
    return int(np.ceil(float(esp_system_head(q_target, p)) / h1)) if h1 > 0 else 0


# =============================================================================
# 3e. OIL RESERVOIR SIMULATION: 2-D two-phase (oil-water), slightly compressible undersaturated oil, structure with an oil-water contact,
#     primary depletion or peripheral waterflood.  IMPES, no gravity / capillarity, dead oil (no free gas).  Educational.
# =============================================================================
RES_LAYOUTS = {
    "depletion1": "Primary depletion - 1 producer",
    "depletion4": "Primary depletion - 4 producers",
    "flood": "Peripheral waterflood - 3 injectors down-dip + 4 producers",
    "corner": "Corner injector / corner producer (test layout)",
}
RES_DEFAULTS = dict(nx=30, ny=30, dx=120.0, h=50.0, phi=0.20, k_md=120.0, hetero=0.7, corr=3.0, seed=11, swc=0.20, sor=0.25,
                    krw0=0.35, kro0=0.90, nw=2.5, no=2.0, mu_w=0.5, mu_o=1.5, bo_i=1.25, bw=1.01, co=1.2e-5, cw=3.0e-6, cf=4.0e-6,
                    p_i=4200.0, p_b=2000.0, pwf=2400.0, q_cap=350.0, q_inj=1800.0, years=10.0, aquifer=0.003, layout="depletion4", relief=400.0,
                    z_top=8000.0, owc_frac=0.80, tz=40.0, rw=0.35, snaps=8, anis=1.0, dt_max=30.0, pvi_max=None)


def validate_res(p: dict) -> list[str]:
    e = []
    if p["layout"] not in RES_LAYOUTS:
        e.append("Unknown well layout.")
    if not (10 <= p["nx"] <= 60 and 1 <= p["ny"] <= 60):
        e.append("Grid must be 10-60 cells in x and 1-60 in y.")
    if p["dx"] <= 0 or p["h"] <= 0 or not 0.02 <= p["phi"] <= 0.45 or p["k_md"] <= 0:
        e.append("Cell size, thickness, porosity (0.02-0.45) and permeability must be positive.")
    if not 0 <= p["swc"] < 1 or not 0 <= p["sor"] < 1 or p["swc"] + p["sor"] >= 0.95:
        e.append("Connate water and residual oil must satisfy Swc + Sor < 0.95.")
    if p["mu_w"] <= 0 or p["mu_o"] <= 0 or p["nw"] <= 0 or p["no"] <= 0 or not 0 < p["krw0"] <= 1 or not 0 < p["kro0"] <= 1:
        e.append("Viscosities, Corey exponents and end-point relative permeabilities must be positive (<= 1).")
    if p["bo_i"] < 1.0 or p["co"] < 0 or p["cw"] < 0 or p["cf"] < 0:
        e.append("Bo must be >= 1 and compressibilities non-negative.")
    if not p["p_i"] > p["pwf"] > 0:
        e.append("Initial pressure must exceed the producers' bottom-hole pressure (which must be positive).")
    if p["q_inj"] < 0 or p["q_cap"] <= 0 or p["years"] <= 0:
        e.append("Injection rate must be non-negative; well rate cap and simulated years must be positive.")
    if not 0 <= p["aquifer"] <= 5:
        e.append("Aquifer strength must be 0-5.")
    if not 0 <= p["hetero"] <= 2.5:
        e.append("Heterogeneity (log-sigma) must be 0-2.5.")
    if not 0.0 < p["owc_frac"] <= 1.5:
        e.append("OWC position must be 0-1.5 of the structural relief (above 1 = no water leg).")
    return e


def corey_krw_kro(sw, p):
    sn = np.clip((sw - p["swc"]) / (1.0 - p["swc"] - p["sor"]), 0.0, 1.0)
    return p["krw0"] * sn ** p["nw"], p["kro0"] * (1.0 - sn) ** p["no"]


def fractional_flow(sw, p):
    krw, kro = corey_krw_kro(sw, p)
    lw, lo = krw / p["mu_w"], kro / p["mu_o"]
    return lw / np.maximum(lw + lo, 1e-30)


def welge_breakthrough_pvi(p: dict) -> float:
    """Analytic 1-D Buckley-Leverett (Welge) pore volumes injected at breakthrough: 1 / fw'(Sw_front)."""
    s = np.linspace(p["swc"] + 1e-6, 1.0 - p["sor"], 4000)
    fw = fractional_flow(s, p)
    slope = fw / np.maximum(s - p["swc"], 1e-12)
    i = int(np.argmax(slope))
    return float(1.0 / np.gradient(fw, s)[i])


def permeability_field(p: dict) -> np.ndarray:
    rng = np.random.default_rng(int(p["seed"]))
    z = rng.normal(size=(p["ny"], p["nx"]))
    if p["corr"] > 0 and p["hetero"] > 0:
        from scipy.ndimage import gaussian_filter
        z = gaussian_filter(z, sigma=p["corr"], mode="wrap")
        z = (z - z.mean()) / max(z.std(), 1e-12)
    else:
        z = np.zeros_like(z)
    return p["k_md"] * np.exp(p["hetero"] * z - 0.5 * p["hetero"] ** 2)


def structure_depth(p: dict) -> np.ndarray:
    """Top-structure depth (ft TVDSS): tilted block with a gentle anticline; deeper towards +j (down-dip)."""
    ny, nx = int(p["ny"]), int(p["nx"])
    jj = np.linspace(0, 1, ny)[:, None] * np.ones((1, nx))
    ii = np.ones((ny, 1)) * np.linspace(0, 1, nx)[None, :]
    return p["z_top"] + p["relief"] * (0.85 * jj + 0.15 * (1 - np.sin(np.pi * ii)) * jj)


def res_wells(p: dict):
    nx, ny = int(p["nx"]), int(p["ny"])
    lay = p["layout"]
    c = lambda f: int(round(f * (nx - 1)))  # noqa: E731
    r = lambda f: int(round(f * (ny - 1)))  # noqa: E731
    if lay == "depletion1":
        return [("P1", "producer", r(0.30), c(0.50))]
    if lay == "depletion4":
        return [(f"P{k + 1}", "producer", r(a), c(b)) for k, (a, b) in enumerate(((0.25, 0.25), (0.25, 0.75), (0.55, 0.25), (0.55, 0.75)))]
    if lay == "flood":
        w = [(f"P{k + 1}", "producer", r(a), c(b)) for k, (a, b) in enumerate(((0.20, 0.20), (0.20, 0.80), (0.45, 0.30), (0.45, 0.70)))]
        return w + [(f"I{k + 1}", "injector", r(0.88), c(b)) for k, b in enumerate((0.20, 0.50, 0.80))]
    return [("I1", "injector", 0, 0), ("P1", "producer", ny - 1, nx - 1)]


def _res_run(p: dict) -> dict:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    nx, ny = int(p["nx"]), int(p["ny"])
    n = nx * ny
    dx = dy = float(p["dx"])
    h = float(p["h"])
    k = permeability_field(p)
    kx, ky = k, k * p["anis"]
    depth = structure_depth(p)
    owc = p["z_top"] + p["owc_frac"] * p["relief"]
    # initial saturations from the contact: oil above the OWC (connate water), a short transition, water leg below it
    s_w_leg = 1.0 - p["sor"]
    sw0 = np.clip(p["swc"] + (s_w_leg - p["swc"]) * np.clip((depth - (owc - p["tz"])) / max(p["tz"], 1e-6), 0.0, 1.0), p["swc"], s_w_leg)
    if p["layout"] == "corner":
        sw0 = np.full_like(sw0, p["swc"])
    phi = np.full((ny, nx), float(p["phi"]))
    pv0 = (phi * dx * dy * h / 5.6146).ravel()  # rb
    idx = np.arange(n).reshape(ny, nx)
    c = 0.001127 * h
    kxf = 2.0 * kx[:, :-1] * kx[:, 1:] / (kx[:, :-1] + kx[:, 1:])
    kyf = 2.0 * ky[:-1, :] * ky[1:, :] / (ky[:-1, :] + ky[1:, :])
    gx = c * kxf * dy / dx
    gy = c * kyf * dx / dy
    wells = res_wells(p)
    wcell = [int(idx[j, i]) for (_nm, _t, j, i) in wells]
    wtype = [t for (_n, t, _j, _i) in wells]
    re = 0.208 * dx
    wi_geo = [2.0 * np.pi * 0.001127 * h * float(k[j, i]) / np.log(re / p["rw"]) for (_n, _t, j, i) in wells]
    aq_cells = idx[-1, :].astype(int) if (p["aquifer"] > 0 and p["layout"] != "corner") else np.array([], dtype=int)
    aq_t = p["aquifer"] * c * k[-1, :] * (dx / dy) * (p["krw0"] / p["mu_w"]) if len(aq_cells) else np.array([])  # edge-water influx index (rb/d/psi)
    p_cur = np.full(n, float(p["p_i"]))
    sw = sw0.ravel().copy()
    bo = lambda pp: p["bo_i"] * np.exp(p["co"] * (p["p_i"] - pp))  # noqa: E731
    cinj = [w for w, t in zip(range(len(wells)), wtype) if t == "injector"]
    qinj_each = float(p["q_inj"]) / max(len(cinj), 1)
    oil_in_place0 = float((pv0 * (1.0 - sw) / bo(p_cur)).sum())
    ooip = float((pv0 * (1.0 - sw) / p["bo_i"]).sum())  # STB (initial reservoir-oil, original Bo)
    t = 0.0
    t_end = float(p["years"]) * 365.25
    cum = dict(o=0.0, w=0.0, inj=0.0, prod_rb=0.0)
    well_cum = np.zeros(len(wells))
    hist, snaps = [], []
    snap_times = np.linspace(0.0, t_end, int(p["snaps"]) + 1)[1:]
    k_snap = 0
    snaps.append(dict(t=0.0, sw=sw.reshape(ny, nx).copy(), p=p_cur.reshape(ny, nx).copy()))
    s_grid = np.linspace(p["swc"], 1.0 - p["sor"], 2000)
    dfmax = float(np.max(np.gradient(fractional_flow(s_grid, p), s_grid)))
    dt_p = 1.0
    steps = 0
    max_steps = 20000
    while t < t_end - 1e-9 and steps < max_steps:
        dt = min(dt_p, p["dt_max"], t_end - t)
        krw, kro = corey_krw_kro(sw, p)
        lw, lo = krw / p["mu_w"], kro / p["mu_o"]
        lt = lw + lo
        L2 = lt.reshape(ny, nx)
        tx = gx * 0.5 * (L2[:, :-1] + L2[:, 1:])
        ty = gy * 0.5 * (L2[:-1, :] + L2[1:, :])
        so = 1.0 - sw
        ct = p["co"] * so + p["cw"] * sw + p["cf"]
        acc = pv0 * ct / dt
        rows, cols, vals = [], [], []
        diag = acc.copy()
        a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-tx.ravel(), -tx.ravel()]
        np.add.at(diag, a, tx.ravel()); np.add.at(diag, b, tx.ravel())
        a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-ty.ravel(), -ty.ravel()]
        np.add.at(diag, a, ty.ravel()); np.add.at(diag, b, ty.ravel())
        rhs = acc * p_cur
        capped = [False] * len(wells)
        for _it in range(4):  # switch producers that exceed the rate cap to rate control
            d2 = diag.copy()
            r2 = rhs.copy()
            if len(aq_cells):
                d2[aq_cells] += aq_t
                r2[aq_cells] += aq_t * p["p_i"]
            for w, (cell, ty_) in enumerate(zip(wcell, wtype)):
                if ty_ == "injector":
                    r2[cell] += qinj_each
                else:
                    wi = wi_geo[w] * lt[cell]
                    if capped[w]:
                        r2[cell] -= p["q_cap"] * bo(p_cur[cell])  # cap in stb/d -> rb/d
                    else:
                        d2[cell] += wi
                        r2[cell] += wi * p["pwf"]
            A = coo_matrix((np.concatenate(vals + [d2]), (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))), shape=(n, n)).tocsr()
            pr = spsolve(A, r2)
            changed = False
            for w, (cell, ty_) in enumerate(zip(wcell, wtype)):
                if ty_ == "producer" and not capped[w]:
                    q_rb = wi_geo[w] * lt[cell] * (pr[cell] - p["pwf"])
                    if q_rb / bo(pr[cell]) > p["q_cap"]:
                        capped[w] = True
                        changed = True
            if not changed:
                break
        P = pr.reshape(ny, nx)
        ux = tx * (P[:, :-1] - P[:, 1:])
        uy = ty * (P[:-1, :] - P[1:, :])
        qprod = np.zeros(len(wells))
        for w, (cell, ty_) in enumerate(zip(wcell, wtype)):
            if ty_ == "producer":
                qprod[w] = p["q_cap"] * bo(p_cur[cell]) if capped[w] else wi_geo[w] * lt[cell] * (pr[cell] - p["pwf"])
        fw = (lw / np.maximum(lt, 1e-30))
        out = np.zeros((ny, nx))
        out[:, :-1] += np.maximum(ux, 0); out[:, 1:] += np.maximum(-ux, 0)
        out[:-1, :] += np.maximum(uy, 0); out[1:, :] += np.maximum(-uy, 0)
        out_flat = out.ravel()
        for w, cell in enumerate(wcell):
            if wtype[w] == "producer":
                out_flat[cell] += qprod[w]
        dt_cfl = 0.8 * float((pv0 / np.maximum(out_flat * dfmax, 1e-9)).min())
        nsub = int(np.clip(np.ceil(dt / max(dt_cfl, 1e-6)), 1, 40))
        dts = dt / nsub
        dp = pr - p_cur
        q_aq = np.zeros(n)
        if len(aq_cells):
            q_aq[aq_cells] = aq_t * (p["p_i"] - pr[aq_cells])  # water entering from the aquifer (rb/d)
        fw2 = fw.reshape(ny, nx)
        for sub in range(nsub):
            wxs = ux * np.where(ux >= 0, fw2[:, :-1], fw2[:, 1:])
            wys = uy * np.where(uy >= 0, fw2[:-1, :], fw2[1:, :])
            net = np.zeros((ny, nx))
            net[:, :-1] += wxs; net[:, 1:] -= wxs
            net[:-1, :] += wys; net[1:, :] -= wys
            nf = net.ravel()
            nf -= q_aq
            for w, cell in enumerate(wcell):
                if wtype[w] == "injector":
                    nf[cell] -= qinj_each
                else:
                    nf[cell] += qprod[w] * fw[cell]
            sw = sw - dts * nf / pv0
            if sub == 0:
                sw = sw - sw * (p["cw"] + p["cf"]) * dp  # water/rock compressibility term (once per pressure step)
            sw = np.clip(sw, p["swc"], 1.0 - p["sor"])
            fw = fractional_flow(sw, p)
            fw2 = fw.reshape(ny, nx)
        p_cur = pr
        qo = qw = 0.0
        for w, cell in enumerate(wcell):
            if wtype[w] == "producer":
                qo_w = qprod[w] * (1.0 - fw[cell]) / bo(pr[cell])
                qw_w = qprod[w] * fw[cell] / p["bw"]
                qo += qo_w; qw += qw_w
                well_cum[w] += qo_w * dt
                cum["prod_rb"] += qprod[w] * dt
        cum["o"] += qo * dt
        cum["w"] += qw * dt
        cum["inj"] += qinj_each * len(cinj) * dt
        cum["aq"] = cum.get("aq", 0.0) + float(q_aq.sum()) * dt
        t += dt
        pav = float((pr * pv0).sum() / pv0.sum())
        oip = float((pv0 * (1.0 - sw) / bo(pr)).sum())
        hist.append((t / 365.25, qo, qw, qinj_each * len(cinj), cum["o"], qw / max(qo + qw, 1e-12), cum["o"] / ooip, pav,
                     float(np.mean([pr[c_] for c_, ty_ in zip(wcell, wtype) if ty_ == "producer"])), float(pr.min()), oip, cum["inj"]))
        if k_snap < len(snap_times) and t >= snap_times[k_snap] - 1e-9:
            snaps.append(dict(t=float(t / 365.25), sw=sw.reshape(ny, nx).copy(), p=P.copy()))
            k_snap += 1
        steps += 1
        dt_p = min(dt_p * 1.25, p["dt_max"])
    ts = pd.DataFrame(hist, columns=["years", "q_oil", "q_water", "q_inj", "cum_oil", "water_cut", "recovery", "p_avg", "p_prod", "p_min", "oip_stb", "cum_inj_rb"])
    # oil balance: oil produced + oil still in place (at current Bo) must equal the original oil in place (to first order in compressibility)
    mb_err = abs(ts.cum_oil.iloc[-1] + ts.oip_stb.iloc[-1] - oil_in_place0) / oil_in_place0
    bt = float(ts.years[ts.water_cut > 0.05].iloc[0]) if (ts.water_cut > 0.05).any() else float("nan")
    bt_pvi = float("nan")
    if cum["inj"] > 0:
        pvt_ = float(pv0.sum())
        cinjrb = ts.cum_inj_rb.values
        ok = np.where(ts.water_cut.values > 0.05)[0]
        bt_pvi = float(cinjrb[ok[0]] / pvt_) if len(ok) else float("nan")
    return dict(ts=ts, snaps=snaps, k=k, depth=depth, owc=float(owc), wells=[(nm, ty_, j, i) for (nm, ty_, j, i) in wells], well_cum=well_cum,
                mass_balance_err=float(mb_err), ooip=float(ooip), pvt=float(pv0.sum()), steps=steps, bt_years=bt, bt_pvi=bt_pvi,
                final_recovery=float(ts.recovery.iloc[-1]), below_pb=bool(ts.p_min.min() < p["p_b"]),
                vrr=float(cum["inj"] / max(cum["prod_rb"], 1e-9)), aq_frac=float(cum.get("aq", 0.0) / max(cum["prod_rb"], 1e-9)), capped_any=bool(any(capped)), params=p)


@functools.lru_cache(maxsize=16)
def _res_cached(items: tuple):
    return _res_run(dict(items))


def res_simulate(params: dict) -> dict:
    p = {**RES_DEFAULTS, **params}
    errs = validate_res(p)
    if errs:
        raise ValueError(" ".join(errs))
    return _res_cached(tuple(sorted(p.items())))


# =============================================================================
# 4. MACHINE LEARNING (synthetic data, Isolation Forest, baseline)
# =============================================================================
FEATURES = ["torque_dev", "spp_dev", "flow_dev", "torque_cv", "rpm_cv", "torque_ratio", "spp_ratio"]
BASE_CH = ["torque_kftlbf", "spp_psi", "flow_gpm"]
ANOM_TYPES = ["elevated_torque", "pressure_deviation", "flow_deviation", "stick_slip"]
ML_SEED = 42
WARMUP_SAMPLES = 15  # rolling features need >= 15 samples (30 s) before they are defined
ML_VERSION = "3"  # bump when features / generator change: part of every ML cache key


def build_features(df: pd.DataFrame, geom: dict | None = None) -> pd.DataFrame:
    """
    Features for ONE run, derived only from simulated sensors and control inputs (labels are never used -> no leakage).
      torque_dev / spp_dev : deviation from the run's own trailing 4-minute median (causal: past samples only)
      flow_dev             : (measured - set-point) / set-point
      torque_cv / rpm_cv   : 16 s rolling coefficient of variation (stick-slip-like signatures)
      torque_ratio / spp_ratio : measured / model-expected value (physics residuals)
    Run-relative deviations remove between-run differences (e.g. formation resistance) that otherwise swamp the signal.
    """
    g = {k: SIM_DEFAULTS[k] for k in GEOM_KEYS}
    if geom:
        g.update({k: geom[k] for k in GEOM_KEYS if k in geom})
    f = pd.DataFrame(index=df.index)
    trail = lambda c: df[c].rolling(120, min_periods=15).median().shift(1)  # noqa: E731
    f["torque_dev"] = (df["torque_kftlbf"] / trail("torque_kftlbf") - 1.0).fillna(0.0)
    f["spp_dev"] = (df["spp_psi"] / trail("spp_psi") - 1.0).fillna(0.0)
    setp = df["flow_set_gpm"].replace(0, np.nan)
    f["flow_dev"] = ((df["flow_gpm"] - df["flow_set_gpm"]) / setp).fillna(0.0)
    tr = df["torque_kftlbf"].rolling(8, min_periods=2)
    f["torque_cv"] = (tr.std() / tr.mean()).fillna(0.0)
    rr = df["rpm"].rolling(8, min_periods=2)
    f["rpm_cv"] = (rr.std() / rr.mean().replace(0, np.nan)).fillna(0.0)
    f["torque_ratio"] = df["torque_kftlbf"] / expected_torque(df["wob_klbf"], df["rpm"], df["depth_ft"])
    exp = expected_spp(df["flow_gpm"].values, df["mw_ppg"].values, df["depth_ft"].values, g)
    f["spp_ratio"] = df["spp_psi"] / exp
    return f


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator, n_events: int) -> pd.DataFrame:
    """Overlay labelled synthetic anomalies on measured channels (non-overlapping segments)."""
    df = df.copy()
    n = len(df)
    df["label"] = 0
    df["anom_type"] = "normal"
    if n_events <= 0:
        return df
    # place segments sequentially with random gaps
    seg_len = rng.integers(25, 46, n_events)
    slack = n - int(seg_len.sum()) - 120
    gaps = np.sort(rng.integers(60, max(61, slack + 60), n_events)) if slack > 0 else np.arange(n_events) * 100
    cursor = 60
    kinds = rng.choice(ANOM_TYPES, n_events, replace=True)
    tt = df["t_s"].values
    for i in range(n_events):
        start = min(cursor + int(rng.integers(20, 120)), n - int(seg_len[i]) - 1)
        end = start + int(seg_len[i])
        cursor = end
        s = slice(start, end)
        kind = kinds[i]
        if kind == "elevated_torque":
            df.loc[df.index[s], "torque_kftlbf"] *= rng.uniform(1.35, 1.8)
        elif kind == "pressure_deviation":
            m = rng.choice([rng.uniform(1.25, 1.5), rng.uniform(0.7, 0.85)])  # plugging / washout-like
            df.loc[df.index[s], "spp_psi"] *= m
        elif kind == "flow_deviation":
            m = rng.uniform(0.75, 0.88)
            df.loc[df.index[s], "flow_gpm"] *= m
            df.loc[df.index[s], "spp_psi"] *= m ** 1.8  # consistent pressure response
        else:  # stick_slip
            a = rng.uniform(0.25, 0.45)
            ph = rng.uniform(0, 2 * np.pi)
            wave = np.sin(2 * np.pi * tt[s] / SS_PERIOD + ph)
            df.loc[df.index[s], "torque_kftlbf"] *= 1.0 + a * wave
            df.loc[df.index[s], "rpm"] *= 1.0 + 0.6 * a * np.sin(2 * np.pi * tt[s] / SS_PERIOD + ph + 0.5)
        df.loc[df.index[s], "label"] = 1
        df.loc[df.index[s], "anom_type"] = kind
    return df


def generate_run(seed: int, anomalous: bool) -> pd.DataFrame:
    """One synthetic run: operating condition drawn from the seed, measured sensors, optional labelled anomalies."""
    rng = np.random.default_rng(10_000 + seed)
    params = dict(
        wob=rng.uniform(15, 35), rpm=rng.uniform(80, 160), flow=rng.uniform(400, 700), mw=rng.uniform(9.0, 12.0),
        fr=rng.uniform(0.8, 1.5), fric=0.0, ss=0.0, duration_min=30.0, start_depth=rng.uniform(5000, 11000),
        noise=rng.uniform(0.008, 0.015), wander=2.0, seed=seed,
    )
    df = simulate_drilling(params)
    df = _inject_anomalies(df, rng, int(rng.integers(2, 5)) if anomalous else 0)
    feats = build_features(df)
    out = pd.concat([df, feats], axis=1)
    out["run_id"] = seed
    return out


def generate_dataset(n_train=16, n_test=12, n_test_normal=2):
    """Training = normal runs only (seeds 1..); test = independent seeds (1000..) with anomalies + a few normal runs."""
    train = pd.concat([generate_run(s, False) for s in range(1, n_train + 1)], ignore_index=True)
    test_runs = [generate_run(1000 + s, s >= n_test_normal) for s in range(n_test)]
    test = pd.concat(test_runs, ignore_index=True)
    return train, test


@dataclass
class Detector:
    model: IsolationForest
    threshold: float
    mu: dict
    sd: dict
    n_estimators: int
    pct: float
    n_train_rows: int

    def scores(self, feats: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (negated sklearn score_samples). NOT a calibrated probability."""
        return -self.model.score_samples(feats[FEATURES].values)

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return (self.scores(feats) > self.threshold).astype(int)

    def baseline_predict(self, df: pd.DataFrame, k: float = 3.0) -> np.ndarray:
        flag = np.zeros(len(df), dtype=bool)
        for c in BASE_CH:
            z = (df[c].values - self.mu[c]) / self.sd[c]
            flag |= np.abs(z) > k
        return flag.astype(int)


def train_detector(train: pd.DataFrame, n_estimators=150, pct=98.0, seed=ML_SEED) -> Detector:
    model = IsolationForest(n_estimators=int(n_estimators), max_samples=256, contamination="auto",
                            random_state=seed).fit(train[FEATURES].values)
    s = -model.score_samples(train[FEATURES].values)
    thr = float(np.quantile(s, pct / 100.0))
    return Detector(model, thr, {c: float(train[c].mean()) for c in BASE_CH}, {c: float(train[c].std()) for c in BASE_CH},
                    int(n_estimators), float(pct), len(train))


def eval_binary(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return dict(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        tn=tn, fp=fp, fn=fn, tp=tp, flagged=int(y_pred.sum()),
    )


def event_recall(df: pd.DataFrame, pred: np.ndarray) -> tuple[int, int]:
    """(# true anomaly segments overlapped by >=1 flagged sample, # true segments)."""
    hit = total = 0
    d = df.reset_index(drop=True)
    d["pred"] = pred
    for _, run in d.groupby("run_id"):
        lab = run["label"].values
        pr = run["pred"].values
        starts = np.where((lab == 1) & (np.concatenate([[0], lab[:-1]]) == 0))[0]
        for s0 in starts:
            e0 = s0
            while e0 < len(lab) and lab[e0] == 1:
                e0 += 1
            total += 1
            hit += int(pr[s0:e0].any())
    return hit, total


@st.cache_data(show_spinner=False)
def cached_ml_data(n_train: int = 16, n_test: int = 12, ml_version: str = ML_VERSION):
    return generate_dataset(n_train, n_test)


@st.cache_resource(show_spinner=False, max_entries=16)
def cached_detector(n_estimators: int = 150, pct: float = 98.0, n_train: int = 16, n_test: int = 12, ml_version: str = ML_VERSION) -> Detector:
    train, _ = cached_ml_data(n_train, n_test)
    return train_detector(train, n_estimators, pct)


# =============================================================================
# 5. 3D GEOMETRY (procedural, lightweight)
# =============================================================================
CAMERAS = {
    "Isometric": dict(eye=dict(x=1.05, y=-1.25, z=0.62), center=dict(x=0, y=0, z=0.05), up=dict(x=0, y=0, z=1)),
    "Front": dict(eye=dict(x=0.0, y=-1.7, z=0.35), center=dict(x=0, y=0, z=0.05), up=dict(x=0, y=0, z=1)),
    "Side": dict(eye=dict(x=1.7, y=0.0, z=0.35), center=dict(x=0, y=0, z=0.05), up=dict(x=0, y=0, z=1)),
    "Top": dict(eye=dict(x=0.0, y=0.01, z=1.9), center=dict(x=0, y=0, z=0.0), up=dict(x=0, y=1, z=0)),
    "Rig floor close-up": dict(eye=dict(x=0.45, y=-0.55, z=-0.05), center=dict(x=0, y=0, z=-0.12), up=dict(x=0, y=0, z=1)),
}
PRESET_NAMES = list(CAMERAS)


GFX_SCALE = {"Standard": 1.0, "High": 1.6, "Ultra": 3.0}


def gfx_name() -> str:
    try:
        v = st.session_state.get("w_gfx", "Ultra")
    except Exception:  # no Streamlit runtime (unit tests)
        v = "High"
    return v if v in GFX_SCALE else "Ultra"


def gq() -> float:
    """Mesh-density multiplier for the chosen graphics quality (more radial segments on curved parts)."""
    return GFX_SCALE[gfx_name()]


class MeshBuilder:
    """Accumulates triangles for a single Mesh3d trace."""

    def __init__(self):
        self.x, self.y, self.z, self.i, self.j, self.k = [], [], [], [], [], []

    def _add(self, verts, faces):
        off = len(self.x)
        v = np.asarray(verts, dtype=float)
        self.x.extend(v[:, 0]); self.y.extend(v[:, 1]); self.z.extend(v[:, 2])
        for a, b, c in faces:
            self.i.append(a + off); self.j.append(b + off); self.k.append(c + off)

    def box(self, cx, cy, cz, sx, sy, sz):
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        v = [(cx + dx * hx, cy + dy * hy, cz + dz * hz) for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
        f = [(0, 1, 3), (0, 3, 2), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 3, 7), (1, 7, 5)]
        self._add(v, f)

    @staticmethod
    def _map(axis, cx, cy, cz, u, v, w):
        if axis == "z":
            return cx + u, cy + v, cz + w
        if axis == "x":
            return cx + w, cy + u, cz + v
        return cx + v, cy + w, cz + u  # axis "y"

    def frustum(self, cx, cy, cz, length, r0, r1, axis="z", n=20, mod=None, phase=0.0):
        """Cylinder / cone from base (cx,cy,cz) extending `length` along axis. mod(theta)->radius factor."""
        n = max(6, int(round(n * gq())))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        fac = np.ones(n) if mod is None else mod(th + phase)
        verts = []
        for w, r in ((0.0, r0), (length, r1)):
            for t_, fct in zip(th, fac):
                verts.append(self._map(axis, cx, cy, cz, r * fct * np.cos(t_), r * fct * np.sin(t_), w))
        verts.append(self._map(axis, cx, cy, cz, 0, 0, 0.0))
        verts.append(self._map(axis, cx, cy, cz, 0, 0, length))
        faces = []
        for a in range(n):
            b = (a + 1) % n
            faces += [(a, b, n + b), (a, n + b, n + a), (2 * n, b, a), (2 * n + 1, n + a, n + b)]
        self._add(verts, faces)

    def tube(self, z0, z1, r_in, r_out, th0=0.0, th1=2 * np.pi, n=36):
        """Vertical tube (annular sector) between z0 and z1; partial sector gives a cutaway."""
        m = max(4, int(n * gq() * (th1 - th0) / (2 * np.pi)))
        th = np.linspace(th0, th1, m + 1)
        rings = [(r_out, z0), (r_out, z1), (r_in, z0), (r_in, z1)]
        verts = [(r * np.cos(t_), r * np.sin(t_), z) for r, z in rings for t_ in th]
        ob, ot, ib, it = (lambda k, r=r: r * (m + 1) + k for r in range(4))  # noqa: E731
        faces = []
        for k in range(m):
            k1 = k + 1
            faces += [(ob(k), ob(k1), ot(k1)), (ob(k), ot(k1), ot(k)),            # outer wall
                      (ib(k), it(k1), ib(k1)), (ib(k), it(k), it(k1)),            # inner wall
                      (ot(k), ot(k1), it(k1)), (ot(k), it(k1), it(k)),            # top annulus
                      (ob(k), ib(k1), ob(k1)), (ob(k), ib(k), ib(k1))]            # bottom annulus
        if (th1 - th0) < 2 * np.pi - 1e-6:  # side caps for sector
            for k in (0, m):
                faces += [(ob(k), ib(k), it(k)), (ob(k), it(k), ot(k))]
        self._add(verts, faces)

    def trace(self, name, color, opacity=1.0):
        return go.Mesh3d(
            x=self.x, y=self.y, z=self.z, i=self.i, j=self.j, k=self.k, color=color, opacity=opacity,
            flatshading=True, name=name, hovertext=name, hoverinfo="text", showlegend=False,
            lighting=dict(ambient=0.42, diffuse=0.95, specular=0.65, roughness=0.32, fresnel=0.35),
            lightposition=dict(x=300, y=-400, z=600),
        )


def _line3d(pts, color, width=4, name="", dash=None):
    p = np.asarray(pts, float)
    return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", line=dict(color=color, width=width, dash=dash or "solid"),
                        name=name, hovertext=name, hoverinfo="text", showlegend=False)


def _segments3d(segs, color, width=2, name=""):
    X, Y, Z = [], [], []
    for a, b in segs:
        X += [a[0], b[0], None]; Y += [a[1], b[1], None]; Z += [a[2], b[2], None]
    return go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(color=color, width=width), name=name,
                        hoverinfo="skip", showlegend=False)


# ---- equipment catalogue (educational text) ---------------------------------
EQUIP = {
    "derrick": dict(
        name="Derrick / Mast", color="#475569",
        purpose="Load-bearing steel structure that carries the crown block, traveling block, top drive and the full hook load while drilling, tripping or running casing.",
        components=["Legs and girts", "Diagonal bracing", "Crown block (fixed sheaves)", "Racking board / ladders"],
        variables=["Hook load (klbf)", "Static hook-load rating (klbf)", "Mast height (ft)", "Wind loading"],
        explanation="The block-and-tackle formed by the crown block, traveling block and drilling line gives a mechanical advantage, so the drawworks can lift loads of hundreds of thousands of pounds with a modest line pull. Structural capacity is rated with a safety factor against the maximum expected hook load."),
    "rig_floor": dict(
        name="Rig Floor", color="#64748b",
        purpose="Working platform where the drill crew handles pipe, makes up connections and controls drilling.",
        components=["Floor plates", "Rotary table / drill-floor opening", "Slips and tongs area", "Driller's console"],
        variables=["Floor loading (klbf)", "Floor height above ground (ft)"],
        explanation="The floor sits on the substructure, leaving room underneath for the BOP stack and wellhead. Its opening centres the drill string over the wellbore."),
    "top_drive": dict(
        name="Top Drive & Traveling Block", color="#c2410c",
        purpose="Rotates the drill string from the top and travels up and down the derrick, hanging from the traveling block.",
        components=["Electric/hydraulic motor", "Gearbox", "Swivel and gooseneck", "Pipe handler", "Traveling block and hook"],
        variables=["RPM", "Torque (kft-lbf)", "Hook load (klbf)", "Block position (ft)"],
        explanation="Mud enters through the swivel while the string rotates. Compared with a kelly system, a top drive can drill stands (about 90 ft) and ream/back-ream while pulling out."),
    "drawworks": dict(
        name="Drawworks", color="#b45309",
        purpose="Winch that spools the drilling line to raise and lower the traveling block and control weight on bit.",
        components=["Drum", "Brake system", "Drive motors", "Drilling line and deadline anchor"],
        variables=["Line speed (ft/min)", "Line pull (klbf)", "Brake status"],
        explanation="By paying out the line in a controlled way the driller slacks off the string weight to apply WOB: WOB is the reduction of hook load when the bit is on bottom."),
    "drill_string": dict(
        name="Drill String", color="#334155",
        purpose="Transmits rotation and axial load to the bit and carries drilling fluid to the bottom of the hole.",
        components=["Drill pipe", "Heavy-weight pipe", "Drill collars (BHA)", "Stabilizers and subs"],
        variables=["Outer / inner diameter (in)", "Weight per foot (lb/ft)", "Torque and tension limits"],
        explanation="Pipe is kept mostly in tension; collars near the bit supply the compressive weight. The string's bore is the first leg of the hydraulic circuit (pipe pressure loss)."),
    "bit": dict(
        name="Drill Bit", color="#0e7490",
        purpose="Breaks the rock by crushing/shearing and jets drilling fluid to clean the cutters and lift cuttings.",
        components=["Cutting structure (PDC blades or roller cones)", "Nozzles", "Gauge section", "Bit body"],
        variables=["WOB (klbf)", "RPM", "Total flow area (in^2)", "Bit pressure drop (psi)"],
        explanation="Nozzle velocity is set by flow rate and total flow area. Pressure drop across the bit grows with the square of flow rate and is typically the largest single hydraulic loss."),
    "wellhead": dict(
        name="Wellhead", color="#15803d",
        purpose="Surface termination of the casing strings; provides pressure containment and the base for the BOP stack.",
        components=["Casing head", "Casing hangers", "Flanged spools", "Side-outlet valves"],
        variables=["Working pressure rating (psi)", "Casing sizes (in)"],
        explanation="Each casing string is landed in a hanger inside the head. Side outlets allow annulus access for monitoring or kill operations."),
    "bop": dict(
        name="BOP Stack", color="#b91c1c",
        purpose="Blowout preventers that close the well to control unexpected formation pressure (a kick).",
        components=["Annular preventer", "Pipe rams", "Blind/shear rams", "Choke and kill lines", "Accumulator unit"],
        variables=["Working pressure (psi)", "Bore size (in)", "Accumulator pressure (psi)"],
        explanation="The annular preventer seals around any pipe size; rams seal on pipe or the open hole. Pressure-control equipment is tested and operated under strict procedures that this simulation does not reproduce."),
    "mud_tanks": dict(
        name="Mud Tanks (Pits)", color="#2563eb",
        purpose="Store and condition drilling fluid and provide volume for monitoring gains and losses.",
        components=["Active pits", "Agitators", "Mixing hopper", "Pit-volume sensors"],
        variables=["Mud weight (ppg)", "Viscosity", "Pit volume (bbl)"],
        explanation="Pit volume is watched continuously: an unexplained rise can indicate a kick, a drop can indicate losses. Solids control equipment cleans mud before it returns here."),
    "mud_pumps": dict(
        name="Mud Pumps", color="#d97706",
        purpose="High-pressure reciprocating pumps that circulate drilling fluid down the drill string.",
        components=["Triplex fluid ends", "Power end and motor", "Pulsation dampeners", "Relief valve"],
        variables=["Flow rate (gpm)", "Pump pressure (psi)", "Stroke rate (spm)", "Hydraulic horsepower"],
        explanation="Hydraulic horsepower = pressure x flow / 1714 (psi, gpm). Flow is set by stroke rate and liner size; pressure results from the total circuit resistance."),
    "standpipe": dict(
        name="Standpipe & Rotary Hose", color="#0369a1",
        purpose="Carries high-pressure mud from the pump manifold up the derrick to the top drive swivel.",
        components=["Discharge manifold", "Vertical standpipe", "Flexible rotary hose", "Pressure gauge / transducer"],
        variables=["Standpipe pressure, SPP (psi)", "Flow rate (gpm)"],
        explanation="SPP is the sum of friction losses in surface lines, drill string, bit and annulus, a key real-time indicator for washouts, plugged nozzles or hole problems."),
    "flowline": dict(
        name="Flowline", color="#92400e",
        purpose="Gravity line carrying returning mud from the bell nipple to the shale shaker.",
        components=["Bell nipple", "Flow line", "Flow sensor (paddle)", "Gas detector"],
        variables=["Return flow (gpm)", "Mud temperature", "Gas units"],
        explanation="Comparing return flow with pump flow helps detect influx or losses. Returns exit the well after traveling up the annulus."),
    "shaker": dict(
        name="Shale Shaker", color="#7c3aed",
        purpose="Vibrating screens that remove drilled cuttings from returning mud.",
        components=["Vibrating screens", "Feed box", "Motors", "Cuttings discharge chute"],
        variables=["Screen mesh", "Vibration G-force", "Cuttings volume"],
        explanation="The first stage of solids control. Cuttings samples taken here give the geologist a real-time look at the formation being drilled."),
    "solids": dict(
        name="Solids Control", color="#0891b2",
        purpose="Removes drilled solids from the returning mud so it can be re-used at the correct density and viscosity.",
        components=["Shale shakers", "Degasser", "Desander and desilter hydrocyclones", "Decanting centrifuge", "Mud cleaner"],
        variables=["Cut point (microns)", "Underflow density (ppg)", "Mud weight in/out (ppg)", "Gas in mud (%)"],
        explanation="Equipment is arranged in order of decreasing particle size. The degasser strips entrained gas so mud weight readings stay honest; hydrocyclones and the centrifuge remove fine solids that raise viscosity and ECD."),
    "power": dict(
        name="Power House", color="#ca8a04",
        purpose="Diesel generator sets and drive house supplying electric power to the drawworks, top drive, mud pumps and rig services.",
        components=["Diesel-engine generator sets", "VFD / SCR drive house", "Motor control centre", "Exhaust stacks"],
        variables=["Installed power (kW)", "Load (%)", "Fuel rate (gal/h)"],
        explanation="Modern rigs are diesel-electric: engines drive AC generators and variable-frequency drives feed the large motors. Power demand peaks while tripping and pumping at the same time."),
    "accumulator": dict(
        name="BOP Control Unit", color="#dc2626",
        purpose="Stores hydraulic energy so the BOP stack can be closed quickly even if rig power is lost.",
        components=["Nitrogen-charged accumulator bottles", "Pumps (electric and air)", "Manifold and regulators", "Remote control panels"],
        variables=["System pressure (psi, typically 3,000)", "Usable fluid volume (gal)", "Closing time (s)"],
        explanation="The unit must close the annular and rams within regulatory times and hold enough usable volume for several closures with the pumps off. It is tested regularly; this simulation does not reproduce those procedures."),
    "choke": dict(
        name="Choke Manifold & Gas Buster", color="#9f1239",
        purpose="Routes well returns through adjustable chokes during a kick so the driller can hold back-pressure while killing the well.",
        components=["Adjustable chokes", "Gate valves", "Pressure gauges", "Mud-gas separator (poor-boy degasser)", "Flare line"],
        variables=["Choke position (%)", "Casing pressure (psi)", "Flow out (gpm)"],
        explanation="While circulating a kick the choke is adjusted to keep bottom-hole pressure just above formation pressure. Gas is separated and sent to the flare line."),
    "pipe_rack": dict(
        name="Pipe Rack & Catwalk", color="#a16207",
        purpose="Stores and stages tubulars (drill pipe, casing, collars) and delivers them to the V-door.",
        components=["Pipe racks", "Catwalk", "V-door ramp", "Pipe handling machine"],
        variables=["Joints on location", "Length per joint (ft)", "Setback load (klbf)"],
        explanation="Pipe is tallied and measured on the rack before running. A catwalk and ramp bring each joint to the rig floor safely."),
    "cabins": dict(
        name="Rig Offices & Cabins", color="#64748b",
        purpose="Living and working trailers: company-man office, tool-pusher cabin, mud-logging unit and first-aid.",
        components=["Company-man office", "Mud-logging unit", "Tool-pusher cabin", "Change house"],
        variables=["Mud-log gas (units)", "ROP (ft/h)", "Pit volume (bbl)"],
        explanation="The mud-logging unit records drilling parameters, cuttings lithology and gas shows in real time and alerts the driller to trends."),
    "tanks": dict(
        name="Fuel & Water Tanks", color="#0369a1",
        purpose="Bulk storage for diesel and drill water supporting the rig.",
        components=["Diesel storage tank", "Fresh-water tank", "Transfer pumps"],
        variables=["Fuel level (bbl)", "Water usage (bbl/day)"],
        explanation="Diesel feeds the generator sets, while water is used for mixing mud and cleaning. Secondary containment prevents spills."),
    "support": dict(
        name="Substructure, Catwalk & Pipe Rack", color="#94a3b8",
        purpose="Elevates the rig floor (leaving room for the BOP) and stages tubulars for handling.",
        components=["Substructure posts and beams", "Catwalk", "Pipe rack with tubulars"],
        variables=["Setback load (klbf)", "Substructure height (ft)"],
        explanation="The substructure carries the combined hook load and pipe setback load to the ground. The catwalk lets pipe be brought up to the V-door."),
    "ground": dict(
        name="Ground Surface & Wellbore", color="#a8a29e",
        purpose="Surface grade and the schematic underground wellbore.",
        components=["Location pad", "Cellar", "Borehole (schematic)"],
        variables=["Ground elevation", "Depth (not to scale here)"],
        explanation="The underground portion is shortened for display; see the Wellbore Cutaway page for depth-based geometry."),
}

LABEL_POS = {
    "derrick": (0, 0, 146), "rig_floor": (30, 24, 21), "top_drive": (11, 0, 76), "drawworks": (40, 0, 31),
    "drill_string": (7, 0, 38), "bit": (8, 0, -62), "wellhead": (11, 0, -3), "bop": (15, 0, 12),
    "mud_tanks": (-66, -22, 17), "mud_pumps": (-46, -36, 11), "standpipe": (-24, -14, 70), "flowline": (-30, 12, 23),
    "shaker": (-62, 14, 20), "support": (-5, -52, 5), "ground": (40, 30, 0),
    "solids": (-64, 30, 24), "power": (-98, 6, 15), "accumulator": (28, 40, 14), "choke": (22, -32, 10),
    "pipe_rack": (-8, -50, 6), "cabins": (52, 34, 10), "tanks": (-98, -36, 10),
}


def geo_ground(color, op):
    surf = go.Surface(x=[-110, 60], y=[-70, 60], z=np.zeros((2, 2)), colorscale=[[0, color], [1, color]], showscale=False,
                      opacity=0.55 * op, hoverinfo="skip", showlegend=False, name="Ground")
    bore = _line3d([(0, 0, 0), (0, 0, -64)], "#1f2937", 16, "Borehole (schematic)")
    casing = _line3d([(0, 0, 0), (0, 0, -34)], "#9aa3b2", 10, "Surface casing (schematic)")
    return [surf, bore, casing]


def geo_support(color, op):
    mb = MeshBuilder()
    for sx in (-1, 1):
        for sy in (-1, 1):
            mb.box(sx * 17, sy * 15, 9, 2.2, 2.2, 18)
    for sy in (-1, 1):
        mb.box(0, sy * 15, 17.2, 36, 1.4, 1.2)
    for sx in (-1, 1):
        mb.box(sx * 17, 0, 17.2, 1.4, 30, 1.2)
    for sx in (26, 34):
        for sy in (-1, 1):
            mb.box(sx, sy * 9, 9, 1.8, 1.8, 18)
    mb.box(-8, -40, 0.9, 52, 8, 1.8)  # catwalk / rack
    for yy in (-42.5, -40.5, -38.5, -36.5):
        mb.frustum(-34, yy, 2.5, 52, 0.8, 0.8, axis="x", n=10)
    segs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            segs.append(((sx * 17, sy * 15, 1), (sx * 17, -sy * 15, 17)))
    return [mb.trace(EQUIP["support"]["name"], color, op), _segments3d(segs, color, 2, "Bracing")]


def geo_rig_floor(color, op):
    mb = MeshBuilder()
    mb.box(0, 0, 18.75, 40, 36, 1.5)
    mb.box(30, 0, 18.75, 16, 20, 1.5)
    mb.box(-24, -4, 21.5, 7, 7, 5)  # driller's cabin
    mb.frustum(0, 0, 19.5, 1.6, 3.6, 3.6, n=24)  # rotary table
    mb.box(7, 5, 21.0, 2.2, 2.2, 3.0)  # iron roughneck
    mb.box(-6, 9, 20.8, 3, 3, 2.2)  # catheads / tong hanger
    mb.frustum(-12, 12, 19.5, 2.4, 1.2, 1.2, n=12)  # mousehole
    segs = []
    for y in (-18, 18):  # handrails
        segs.append(((-20, y, 22), (20, y, 22)))
    return [mb.trace("Rig Floor", color, op), _segments3d(segs, color, 3, "Handrail")]


def _lattice(z0=19.5, z1=130.0, b=8.0, t=3.0, levels=9):
    zs = np.linspace(z0, z1, levels)
    w = b + (t - b) * (zs - z0) / (z1 - z0)
    girts, braces = [], []
    for zi, wi in zip(zs, w):
        cs = [(-wi, -wi), (wi, -wi), (wi, wi), (-wi, wi)]
        for a in range(4):
            p, q = cs[a], cs[(a + 1) % 4]
            girts.append(((p[0], p[1], zi), (q[0], q[1], zi)))
    faces = [((-1, -1), (1, -1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))]
    for lv in range(levels - 1):
        wl, wu = w[lv], w[lv + 1]
        for (e1, e2) in faces:
            p1 = (e1[0] * wl, e1[1] * wl, zs[lv]); p2 = (e2[0] * wl, e2[1] * wl, zs[lv])
            q1 = (e1[0] * wu, e1[1] * wu, zs[lv + 1]); q2 = (e2[0] * wu, e2[1] * wu, zs[lv + 1])
            braces += [(p1, q2), (p2, q1)]
    legs = [((sx * b, sy * b, z0), (sx * t, sy * t, z1)) for sx in (-1, 1) for sy in (-1, 1)]
    return legs, girts, braces


def geo_derrick(color, op):
    legs, girts, braces = _lattice()
    mb = MeshBuilder()
    mb.box(0, 0, 133, 9, 9, 6)
    for dx in (-2.2, 0, 2.2):
        mb.frustum(dx, -3.2, 138, 1.2, 1.6, 1.6, axis="y", n=14)
    mb.box(0, 0, 100, 7.0, 0.8, 0.5)  # racking board (monkey board)
    mb.box(0, 0, 20.6, 13, 13, 0.6)  # setback area
    stands = [((3.2 + 0.9 * (k % 2), -4.5 + 1.2 * k, 20.9), (2.6, -4.5 * 0.5 + 0.6 * k - 1.3, 99.5)) for k in range(8)]
    return [_segments3d(legs, color, 7, "Derrick legs"), _segments3d(girts, color, 3, "Girts"),
            _segments3d(braces, color, 2, "Bracing"), mb.trace("Crown block & racking board", color, op),
            _segments3d(stands, "#64748b", 5, "Racked drill-pipe stands")]


def geo_top_drive(color, op):
    mb = MeshBuilder()
    mb.box(0, 0, 70, 5, 4, 7)
    mb.frustum(0, 0, 73.5, 4, 1.5, 1.5, n=16)
    mb.box(0, 0, 80, 6, 3.5, 5)
    mb.frustum(0, 0, 76.5, 3, 0.5, 0.5, n=10)
    cables = [((sx * 2.4, sy * 1.5, 82.5), (sx * 2.2, sy * 2.2, 130)) for sx in (-1, 1) for sy in (-1, 1)]
    return [mb.trace("Top Drive & Traveling Block", color, op), _segments3d(cables, "#334155", 2, "Drilling line")]


def geo_drawworks(color, op):
    mb = MeshBuilder()
    mb.box(30, 0, 22, 12, 16, 4)
    mb.frustum(30, -8.5, 24.5, 17, 3.2, 3.2, axis="y", n=20)
    mb.frustum(30, -8.8, 24.5, 0.6, 4.2, 4.2, axis="y", n=20)
    mb.frustum(30, 8.2, 24.5, 0.6, 4.2, 4.2, axis="y", n=20)
    line = _line3d([(27, 0, 28), (12, 0, 90), (2.5, 0, 130)], "#334155", 2, "Drilling line")
    return [mb.trace("Drawworks", color, op), line]


def geo_drill_string(color, op):
    mb = MeshBuilder()
    mb.frustum(0, 0, -60, 12, 0.9, 0.9, n=12)  # drill collars
    return [_line3d([(0, 0, 66.5), (0, 0, -48)], color, 7, "Drill pipe"), mb.trace("Drill collars (BHA)", "#8f98a8", op)]


def geo_wellhead(color, op):
    mb = MeshBuilder()
    mb.frustum(0, 0, -0.5, 3.0, 2.4, 2.4, n=20)
    mb.frustum(0, 0, 2.0, 0.6, 3.2, 3.2, n=20)
    mb.box(3.2, 0, 1.0, 2.6, 1.2, 1.2)
    mb.box(-3.2, 0, 1.0, 2.6, 1.2, 1.2)
    return [mb.trace("Wellhead", color, op)]


def geo_bop(color, op):
    mb = MeshBuilder()
    mb.frustum(0, 0, 2.8, 2.4, 2.8, 2.8, n=20)  # spool
    mb.box(0, 0, 6.0, 13, 4.2, 3.2)  # pipe rams
    mb.box(0, 0, 9.4, 13, 4.2, 3.2)  # blind/shear rams
    mb.frustum(0, 0, 11.4, 5, 3.4, 3.0, n=22)  # annular
    mb.frustum(0, 0, 16.4, 1.8, 1.9, 1.9, n=16)  # bell nipple
    lines = [_line3d([(6.5, 0, 6.0), (13, 0, 6.0), (13, 0, 1.5)], "#fca5a5", 4, "Choke line"),
             _line3d([(-6.5, 0, 9.4), (-13, 0, 9.4), (-13, 0, 1.5)], "#fca5a5", 4, "Kill line")]
    return [mb.trace("BOP Stack", color, op)] + lines


def geo_mud_tanks(color, op):
    mb = MeshBuilder()
    mb.box(-66, -10, 5.5, 30, 14, 11)
    mb.box(-66, 8, 5.5, 30, 14, 11)
    for xx in (-74, -66, -58):
        mb.frustum(xx, -10, 11, 6, 0.35, 0.35, n=8)
        mb.box(xx, -10, 17.2, 2.2, 1.4, 1.2)
    rails = [((-81, -17, 12.5), (-51, -17, 12.5)), ((-81, 15, 12.5), (-51, 15, 12.5))]
    return [mb.trace("Mud Tanks", color, op), _segments3d(rails, "#64748b", 2, "Rails")]


def geo_mud_pumps(color, op):
    mb = MeshBuilder()
    for xx in (-40, -52):
        mb.box(xx, -28, 3.5, 8, 10, 7)
        mb.frustum(xx, -28, 7, 4, 1.6, 1.6, n=14)
        mb.box(xx, -34, 3.0, 5, 3, 6)
    return [mb.trace("Mud Pumps", color, op)]


def geo_standpipe(color, op):
    pts = [(-40, -22, 6), (-21, -22, 6), (-21, -14, 6), (-21, -14, 66), (-10, -6, 74), (0, 0, 72)]
    return [_line3d(pts, color, 7, "Standpipe & rotary hose")]


def geo_flowline(color, op):
    pts = [(3.5, 0, 17), (-20, 4, 17), (-40, 6, 16), (-55, 8, 15.5)]
    return [_line3d(pts, color, 8, "Flowline")]


def geo_shaker(color, op):
    mb = MeshBuilder()
    mb.box(-62, 8, 13.4, 14, 8, 4)
    mb.box(-62, 8, 15.8, 12, 6, 0.6)
    mb.box(-69.5, 8, 12.4, 2, 4, 2)
    return [mb.trace("Shale Shaker", color, op)]


# ---- detailed bit + extra rig equipment (schematic geometry) ----------------------------------
def _cone_between(self, p0, p1, r0, r1, n=14):
    """Frustum between two arbitrary points (used for roller cones, stabiliser blades, pipe stacks)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ln = float(np.linalg.norm(ax))
    if ln < 1e-9:
        return
    ax /= ln
    ref = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ax, ref)
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    n = max(6, int(round(n * gq())))
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = []
    for p, r in ((p0, r0), (p1, r1)):
        for t_ in th:
            verts.append(p + r * (np.cos(t_) * u + np.sin(t_) * v))
    verts += [p0, p1]
    faces = []
    for a in range(n):
        b = (a + 1) % n
        faces += [(a, b, n + b), (a, n + b, n + a), (2 * n, b, a), (2 * n + 1, n + a, n + b)]
    self._add(verts, faces)


MeshBuilder.cone_between = _cone_between
BIT_TYPES = ["PDC (6 blades)", "Roller cone (tricone)"]


def bit_meshes(kind: str, cx: float, cy: float, zf: float, rad: float, phase: float = 0.0, n_blades: int = 6, lite: bool = False) -> list:
    """
    Schematic drill bit with its cutting face at z = zf and bit radius `rad` (scene units).
    PDC: steel body, helical blades, discrete PDC cutters, nozzles between blades, gauge pads, API pin.
    Roller cone: three tilted cones with carbide inserts, nozzles between cones, bit legs and pin.
    Geometry is representative, not a manufacturer design.
    """
    body, cut, noz, gauge = MeshBuilder(), MeshBuilder(), MeshBuilder(), MeshBuilder()
    R = rad
    if kind.startswith("PDC"):
        body.frustum(cx, cy, zf + 0.10 * R, 0.80 * R, 0.90 * R, 0.80 * R, n=40)  # bit body / crown
        body.frustum(cx, cy, zf + 0.90 * R, 0.55 * R, 0.80 * R, 0.52 * R, n=32)  # shank taper
        body.frustum(cx, cy, zf + 1.45 * R, 0.80 * R, 0.40 * R, 0.40 * R, n=24)  # API pin (threaded, schematic)
        for b in range(n_blades):
            a0 = phase + 2 * np.pi * b / n_blades
            for k, r_f in enumerate(np.linspace(0.10, 0.98, 7)):
                a = a0 + 0.9 * r_f  # swept (spiral) blade
                px, py = cx + R * r_f * np.cos(a), cy + R * r_f * np.sin(a)
                zr = zf + R * (0.10 + 0.20 * (1 - r_f ** 2))  # blade crest follows the bit profile
                body.frustum(px, py, zr - 0.25 * R, 0.30 * R, 0.13 * R, 0.13 * R, n=8)
                if k >= 1:  # PDC cutter at the blade's leading face
                    ca = a + 0.10
                    cut.frustum(cx + R * r_f * np.cos(ca), cy + R * r_f * np.sin(ca), zr - 0.05 * R, 0.20 * R, 0.12 * R, 0.12 * R, n=8)
            ga = a0 + 0.9 * 0.98
            gauge.frustum(cx + R * 0.99 * np.cos(ga), cy + R * 0.99 * np.sin(ga), zf + 0.12 * R, 0.28 * R, 0.085 * R, 0.085 * R, n=8)  # gauge pad
            na = phase + 2 * np.pi * (b + 0.5) / n_blades + 0.5
            noz.frustum(cx + 0.52 * R * np.cos(na), cy + 0.52 * R * np.sin(na), zf + 0.20 * R, 0.22 * R, 0.07 * R, 0.07 * R, n=10)  # nozzle
    else:
        body.frustum(cx, cy, zf + 0.55 * R, 0.55 * R, 0.82 * R, 0.78 * R, n=30)  # bit body
        body.frustum(cx, cy, zf + 1.10 * R, 0.55 * R, 0.78 * R, 0.50 * R, n=28)
        body.frustum(cx, cy, zf + 1.65 * R, 0.80 * R, 0.40 * R, 0.40 * R, n=24)  # pin
        for c in range(3):
            a = phase + 2 * np.pi * c / 3
            ex, ey = np.cos(a), np.sin(a)
            p0 = np.array([cx + 0.10 * R * ex, cy + 0.10 * R * ey, zf + 0.02 * R])
            p1 = np.array([cx + 0.78 * R * ex, cy + 0.78 * R * ey, zf + 0.62 * R])
            cut.cone_between(p0, p1, 0.07 * R, 0.52 * R, n=18)  # cone shell (metal)
            for t in ([] if lite else np.linspace(0.25, 0.95, 5)):  # carbide insert rows (skipped in the lite model used while playing)
                pc = p0 + t * (p1 - p0)
                rr = 0.07 * R + t * (0.52 - 0.07) * R
                for k in range(10):
                    q = 2 * np.pi * k / 10 + 0.3 * t
                    e1 = np.array([-ey, ex, 0.0])
                    e2 = np.cross((p1 - p0) / np.linalg.norm(p1 - p0), e1)
                    pos = pc + rr * (np.cos(q) * e1 + np.sin(q) * e2)
                    gauge.cone_between(pos, pos + 0.10 * R * (np.cos(q) * e1 + np.sin(q) * e2) + 0.0, 0.05 * R, 0.03 * R, n=6)
            na = a + np.pi / 3
            noz.frustum(cx + 0.62 * R * np.cos(na), cy + 0.62 * R * np.sin(na), zf + 0.45 * R, 0.30 * R, 0.07 * R, 0.07 * R, n=10)
    return [body.trace("Bit body", "#3f5a80" if kind.startswith("PDC") else "#475569"), cut.trace("PDC cutters" if kind.startswith("PDC") else "Cone shells", "#f8fafc" if kind.startswith("PDC") else "#64748b"),
            gauge.trace("Gauge / inserts", "#d97706"), noz.trace("Nozzles", "#111827")]


def geo_bit(color, op, phase=0.0):
    out = bit_meshes(BIT_TYPES[0], 0.0, 0.0, -62.4, 1.7, phase)
    for t in out:
        t.opacity = op
    return out


def geo_solids(color, op):
    mb = MeshBuilder()
    mb.box(-64, 30, 6, 16, 6, 12)  # solids-control skid / tank
    for k, xx in enumerate((-70, -66, -62, -58)):  # desander / desilter cones
        mb.cone_between((xx, 30, 18), (xx, 30, 13), 0.1, 1.5, 12)
        mb.frustum(xx, 30, 18, 4.0, 1.5, 1.5, n=12)
    mb.frustum(-52, 30, 12, 7, 2.0, 2.0, n=16)  # degasser vessel
    mb.box(-76, 30, 5, 6, 4, 8)  # centrifuge
    return [mb.trace("Solids Control", color, op)]


def geo_power(color, op):
    mb = MeshBuilder()
    for yy in (-6, 6, 18):
        mb.box(-98, yy, 5, 14, 6, 9)  # generator sets
        mb.frustum(-103, yy, 9.5, 5, 0.6, 0.6, n=8)  # exhaust stack
    mb.box(-86, 2, 4.5, 9, 18, 8)  # VFD / SCR house
    return [mb.trace("Power House", color, op)]


def geo_accum(color, op):
    mb = MeshBuilder()
    mb.box(28, 40, 4.5, 14, 6, 8)  # BOP control unit skid
    for k in range(6):
        mb.frustum(23 + 2 * k, 40, 8.5, 6, 0.8, 0.8, n=10)  # accumulator bottles
    return [mb.trace("BOP Control Unit", color, op), _line3d([(28, 40, 6), (15, 20, 6), (3, 3, 9)], "#fca5a5", 3, "Hydraulic control line")]


def geo_choke(color, op):
    mb = MeshBuilder()
    mb.box(20, -32, 2.5, 10, 6, 5)  # choke manifold skid
    for k in range(3):
        mb.frustum(16 + 4 * k, -32, 5, 3, 0.7, 0.7, n=10)
    mb.frustum(34, -34, 1.0, 8.0, 3.0, 3.0, axis="x", n=16)  # gas buster / poor-boy degasser
    return [mb.trace("Choke Manifold", color, op), _line3d([(13, 0, 1.5), (13, -32, 1.5), (16, -32, 2.5)], "#fca5a5", 4, "Choke line to manifold"),
            _line3d([(38, -34, 2.0), (60, -50, 2.0)], "#94a3b8", 4, "Flare line")]


def geo_pipe_rack(color, op):
    mb = MeshBuilder()
    for yy in (-44, -47, -50):
        mb.box(-8, yy, 0.5, 52, 2.2, 1.0)
    for r in range(3):
        for k in range(8):
            mb.frustum(-34, -44 - 1.6 * (k % 3) - 3 * r + 3, 2.0 + 0.9 * (k // 3), 52, 0.6, 0.6, axis="x", n=8)
    mb.cone_between((18, -42, 1.5), (30, -42, 20), 0.5, 0.5, 6)  # V-door ramp (schematic)
    return [mb.trace("Pipe Rack", color, op)]


def geo_cabins(color, op):
    mb = MeshBuilder()
    for yy in (24, 34, 44):
        mb.box(52, yy, 3, 14, 6, 6)
    return [mb.trace("Cabins", color, op)]


def geo_tanks(color, op):
    mb = MeshBuilder()
    for yy in (-30, -42):
        mb.frustum(-98, yy, 0.5, 12, 4.5, 4.5, axis="x", n=18)  # fuel / water tanks (lying cylinders)
    return [mb.trace("Fuel & Water Tanks", color, op)]


GEOMETRY = {
    "ground": (geo_ground, "#a8a29e"), "support": (geo_support, EQUIP["support"]["color"]),
    "rig_floor": (geo_rig_floor, EQUIP["rig_floor"]["color"]), "derrick": (geo_derrick, EQUIP["derrick"]["color"]),
    "top_drive": (geo_top_drive, EQUIP["top_drive"]["color"]), "drawworks": (geo_drawworks, EQUIP["drawworks"]["color"]),
    "drill_string": (geo_drill_string, EQUIP["drill_string"]["color"]), "bit": (geo_bit, EQUIP["bit"]["color"]),
    "wellhead": (geo_wellhead, EQUIP["wellhead"]["color"]), "bop": (geo_bop, EQUIP["bop"]["color"]),
    "mud_tanks": (geo_mud_tanks, EQUIP["mud_tanks"]["color"]), "mud_pumps": (geo_mud_pumps, EQUIP["mud_pumps"]["color"]),
    "standpipe": (geo_standpipe, EQUIP["standpipe"]["color"]), "flowline": (geo_flowline, EQUIP["flowline"]["color"]),
    "shaker": (geo_shaker, EQUIP["shaker"]["color"]),
    "solids": (geo_solids, EQUIP["solids"]["color"]), "power": (geo_power, EQUIP["power"]["color"]),
    "accumulator": (geo_accum, EQUIP["accumulator"]["color"]), "choke": (geo_choke, EQUIP["choke"]["color"]),
    "pipe_rack": (geo_pipe_rack, EQUIP["pipe_rack"]["color"]), "cabins": (geo_cabins, EQUIP["cabins"]["color"]),
    "tanks": (geo_tanks, EQUIP["tanks"]["color"]),
}


def build_rig_traces(selected=None, visible=None, dim=False, labels=True):
    traces = []
    for eid, (fn, base) in GEOMETRY.items():
        if visible is not None and eid not in visible:
            continue
        col = GHOST if dim else (WARM if eid == selected else base)
        traces.extend(fn(col, 0.35 if dim else 1.0))
    return traces


def apply_scene(fig, camera="Isometric", height=640, uirev=None, aspect=None, hint=None):
    hidden = dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    scene = dict(aspectmode="data" if aspect is None else "manual", bgcolor=SCENE_BG,
                 xaxis=hidden, yaxis=hidden, zaxis=hidden, camera=CAMERAS[camera] if isinstance(camera, str) else camera, dragmode="orbit")
    if aspect is not None:
        scene["aspectratio"] = aspect
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=8, b=26), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK), showlegend=False, uirevision=uirev or f"scene-{camera if isinstance(camera, str) else 'custom'}", scene=scene)
    if hint:
        fig.add_annotation(text=hint, xref="paper", yref="paper", x=0.0, y=0.0, yanchor="top", xanchor="left", showarrow=False, align="left",
                           font=dict(size=11, color=MUTED))
    return fig


def rig_annotations(selected=None, visible=None) -> list:
    """Readable equipment labels: boxed annotations with a light background (clearer than bare 3D text)."""
    ids = [e for e in GEOMETRY if (visible is None or e in visible) and e != "ground"]
    out = []
    for e in ids:
        x, y, z = LABEL_POS[e]
        sel = e == selected
        out.append(dict(x=x, y=y, z=z, text=("<b>" + EQUIP[e]["name"] + "</b>") if sel else EQUIP[e]["name"].split(" / ")[0].split(" &")[0],
                        showarrow=False, font=dict(size=13 if sel else 10, color="#ffffff" if sel else INK),
                        bgcolor=WARM if sel else "rgba(255,255,255,0.82)", bordercolor=WARM if sel else EDGE, borderwidth=1, borderpad=2))
    return out


def rig_figure(selected=None, visible=None, camera="Isometric", labels=True, dim=False, height=640, extra=None):
    fig = go.Figure(build_rig_traces(selected, visible, dim, labels))
    if extra:
        for t in extra:
            fig.add_trace(t)
    apply_scene(fig, camera, height, hint="Schematic geometry - proportions simplified; underground section shortened.")
    if labels and not dim:
        fig.update_layout(scene_annotations=rig_annotations(selected, visible))
    return fig


@st.cache_data(show_spinner=False, max_entries=64)
def cached_rig_figure(selected, visible: tuple, camera: str, labels: bool, height: int, gfx: str = "High"):
    """Rig meshes are pure functions of their inputs: build once, reuse across reruns."""
    return rig_figure(selected, set(visible), camera, labels, height=height)


# ---- three.js renderer for the rig (real shadows, reflections, PBR materials) ---------------------------------
THREE_METAL = {"derrick": (0.75, 0.38), "drill_string": (0.85, 0.3), "bit": (0.8, 0.3), "bop": (0.55, 0.35), "wellhead": (0.6, 0.35),
               "top_drive": (0.45, 0.4), "support": (0.5, 0.5), "rig_floor": (0.4, 0.55), "standpipe": (0.7, 0.3), "flowline": (0.5, 0.45),
               "choke": (0.5, 0.4), "pipe_rack": (0.45, 0.5), "drawworks": (0.4, 0.45)}


@st.cache_data(show_spinner=False)
def rig_scene_payload() -> dict:
    """Rig geometry as plain JSON-able arrays for the WebGL viewer (built once)."""
    items, lo, hi = [], np.full(3, 1e9), np.full(3, -1e9)
    for eid, (fn, base) in GEOMETRY.items():
        if eid == "ground":
            continue
        for t in fn(base, 1.0):
            kind = type(t).__name__
            if kind == "Mesh3d":
                P = np.column_stack([t.x, t.y, t.z]).astype(float)
                lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
                items.append(dict(e=eid, k="m", c=t.color, n=t.name, p=np.round(P, 2).ravel().tolist(),
                                  i=np.column_stack([t.i, t.j, t.k]).astype(int).ravel().tolist()))
            elif kind == "Scatter3d":
                xs, ys, zs = (np.array([np.nan if v is None else v for v in a], float) for a in (t.x, t.y, t.z))
                segs, cur = [], []
                for x_, y_, z_ in zip(xs, ys, zs):
                    if np.isnan(x_) or np.isnan(y_) or np.isnan(z_):
                        if len(cur) > 1:
                            segs.append(cur)
                        cur = []
                    else:
                        cur.append([round(float(x_), 2), round(float(y_), 2), round(float(z_), 2)])
                if len(cur) > 1:
                    segs.append(cur)
                if segs:
                    items.append(dict(e=eid, k="l", c=t.line.color, w=float(t.line.width), n=t.name, s=segs))
    return dict(items=items, lo=[float(v) for v in lo], hi=[float(v) for v in hi])


RIG_THREE_JS = r"""
import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';
import { RoomEnvironment } from 'RoomEnvironment';
import { toCreasedNormals } from 'BufferGeometryUtils';
const D = __DATA__, O = __OPTS__;
const box = document.getElementById('v'), msg = document.getElementById('msg');
let renderer;
try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' }); }
catch (e) { msg.textContent = 'WebGL is not available in this browser - switch the renderer to Classic (Plotly).'; throw e; }
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, O.hq ? 2 : 1.25));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
box.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const cv = document.createElement('canvas'); cv.width = 2; cv.height = 256;
const g = cv.getContext('2d'), gr = g.createLinearGradient(0, 0, 0, 256);
gr.addColorStop(0, '#9fc3e6'); gr.addColorStop(0.55, '#dbe7f1'); gr.addColorStop(1, '#f1ece0'); g.fillStyle = gr; g.fillRect(0, 0, 2, 256);
const sky = new THREE.CanvasTexture(cv); sky.colorSpace = THREE.SRGBColorSpace; scene.background = sky;
const pm = new THREE.PMREMGenerator(renderer);
scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture; scene.environmentIntensity = 0.55;
const lo = D.lo, hi = D.hi, cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2;
const R = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
scene.fog = new THREE.Fog(0xdbe7f1, R * 1.6, R * 7);
scene.add(new THREE.HemisphereLight(0xdfeaff, 0x8a7f6a, 0.55));
const sun = new THREE.DirectionalLight(0xfff1dd, 2.6);
sun.position.set(cx + R * 0.9, cy - R * 0.7, R * 1.5); sun.target.position.set(cx, cy, 0);
sun.castShadow = true; sun.shadow.mapSize.set(O.hq ? 4096 : 2048, O.hq ? 4096 : 2048);
const sc = sun.shadow.camera; sc.left = -R * 0.9; sc.right = R * 0.9; sc.top = R * 0.9; sc.bottom = -R * 0.9; sc.near = 1; sc.far = R * 5;
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.25; sun.shadow.radius = 3;
scene.add(sun, sun.target);
// ground: gravel pad + soft surrounding plane that receives shadows
const gcv = document.createElement('canvas'); gcv.width = gcv.height = 256; const gg = gcv.getContext('2d');
gg.fillStyle = '#a7a193'; gg.fillRect(0, 0, 256, 256);
for (let i = 0; i < 5000; i++) { const v = 120 + Math.random() * 90; gg.fillStyle = `rgba(${v},${v - 6},${v - 18},0.55)`; gg.fillRect(Math.random() * 256, Math.random() * 256, 1.5, 1.5); }
const gtex = new THREE.CanvasTexture(gcv); gtex.wrapS = gtex.wrapT = THREE.RepeatWrapping; gtex.repeat.set(14, 10); gtex.colorSpace = THREE.SRGBColorSpace; gtex.anisotropy = 8;
const pad = new THREE.Mesh(new THREE.PlaneGeometry(170, 130), new THREE.MeshStandardMaterial({ map: gtex, roughness: 0.95, metalness: 0, transparent: true, opacity: 0.93 }));
pad.position.set(-25, -5, 0); pad.receiveShadow = true; scene.add(pad);
const out = new THREE.Mesh(new THREE.PlaneGeometry(R * 40, R * 40), new THREE.MeshStandardMaterial({ color: 0xc9c3b3, roughness: 1 }));
out.position.set(-25, -5, -0.06); out.receiveShadow = true; scene.add(out);
const rot = new THREE.Group(); scene.add(rot);            // parts that spin with the string
const ROT = new Set(['drill_string', 'bit']);
const selE = O.sel, mats = [];
const tmpO = new THREE.Object3D(), up = new THREE.Vector3(0, 0, 1);
for (const it of D.items) {
  const [met, rough] = O.mat[it.e] || [0.3, 0.55];
  const sel = it.e === selE;
  const mat = new THREE.MeshStandardMaterial({ color: it.c, metalness: met, roughness: rough, envMapIntensity: 1.0,
    emissive: sel ? 0xc2410c : 0x000000, emissiveIntensity: sel ? 0.55 : 0 });
  mats.push([it.e, mat]);
  const parent = ROT.has(it.e) ? rot : scene;
  if (it.k === 'm') {
    let geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(it.p, 3)); geo.setIndex(it.i);
    geo = toCreasedNormals(geo.toNonIndexed(), 0.6);
    const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.receiveShadow = true; m.userData.e = it.e; parent.add(m);
  } else {
    const rad = Math.max(0.12, it.w * 0.07), n = it.s.reduce((a, s) => a + s.length - 1, 0);
    const im = new THREE.InstancedMesh(new THREE.CylinderGeometry(rad, rad, 1, 7, 1), mat, n); let q = 0;
    for (const s of it.s) for (let a = 0; a < s.length - 1; a++) {
      const A = new THREE.Vector3(...s[a]), B = new THREE.Vector3(...s[a + 1]), d = B.clone().sub(A), L = d.length();
      if (L < 1e-6) { tmpO.scale.set(0, 0, 0); tmpO.updateMatrix(); im.setMatrixAt(q++, tmpO.matrix); continue; }
      tmpO.position.copy(A).add(B).multiplyScalar(0.5);
      tmpO.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize()); tmpO.scale.set(1, L, 1); tmpO.updateMatrix(); im.setMatrixAt(q++, tmpO.matrix);
    }
    im.castShadow = true; im.receiveShadow = true; parent.add(im);
  }
}
// camera
const camera = new THREE.PerspectiveCamera(38, 1, 1, R * 60); camera.up.set(0, 0, 1);
const target = new THREE.Vector3(cx, cy, cz * 0.6);
const P = { 'Isometric': [1.05, -1.25, 0.62], 'Front': [0, -1.7, 0.35], 'Side': [1.7, 0, 0.35], 'Top': [0.001, -0.01, 2.0], 'Rig floor close-up': [0.45, -0.55, 0.02] };
const e = P[O.cam] || P.Isometric, k = O.cam === 'Rig floor close-up' ? R * 0.55 : R * 0.85;
let tgt = O.cam === 'Rig floor close-up' ? new THREE.Vector3(0, 0, 8) : target;
camera.position.set(tgt.x + e[0] * k * 1.1, tgt.y + e[1] * k * 1.1, tgt.z + e[2] * k * 1.1);
const ctl = new OrbitControls(camera, renderer.domElement);
ctl.target.copy(tgt); ctl.enableDamping = true; ctl.dampingFactor = 0.12; ctl.maxPolarAngle = Math.PI * 0.62; ctl.minDistance = 8; ctl.maxDistance = R * 4; ctl.update();
const KEY = 'psrig_' + O.cam;
try { const s = JSON.parse(sessionStorage.getItem(KEY) || 'null'); if (s) { camera.position.fromArray(s.p); ctl.target.fromArray(s.t); ctl.update(); } } catch (_) {}
ctl.addEventListener('end', () => { try { sessionStorage.setItem(KEY, JSON.stringify({ p: camera.position.toArray(), t: ctl.target.toArray() })); } catch (_) {} });
// labels
const labs = [];
if (O.labels) for (const L of O.lab) { const d = document.createElement('div'); d.className = 'lab' + (L.e === selE ? ' sel' : ''); d.textContent = L.t; box.appendChild(d); labs.push([d, new THREE.Vector3(...L.p)]); }
const v = new THREE.Vector3();
function place() { const w = box.clientWidth, h = box.clientHeight;
  for (const [d, p] of labs) { v.copy(p).project(camera); const vis = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
    d.style.display = vis ? 'block' : 'none'; d.style.transform = `translate(-50%,-100%) translate(${(v.x * 0.5 + 0.5) * w}px,${(-v.y * 0.5 + 0.5) * h}px)`; } }
function resize() { const w = box.clientWidth, h = box.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); draw(); }
let spin = false, need = true, raf = 0;
function draw() { need = true; if (!raf) raf = requestAnimationFrame(frame); }
function frame() { raf = 0; if (spin) rot.rotation.z += 0.02; ctl.update(); renderer.render(scene, camera); place(); if (spin || need) { need = false; if (spin) raf = requestAnimationFrame(frame); } }
ctl.addEventListener('change', draw); window.addEventListener('resize', resize);
document.getElementById('spin').onchange = ev => { spin = ev.target.checked; draw(); };
document.getElementById('shd').onchange = ev => { renderer.shadowMap.enabled = ev.target.checked; sun.castShadow = ev.target.checked; for (const [, m] of mats) m.needsUpdate = true; pad.material.needsUpdate = true; draw(); };
resize(); msg.remove();
"""

RIG_THREE_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#e9eef2;font:12px/1.3 "Segoe UI",Arial,sans-serif;color:#16212e;overflow:hidden}
#v{position:relative;width:100%;height:100%}canvas{display:block;outline:0}
.lab{position:absolute;left:0;top:0;pointer-events:none;background:rgba(255,255,255,.86);border:1px solid #cfc9ba;padding:1px 5px;font-size:10.5px;white-space:nowrap;border-radius:2px}
.lab.sel{background:#c2410c;color:#fff;border-color:#c2410c;font-weight:600;font-size:12px}
#bar{position:absolute;right:8px;top:8px;background:rgba(251,250,246,.92);border:1px solid #d9d4c7;padding:4px 8px;display:flex;gap:12px;z-index:3}
#bar label{cursor:pointer;display:flex;gap:4px;align-items:center}
#msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:13px}
#hint{position:absolute;left:8px;bottom:6px;color:#5b6676;font-size:11px}
</style></head><body><div id="v"><div id="bar"><label><input type="checkbox" id="spin"> Rotate string</label><label><input type="checkbox" id="shd" checked> Shadows</label></div>
<div id="msg">Loading 3D viewer...</div><div id="hint">Drag: orbit &middot; Scroll: zoom &middot; Right-drag: pan &middot; Schematic geometry, proportions simplified</div></div>
<script type="importmap">{"imports":{"three":"/app/static/three/three.module.min.js","OrbitControls":"/app/static/three/OrbitControls.js","RoomEnvironment":"/app/static/three/RoomEnvironment.js","BufferGeometryUtils":"/app/static/three/BufferGeometryUtils.js"}}</script>
<script type="module">__JS__</script></body></html>"""


def rig_three_html(selected, visible, camera: str, labels: bool, hq: bool) -> str:
    import json
    data = rig_scene_payload()
    vis = set(visible)
    data = dict(lo=data["lo"], hi=data["hi"], items=[it for it in data["items"] if it["e"] in vis])
    lab = [dict(e=e, p=list(LABEL_POS[e]), t=EQUIP[e]["name"].split(" / ")[0].split(" &")[0]) for e in GEOMETRY if e in vis and e != "ground"]
    opts = dict(sel=selected, cam=camera, labels=bool(labels), hq=bool(hq), lab=lab, mat={k: list(v) for k, v in THREE_METAL.items()})
    js = RIG_THREE_JS.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__OPTS__", json.dumps(opts))
    return RIG_THREE_HTML.replace("__JS__", js)


def show_three(html: str, height: int = 660):
    import streamlit.components.v1 as components
    components.html(html, height=height)



# ---- mud circulation path -----------------------------------------------------
def z_bottom_schematic(tvd_ft):
    return -(8.0 + 52.0 * min(tvd_ft, 15000.0) / 15000.0)


def circulation_segments(z_bot):
    """(name, points, group, kind). group: 'in' (clean mud) or 'out' (returns). kind picks the velocity."""
    return [
        ("Suction (tank to pump)", [(-58, -6, 5), (-50, -20, 3.5), (-44, -28, 3.5)], "in", "pipe"),
        ("Discharge / standpipe / hose", [(-44, -28, 3.5), (-40, -22, 6), (-21, -22, 6), (-21, -14, 6), (-21, -14, 66),
                                          (-10, -6, 74), (0, 0, 72)], "in", "pipe"),
        ("Drill string", [(0, 0, 72), (0, 0, z_bot)], "in", "pipe"),
        ("Annulus", [(0, 0, z_bot), (1.9, 0, z_bot), (1.9, 0, 16), (3.5, 0, 17)], "out", "annulus"),
        ("Flowline", [(3.5, 0, 17), (-20, 4, 17), (-40, 6, 16), (-55, 8, 15.5)], "out", "flow"),
        ("Shaker to tank", [(-55, 8, 15.5), (-58, 8, 13), (-58, 8, 7), (-58, -6, 5)], "out", "flow"),
    ]


def build_circulation_figure(tvd, v_pipe, v_ann, v_flow, q_gpm, animate=True, n_part=56, n_frames=48, height=600):
    z_bot = z_bottom_schematic(tvd)
    segs = circulation_segments(z_bot)
    vel = {"pipe": v_pipe, "annulus": v_ann, "flow": v_flow}
    xs, ys, zs, tau_pts, seg_end, groups = [], [], [], [], [], []
    tau = 0.0
    for _, pts, grp, kind in segs:
        p = np.asarray(pts, float)
        v = vel[kind]
        for a in range(len(p)):
            if a > 0:
                d = np.linalg.norm(p[a] - p[a - 1])
                tau += d / v if v > 1e-9 else d  # (stationary case: arbitrary monotonic parameter)
            xs.append(p[a, 0]); ys.append(p[a, 1]); zs.append(p[a, 2]); tau_pts.append(tau)
        seg_end.append(tau); groups.append(grp)
    xs, ys, zs, tau_pts = map(np.asarray, (xs, ys, zs, tau_pts))
    seg_end = np.asarray(seg_end)
    total = tau_pts[-1]
    col_of = {"in": COOL, "out": WARM}

    def particles(phase):
        t = (np.arange(n_part) / n_part + phase) % 1.0 * total
        px, py, pz = (np.interp(t, tau_pts, arr) for arr in (xs, ys, zs))
        sid = np.minimum(np.searchsorted(seg_end, t, side="left"), len(groups) - 1)
        return px, py, pz, [col_of[groups[s]] for s in sid]

    traces = build_rig_traces(None, None, dim=True, labels=False)
    for name, pts, grp, _ in segs:
        traces.append(_line3d(pts, col_of[grp], 5, name))
    base_idx = len(traces)
    px, py, pz, pc = particles(0.0)
    traces.append(go.Scatter3d(x=px, y=py, z=pz, mode="markers", marker=dict(size=4.5, color=pc, line=dict(width=0)),
                               hoverinfo="skip", showlegend=False))
    for txt, (x, y, z) in {"Mud tank": (-58, -6, 9), "Mud pump": (-44, -28, 10), "Standpipe": (-21, -14, 70),
                           "Drill string": (15, 0, 52), "Bit": (8, 0, z_bot), "Annulus": (13, -9, 22),
                           "Flowline": (-26, 14, 22), "Shale shaker": (-62, 14, 20)}.items():
        traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="text", text=[txt], textfont=dict(color=INK, size=11),
                                   hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    apply_scene(fig, dict(eye=dict(x=0.8, y=-0.98, z=0.42), center=dict(x=-0.02, y=0, z=0.02), up=dict(x=0, y=0, z=1)), height, uirev="circ",
                hint="Marker speed is schematic: each segment's marker speed is proportional to the computed fluid velocity there, not to scale.")
    if animate and q_gpm > 0:
        frames = []
        for f in range(n_frames):
            fx, fy, fz, fc = particles(f / n_frames)
            frames.append(go.Frame(data=[go.Scatter3d(x=fx, y=fy, z=fz, mode="markers", marker=dict(size=4.5, color=fc))],
                                   traces=[base_idx], name=str(f)))
        fig.frames = frames
        dur = int(np.clip(60000.0 / max(q_gpm, 1.0) / 4.0, 40, 400))
        fig.update_layout(updatemenus=[dict(
            type="buttons", direction="left", x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor=PANEL, font=dict(color=INK),
            buttons=[dict(label="Play", method="animate",
                          args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True, transition=dict(duration=0),
                                           mode="immediate")]),
                     dict(label="Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])])
    return fig


# ---- underground cutaway --------------------------------------------------------
LAYERS = [(name, top, bot, col) for name, top, bot, _s, _a, col in LITHOLOGY]
MAX_DEPTH_FT = 12000.0


def cutaway_figure(depth_ft, shoe_ft, hole_in, pipe_od, cutaway=True, exaggerate=False, show=None, height=700):
    show = show or set()
    ftpu = 120.0 / (2.5 if exaggerate else 1.0)  # ft per scene unit
    RS = 0.5  # schematic radial units per inch (radial dims NOT to vertical scale)
    zu = lambda ft: -ft / ftpu  # noqa: E731
    r_hole_c = 12.25 / 2 * RS
    r_cas_o, r_cas_i = 9.625 / 2 * RS, 8.835 / 2 * RS
    r_open = hole_in / 2 * RS
    r_po = pipe_od / 2 * RS
    r_pi = max((pipe_od - 0.724) / 2 * RS, 0.05)
    shoe_ft = min(shoe_ft, depth_ft)
    th1 = np.pi if cutaway else 2 * np.pi
    W = 14.0
    traces = []

    if "layers" in show:
        y0 = 0.0 if cutaway else -W
        for name, top, bot, col in LAYERS:
            mb = MeshBuilder()
            zc, sz = zu((top + bot) / 2), (bot - top) / ftpu
            mb.box(-(W + r_hole_c) / 2, (y0 + W) / 2, zc, W - r_hole_c, W - y0, sz)
            mb.box((W + r_hole_c) / 2, (y0 + W) / 2, zc, W - r_hole_c, W - y0, sz)
            mb.box(0, (r_hole_c + W) / 2, zc, 2 * r_hole_c, W - r_hole_c, sz)
            if not cutaway:
                mb.box(0, -(r_hole_c + W) / 2, zc, 2 * r_hole_c, W - r_hole_c, sz)
            traces.append(mb.trace(name + " (illustrative)", col, 0.92 if cutaway else 0.22))
        lab = [(W + 0.6, 0, zu((t + b) / 2), n) for n, t, b, _ in LAYERS]
        traces.append(go.Scatter3d(x=[l[0] for l in lab], y=[l[1] for l in lab], z=[l[2] for l in lab], mode="text",
                                   text=[l[3] for l in lab], textposition="middle right",
                                   textfont=dict(color=INK, size=10), hoverinfo="skip", showlegend=False))
        ticks = np.arange(0, MAX_DEPTH_FT + 1, 2000)
        traces.append(go.Scatter3d(x=[-W - 0.4] * len(ticks), y=[0] * len(ticks), z=[zu(t) for t in ticks], mode="text+markers",
                                   marker=dict(size=2, color=MUTED), text=[f"{int(t):,} ft" for t in ticks],
                                   textposition="middle left", textfont=dict(color=MUTED, size=10), hoverinfo="skip",
                                   showlegend=False))

    def tube_mesh(name, color, z0, z1, ri, ro, op=1.0):
        if z1 >= z0 - 1e-9:
            return
        mb = MeshBuilder()
        mb.tube(z1, z0, ri, ro, 0.0, th1)
        traces.append(mb.trace(name, color, op))

    z_top, z_td, z_shoe = 0.0, zu(depth_ft), zu(shoe_ft)
    if "cement" in show:
        tube_mesh("Cement", "#c9b48a", z_top, z_shoe, r_cas_o, r_hole_c)
    if "casing" in show:
        tube_mesh("Casing (9 5/8 in, schematic)", "#9aa3b2", z_top, z_shoe, r_cas_i, r_cas_o)
    tube_mesh("Open-hole wall (rock)", "#6b5b4b", z_shoe, z_td, r_open, r_hole_c)
    tube_mesh("Undrilled formation below TD", "#3d3a36", z_td, zu(MAX_DEPTH_FT), 0.01, r_hole_c)
    if "annulus" in show:
        tube_mesh("Annulus fluid (cased)", WARM, z_top, z_shoe, r_po, r_cas_i, 0.28)
        tube_mesh("Annulus fluid (open hole)", WARM, z_shoe, z_td, r_po, r_open, 0.28)
    if "string" in show:
        mb = MeshBuilder()
        mb.tube(z_td, 4.0, r_pi, r_po, 0.0, th1)
        traces.append(mb.trace("Drill pipe", "#aeb6c4", 1.0))
        mb = MeshBuilder()
        mb.frustum(0, 0, z_td, 0.9, r_open * 0.35, r_open * 0.95, n=24, mod=lambda t: 1 + 0.25 * np.cos(6 * t))
        traces.append(mb.trace("Drill bit", "#2dd4bf", 1.0))
    if "mud" in show:
        tube_mesh("Mud inside drill pipe", COOL, z_top, z_td, 0.01, r_pi, 0.35)
        n_c = max(3, min(9, int(depth_ft / 1500)))
        zc = np.linspace(z_top - 0.8, z_td + 1.5, n_c)
        traces.append(go.Cone(x=[0] * n_c, y=[0] * n_c, z=list(zc), u=[0] * n_c, v=[0] * n_c, w=[-1] * n_c, sizemode="absolute",
                              sizeref=0.14, anchor="tip", colorscale=[[0, COOL], [1, COOL]], showscale=False, hoverinfo="skip"))
        xr = (r_po + r_cas_i) / 2
        traces.append(go.Cone(x=[xr] * n_c + [-xr] * n_c, y=[0] * (2 * n_c), z=list(zc) * 2, u=[0] * (2 * n_c), v=[0] * (2 * n_c),
                              w=[1] * (2 * n_c), sizemode="absolute", sizeref=0.14, anchor="tip",
                              colorscale=[[0, WARM], [1, WARM]], showscale=False, hoverinfo="skip"))
    traces.append(go.Scatter3d(
        x=[W + 0.6, W + 0.6], y=[0, 0], z=[z_shoe, z_td], mode="markers+text", marker=dict(size=4, color=[WARM, RED]),
        text=[f"Casing shoe {shoe_ft:,.0f} ft", f"TD {depth_ft:,.0f} ft"], textposition="middle right",
        textfont=dict(color=INK, size=11), hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    ratio = float(np.clip(0.5 * (MAX_DEPTH_FT / ftpu) / (2 * W), 0.8, 4.2))
    cam = dict(eye=dict(x=0.1, y=-2.35, z=0.3), center=dict(x=0, y=0, z=-0.03), up=dict(x=0, y=0, z=1))
    vert = (f"VERTICAL EXAGGERATION x2.5 ON: 1 unit = {ftpu:.0f} ft" if exaggerate else f"Vertical scale: 1 unit = {ftpu:.0f} ft")
    apply_scene(fig, cam, height, uirev="cutaway", aspect=dict(x=1, y=1, z=ratio),
                hint=f"SCHEMATIC. {vert}. Radial dimensions are exaggerated (not to the same scale as depth). "
                     "Geological layers are illustrative, not actual field geology.")
    return fig


# ---- drilling bit view -------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _bit_template(kind: str, rad: float, lite: bool = False, quality: float = 1.0):
    return tuple(bit_meshes(kind, 0.0, 0.0, 0.0, rad, 0.0, lite=lite))


def bit_view_figure(df, idx, height=520, kind=None, lite=False):
    kind = kind if kind in BIT_TYPES else BIT_TYPES[0]
    d0 = float(df["depth_ft"].iloc[0])
    foot_total = max(float(df["depth_ft"].iloc[-1]) - d0, 0.5)
    row = df.iloc[idx]
    foot = float(row["depth_ft"]) - d0
    z_bit = -6.0 - 38.0 * foot / foot_total
    wear = float(row["wear"]) if "wear" in row else 0.0
    traces = []
    strata = ((0, -16, "#8c6d46"), (-16, -32, "#6b7280"), (-32, -50, "#d9b36c"))
    for z0, z1, col in strata:
        mb = MeshBuilder()
        mb.box(0, 0, (z0 + z1) / 2, 18, 18, abs(z1 - z0))
        traces.append(mb.trace("Formation (illustrative)", col, 0.16))
    ang = math.radians(float(row["bit_angle_deg"]) % 360.0)
    rad = 2.4
    for t0 in _bit_template(kind, rad, lite, gq()):  # cached geometry, rotated and shifted per frame (cheap)
        c, sn = math.cos(ang), math.sin(ang)
        x0, y0, z0 = np.asarray(t0.x), np.asarray(t0.y), np.asarray(t0.z)
        traces.append(go.Mesh3d(x=c * x0 - sn * y0, y=sn * x0 + c * y0, z=z0 + z_bit, i=t0.i, j=t0.j, k=t0.k, color=t0.color, opacity=1.0,
                                flatshading=True, name=t0.name, hovertext=t0.name, hoverinfo="text", showlegend=False, lighting=t0.lighting,
                                lightposition=t0.lightposition))
    mb = MeshBuilder()  # BHA above the bit: pin sub, near-bit stabiliser with blades, collar
    mb.frustum(0, 0, z_bit + 3.3 * rad, 1.1 * rad, 0.55 * rad, 0.55 * rad, n=20)
    mb.frustum(0, 0, z_bit + 4.4 * rad, 2.2 * rad, 0.70 * rad, 0.70 * rad, n=20)
    for b in range(4):
        a = ang * 0.0 + b * np.pi / 2
        mb.cone_between((0.7 * rad * np.cos(a), 0.7 * rad * np.sin(a), z_bit + 4.6 * rad), (0.95 * rad * np.cos(a), 0.95 * rad * np.sin(a), z_bit + 5.9 * rad), 0.12 * rad, 0.14 * rad, 8)
    mb.frustum(0, 0, z_bit + 6.6 * rad, 8.0, 0.70 * rad, 0.70 * rad, n=18)
    traces.append(mb.trace("BHA: stabiliser and collar", "#94a3b8"))
    traces.append(_line3d([(0, 0, 14), (0, 0, z_bit + 14.6 * rad * 0.5)], "#aeb6c4", 8, "Drill string"))
    traces.append(_line3d([(0, 0, 0), (0, 0, z_bit)], "#1f2937", 26, "Drilled hole"))
    rng = np.random.default_rng(int(idx))
    pts = rng.normal(0, 1.9, (18, 2))
    traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=z_bit + 2.0 + rng.uniform(0, 9, 18), mode="markers",
                               marker=dict(size=3, color=WARM), hoverinfo="skip", showlegend=False))
    for k in range(6):  # nozzle jet lines
        a = ang + 2 * np.pi * (k + 0.5) / 6 + 0.5
        traces.append(_line3d([(0.55 * rad * np.cos(a), 0.55 * rad * np.sin(a), z_bit + 0.3 * rad),
                               (0.9 * rad * np.cos(a), 0.9 * rad * np.sin(a), z_bit - 1.6)], COOL, 3, "Mud jet"))
    traces.append(go.Scatter3d(x=[6.2], y=[0], z=[z_bit + 2], mode="text", text=[f"{row['depth_ft']:,.1f} ft  |  wear {wear * 100:.0f} %"],
                               textfont=dict(color=INK, size=12), hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    zc = (z_bit + 4.0 + 18.0) / 64.0
    cam = dict(eye=dict(x=0.32, y=-0.38, z=zc + 0.10), center=dict(x=0, y=0, z=zc), up=dict(x=0, y=0, z=1))
    apply_scene(fig, cam, height, uirev="bit", hint=f"{kind}: representative geometry, not a manufacturer design. Rotation = integrated bit RPM; mud jets leave the nozzles.")
    return fig


# ---- wireline scene -----------------------------------------------------------------
def wireline_figure(depth_ft, target_ft, height=640):
    target = max(target_ft, 1.0)
    z_tool = 8.0 - depth_ft * (68.0 / target)
    traces = []
    traces.append(go.Surface(x=[-45, 20], y=[-20, 20], z=np.zeros((2, 2)), colorscale=[[0, "#b9b3a0"], [1, "#b9b3a0"]],
                             showscale=False, opacity=0.55, hoverinfo="skip"))
    mb = MeshBuilder()
    mb.box(-24, 0, 2.2, 22, 7, 1.6)  # chassis
    mb.box(-33.5, 0, 5.0, 5, 6, 5)  # cab
    mb.box(-20, 0, 5.2, 14, 6.5, 6)  # body
    for xx in (-31, -24, -18):
        for yy in (-3.6, 3.6):
            mb.frustum(xx, yy - 0.5, 1.2, 1.0, 1.2, 1.2, axis="y", n=14)
    traces.append(mb.trace("Wireline unit", "#c27c0e"))
    mb = MeshBuilder()
    mb.frustum(-18, -2.2, 9.5, 4.4, 3.0, 3.0, axis="y", n=26)
    mb.frustum(-18, -2.5, 9.5, 0.5, 3.9, 3.9, axis="y", n=26)
    mb.frustum(-18, 2.1, 9.5, 0.5, 3.9, 3.9, axis="y", n=26)
    traces.append(mb.trace("Cable reel", "#9aa7bd"))
    ang = math.radians(depth_ft / 10.0 * 360.0)  # ~10 ft of cable per drum revolution (illustrative)
    spokes = [((-18, 2.7, 9.5), (-18 + 3.6 * math.cos(ang + k * 2.094), 2.7, 9.5 + 3.6 * math.sin(ang + k * 2.094))) for k in range(3)]
    traces.append(_segments3d(spokes, "#111827", 4, "Reel spokes"))
    mb = MeshBuilder()
    mb.frustum(0, 0, 2.2, 12, 0.85, 0.85, n=18)  # lubricator
    mb.frustum(0, 0, 14.2, 1.2, 1.1, 1.1, n=18)  # stuffing box
    mb.frustum(0, 1.2, 16.4, 0.5, 1.4, 1.4, axis="y", n=18)  # sheave
    mb.box(0, 0, 3.0, 4.2, 1.8, 1.4)  # wireline valve
    traces.append(mb.trace("Lubricator", "#94a3b8"))
    mb = MeshBuilder()
    mb.frustum(0, 0, -0.5, 3.0, 2.0, 2.0, n=18)
    mb.frustum(0, 0, 2.0, 0.5, 2.8, 2.8, n=18)
    traces.append(mb.trace("Wellhead", GREEN))
    traces.append(_line3d([(0, 0, 0), (0, 0, -70)], "#1f2937", 14, "Well (schematic)"))
    traces.append(_line3d([(-18, 0, 12.8), (-6, 0, 16.0), (0, 0, 17.8), (0, 0, z_tool + 3.2)], "#111827", 3, "Wireline cable"))
    mb = MeshBuilder()
    mb.frustum(0, 0, z_tool, 3.2, 0.32, 0.32, n=12)
    traces.append(mb.trace("Downhole tool string", WARM))
    mb = MeshBuilder()
    mb.frustum(0, 0, z_tool + 3.2, 0.5, 0.4, 0.4, n=12)
    traces.append(mb.trace("Rope socket", "#6b7280"))
    tk = np.linspace(0, target, 5)
    traces.append(go.Scatter3d(x=[3.5] * 5, y=[0] * 5, z=[8.0 - t * 68.0 / target for t in tk], mode="text+markers",
                               marker=dict(size=2, color=MUTED), text=[f"{t:,.0f} ft" for t in tk], textposition="middle right",
                               textfont=dict(color=MUTED, size=10), hoverinfo="skip", showlegend=False))
    for txt, (x, y, z) in {"Wireline unit": (-24, 0, 11), "Cable reel": (-18, 0, 15), "Lubricator": (2.5, 0, 12),
                           "Wellhead": (3.5, 0, -1.5), "Downhole tool": (2.5, 0, z_tool)}.items():
        traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="text", text=[txt], textfont=dict(color=INK, size=11),
                                   hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    cam = dict(eye=dict(x=0.75, y=-1.75, z=0.3), center=dict(x=-0.08, y=0, z=0.05), up=dict(x=0, y=0, z=1))
    return apply_scene(fig, cam, height, uirev="wl", hint="Schematic. Depth axis scaled so the target depth maps to the bottom of the scene.")


# ---- coiled tubing scene -------------------------------------------------------------
def coiled_tubing_figure(depth_ft, target_ft, height=640):
    target = max(target_ft, 1.0)
    z_tip = 19.0 - depth_ft * (80.0 / target)
    traces = []
    traces.append(go.Surface(x=[-60, 25], y=[-25, 25], z=np.zeros((2, 2)), colorscale=[[0, "#b9b3a0"], [1, "#b9b3a0"]],
                             showscale=False, opacity=0.55, hoverinfo="skip"))
    mb = MeshBuilder()
    mb.box(-36, 0, 2.0, 34, 11, 2.0)  # trailer
    mb.box(-56, 0, 4.0, 7, 8, 6)  # tractor cab
    for xx in (-44, -36, -28):
        mb.frustum(xx, -6.2, 1.2, 1.2, 1.3, 1.3, axis="y", n=14)
    traces.append(mb.trace("Coiled tubing unit (trailer)", "#c2410c"))
    mb = MeshBuilder()
    mb.frustum(-34, -4.0, 11.5, 8.0, 5.0, 5.0, axis="y", n=36)
    mb.frustum(-34, -4.6, 11.5, 0.8, 9.0, 9.0, axis="y", n=36)
    mb.frustum(-34, 3.8, 11.5, 0.8, 9.0, 9.0, axis="y", n=36)
    traces.append(mb.trace("Tubing reel", "#9aa7bd"))
    ang = math.radians(depth_ft / 30.0 * 360.0)
    sp = [((-34, 4.7, 11.5), (-34 + 8.4 * math.cos(ang + k * 1.0472), 4.7, 11.5 + 8.4 * math.sin(ang + k * 1.0472))) for k in range(6)]
    traces.append(_segments3d(sp, "#111827", 3, "Reel spokes"))
    # tubing: reel top -> gooseneck (cubic Bezier ending vertical) -> injector -> well
    P = np.array([(-14, 0, 20), (-6, 0, 37), (0, 0, 37), (0, 0, 28.5)], float)
    s = np.linspace(0, 1, 24)[:, None]
    bez = (1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1] + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3]
    traces.append(_line3d([(-34, 0, 16.5), (-14, 0, 20)], "#0f766e", 5, "Coiled tubing"))
    traces.append(_line3d(bez.tolist(), "#0f766e", 5, "Coiled tubing (gooseneck)"))
    traces.append(_line3d([(0, 0, 28.5), (0, 0, z_tip)], "#0f766e", 5, "Coiled tubing (in well)"))
    mb = MeshBuilder()
    mb = MeshBuilder()
    traces.append(_line3d([(-14, 0, 19.2), (-6, 0, 36), (0, 0, 36.3), (0.9, 0, 28)], "#64748b", 3, "Gooseneck guide arch"))
    mb.box(0, 0, 24.5, 5.5, 5.5, 7)  # injector head
    mb.box(2.6, 0, 24.5, 1.2, 5.8, 7.4)
    mb.box(-2.6, 0, 24.5, 1.2, 5.8, 7.4)
    traces.append(mb.trace("Injector head", "#e4572e"))
    mb = MeshBuilder()
    mb.frustum(0, 0, 18.5, 2.6, 1.3, 1.3, n=18)  # stripper
    for zz in (13.5, 16.0):
        mb.box(0, 0, zz, 9, 3.4, 2.2)  # BOP rams
    mb.frustum(0, 0, 3.0, 8.0, 1.8, 1.8, n=18)  # riser
    traces.append(mb.trace("Stripper / BOP", "#dc2626"))
    mb = MeshBuilder()
    mb.frustum(0, 0, -0.5, 3.0, 2.2, 2.2, n=18)
    mb.frustum(0, 0, 2.0, 0.5, 3.0, 3.0, n=18)
    traces.append(mb.trace("Wellhead", GREEN))
    segs = [((sx * 6, sy * 5, 0), (sx * 3, sy * 2.6, 22)) for sx in (-1, 1) for sy in (-1, 1)]
    traces.append(_segments3d(segs, "#7b8794", 5, "Support frame"))
    traces.append(_line3d([(0, 0, 0), (0, 0, -64)], "#1f2937", 14, "Well (schematic)"))
    mb = MeshBuilder()
    mb.frustum(0, 0, z_tip - 0.8, 0.9, 0.3, 0.3, n=12)
    traces.append(mb.trace("BHA / nozzle", WARM))
    tk = np.linspace(0, target, 5)
    traces.append(go.Scatter3d(x=[3.5] * 5, y=[0] * 5, z=[19.0 - t * 80.0 / target for t in tk], mode="text+markers",
                               marker=dict(size=2, color=MUTED), text=[f"{t:,.0f} ft" for t in tk], textposition="middle right",
                               textfont=dict(color=MUTED, size=10), hoverinfo="skip", showlegend=False))
    for txt, (x, y, z) in {"Tubing reel": (-34, 0, 22), "Gooseneck": (-7, 0, 39), "Injector head": (5, 0, 29),
                           "Stripper / BOP": (7, 0, 15), "Wellhead": (5, 0, -1.5), "BHA": (2.5, 0, z_tip)}.items():
        traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="text", text=[txt], textfont=dict(color=INK, size=11),
                                   hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    cam = dict(eye=dict(x=0.65, y=-1.8, z=0.32), center=dict(x=-0.08, y=0, z=0.05), up=dict(x=0, y=0, z=1))
    return apply_scene(fig, cam, height, uirev="ct", hint="Schematic. Depth axis scaled so the target depth maps to the bottom of the scene.")


# =============================================================================
# 6. UI HELPERS & SESSION STATE
# =============================================================================
HOWTO = {
    "Operations Center": "**Top row:** live status of the Drilling Simulator run and the ML anomaly detector. **3D view:** the whole location; use *Graphics* (top right) to trade detail for speed. Every number here comes from the Drilling Simulator page; press Start there to see it move.",
    "3D Rig Explorer": "Drag = orbit, scroll = zoom, right-drag = pan. Pick an item in *Select equipment* to highlight it (orange label) and read its purpose, components and variables. *Visibility* hides items so you can look inside the layout.",
    "Wellbore Cutaway": "Left controls set depth, casing shoe and hole size. The cutaway removes a wedge of the earth so you can see the formation colours, casing (grey), cement, drill string and mud flow (blue down the pipe, orange up the annulus). Vertical scale can be exaggerated.",
    "Mud Circulation Lab": "Blue markers are clean mud going down the drill string; orange markers are returns coming up the annulus. Marker speed is proportional to the computed velocity. The metrics give hydrostatic pressure, friction losses, standpipe pressure (SPP) and ECD.",
    "Drilling Simulator": "Press **Start**: the bit drills at the ROP the model computes from WOB, RPM, flow, mud weight, formation strength, bit wear and hole cleaning. Charts: depth, ROP, torque and SPP (dashed = no-disturbance baseline). Lower charts (when paused) show the pressure window - ECD must stay between pore pressure (blue) and fracture gradient (orange) - plus MSE and bit wear.",
    "Well Path & Torque-Drag": "The 3D line is the trajectory coloured by string tension. Charts show tension, torque, inclination and dogleg severity along measured depth. Pick-up > slack-off > rotating tells you how much drag the well adds; a red banner means buckling is predicted.",
    "Well Control": "Enter the shut-in readings (SIDPP, SICP, pit gain). The sheet gives the kill mud weight, initial/final circulating pressure, MAASP and what kind of fluid came in (gas / oil / water). Red banners mean the kill is unsafe as entered.",
    "Wireline Operations": "Press Start to run the tool in hole (RIH) or pull out (POOH); cable tension is indicative. Use the speed and depth controls to see how tension changes.",
    "Coiled Tubing Lab": "Press Start to run the tubing in; the tension/weight indicator shows the lock-up trend. The 3D view shows reel, gooseneck, injector, BOP and the tubing in the well.",
    "Reservoir Simulation": "Map colours: **oil saturation** (dark = oil-rich, pale = swept / water), **pressure**, **permeability**. The oil-water contact (OWC) sits low in the structure; wells are marked. Use the snapshot slider to move through time. Choose *Primary depletion* to see pressure fall and oil rate decline, or *Waterflood* to see injected water sweep the oil.",
    "SRP Lab": "Left: the surface unit (motor, gearbox, counterweights, walking beam, horsehead, polished rod). Right: the downhole pump (plunger, travelling and standing valve). Press Play: the motion is computed in your browser, so it stays smooth. The dynamometer card (load vs position) is the engineer's view of what the pump is doing; a thin rounded card means a healthy full pump.",
    "ESP Lab": "The blue curve is what the pump can deliver (head vs rate); the orange curve is what the well needs. Where they cross is the operating point. The green band is the recommended operating range. Changing frequency moves the pump curve (affinity laws).",
    "AI Diagnostics": "An Isolation Forest is trained only on normal synthetic runs. The tables compare it with a simple static-threshold baseline on separate test runs. Scores are rankings, not failure probabilities.",
    "Scenario Comparison": "Define two runs (A and B). Charts overlay them; the table lists the parameters that differ. CSV buttons export the data.",
    "Engineering Reference": "Equations, units, assumptions and limits for every module, plus live self-checks (press the button to run them).",
}


def show(fig, key=None):
    """Render a Plotly figure full-width (handles Streamlit API differences)."""
    cfg = dict(displaylogo=False, scrollZoom=True, modeBarButtonsToRemove=["lasso2d", "select2d", "toImage"])
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    except Exception:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


def style_fig(fig, height=360, title=None):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
                      margin=dict(l=55, r=20, t=44 if title else 20, b=40),
                      title=dict(text=title, font=dict(size=14)) if title else None,
                      font=dict(color=INK, size=12), legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(gridcolor="#ece8dc", zerolinecolor="#d9d4c7", linecolor=EDGE)
    fig.update_yaxes(gridcolor="#ece8dc", zerolinecolor="#d9d4c7", linecolor=EDGE)
    return fig


def header(title, sub):
    st.markdown(f"# {title}")
    st.markdown(f"<div class='ps-sub'>{sub}</div><hr class='ps-rule'>", unsafe_allow_html=True)
    if title in HOWTO:
        with st.expander("How to read this page", expanded=False):
            st.markdown(HOWTO[title])


def badge(text, kind="ok"):
    return f"<span class='ps-badge ps-{kind}'>{text}</span>"


def slider(label, key, lo, hi, step, default, help=None, fmt=None):
    """Keyed slider with typed defaults (all-float or all-int) and persistent state."""
    if isinstance(default, float):
        lo, hi, step = float(lo), float(hi), float(step)
        if fmt is None:  # show only the decimals the step needs
            d = 0
            while d < 4 and abs(step * 10 ** d - round(step * 10 ** d)) > 1e-9:
                d += 1
            fmt = f"%.{d}f"
    else:
        lo, hi, step = int(lo), int(hi), int(step)
    if key not in st.session_state:
        st.session_state[key] = default
    return st.slider(label, lo, hi, step=step, key=key, help=help, format=fmt)


def toggle(label, key, default=True, help=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.toggle(label, key=key, help=help)


def keep_widget_state():
    """Keep keyed widget values alive when their page is not rendered (Streamlit drops unrendered widget keys)."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("w_"):
            st.session_state[k] = st.session_state[k]


def reset_keys(defaults: dict, extra: dict | None = None):
    for k, v in defaults.items():
        st.session_state[k] = v
    for k, v in (extra or {}).items():
        st.session_state[k] = v


def slider_grid(specs, ncols=2):
    """Compact grid of sliders. specs: (label, key, lo, hi, step, default[, kwargs]). Returns {key: value}."""
    cols = st.columns(ncols)
    out = {}
    for i, spec in enumerate(specs):
        kw = spec[6] if len(spec) > 6 else {}
        with cols[i % ncols]:
            out[spec[1]] = slider(*spec[:6], **kw)
    return out


def run_live(fn, running: bool, interval: float = 0.5):
    """Run a live-updating fragment (auto re-run only while a simulation is running)."""
    if hasattr(st, "fragment"):
        st.fragment(fn, run_every=interval if running else None)()
    else:  # very old Streamlit: static render
        fn()


def init_state():
    ss = st.session_state
    ss.setdefault("sim_running", False)
    ss.setdefault("sim_idx", 0)
    ss.setdefault("wl", dict(depth=0.0, elapsed=0.0, running=False, dir="RIH", last=0.0))
    ss.setdefault("ct", dict(depth=0.0, elapsed=0.0, running=False, dir="RIH", last=0.0))


# ---- drilling-simulator parameter defaults (widget keys) ------------------------------
DRILL_KEYS = dict(
    w_dr_wob=("wob", 22.0), w_dr_rpm=("rpm", 110.0), w_dr_flow=("flow", 520.0), w_dr_mw=("mw", 10.0), w_dr_fr=("fr", 1.0),
    w_dr_fric=("fric", 0.3), w_dr_ss=("ss", 0.3), w_dr_dur=("duration_min", 30.0), w_dr_depth=("start_depth", 8000.0),
    w_dr_hole=("hole_in", 8.5), w_dr_pod=("pipe_od", 5.0), w_dr_pid=("pipe_id", 4.276), w_dr_tfa=("tfa", 0.746),
    w_dr_mu=("mu_cp", 15.0), w_dr_kres=("k_res", 1.0),
    w_dr_rheo=("rheo_model", "Newtonian"), w_dr_t600=("t600", 60.0), w_dr_t300=("t300", 38.0),
    w_dr_lith=("lith_on", 1.0), w_dr_wear0=("bit_wear0", 0.0), w_dr_dcut=("d_cut", 0.25),
    w_dr_op=("overpress", 0.0), w_dr_optop=("op_top", 9500.0),
)
DRILL_DEFAULTS = {k: v[1] for k, v in DRILL_KEYS.items()}


def get_drill_params(disturbed=True) -> dict:
    p = dict(SIM_DEFAULTS)
    for key, (name, default) in DRILL_KEYS.items():
        v = st.session_state.get(key, default)
        p[name] = str(v) if isinstance(default, str) else float(v)
    if not disturbed:
        p["fric"], p["ss"] = 0.0, 0.0
    return p


def get_current_sim():
    """(params, dataframe, playhead index) for the Drilling Simulator state; None on invalid inputs."""
    p = get_drill_params(disturbed=st.session_state.get("w_dr_case", "With disturbances") == "With disturbances")
    try:
        df = cached_sim(freeze(p))
    except ValueError:
        return p, None, 0
    idx = int(min(st.session_state.get("sim_idx", 0), len(df) - 1))
    return p, df, idx


# =============================================================================
# 7. PAGES
# =============================================================================
def anomaly_status_for(p, df, idx):
    """Return (label, kind, flagged_recent, total_flagged) from the trained Isolation Forest on the simulator run."""
    det = cached_detector()
    feats = build_features(df, p)
    pred = det.predict(feats)
    pred[:WARMUP_SAMPLES] = 0  # rolling features are not yet defined during warm-up
    if idx < WARMUP_SAMPLES:
        return "Warming up (needs 30 s of simulated data)", "warn", 0, 0
    lo = max(0, idx - 15)
    recent = int(pred[lo: idx + 1].sum())
    if recent >= 5:
        return "Anomalous pattern flagged", "bad", recent, int(pred[: idx + 1].sum())
    if recent >= 1:
        return "Watch (isolated flags)", "warn", recent, int(pred[: idx + 1].sum())
    return "No flags", "ok", recent, int(pred[: idx + 1].sum())


def page_operations():
    header("Operations Center", "Summary of the current simulation state - every value is derived from the Drilling Simulator run.")
    p, df, idx = get_current_sim()
    ss = st.session_state
    left, right = st.columns([3, 2], gap="medium")
    with left:
        cam = st.radio("Camera", PRESET_NAMES, horizontal=True, key="w_ops_cam", label_visibility="collapsed")
        show(cached_rig_figure(None, tuple(GEOMETRY), cam, True, 600, gfx_name()), key="ops_rig")
    with right:
        state = "RUNNING" if ss.get("sim_running") else ("PAUSED" if idx > 0 else "IDLE")
        kind = {"RUNNING": "ok", "PAUSED": "warn", "IDLE": "warn"}[state]
        st.markdown(badge("SIMULATED DATA - NOT LIVE", "warn") + badge(f"Simulator: {state}", kind), unsafe_allow_html=True)
        if df is None:
            st.error("Drilling Simulator inputs are invalid - fix them on the Drilling Simulator page.")
        else:
            r = df.iloc[idx]
            stat, k2, recent, tot = anomaly_status_for(p, df, idx)
            st.markdown(badge(f"Anomaly detector: {stat}", k2), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Simulated depth", f"{r.depth_ft:,.1f} ft")
            c2.metric("ROP", f"{r.rop_ftph:,.1f} ft/h")
            c1.metric("WOB", f"{r.wob_klbf:,.1f} klbf")
            c2.metric("RPM (surface)", f"{r.rpm:,.0f}")
            c1.metric("Flow rate", f"{r.flow_gpm:,.0f} gpm")
            c2.metric("Hydrostatic pressure", f"{r.hyd_psi:,.0f} psi")
            c1.metric("Torque indicator", f"{r.torque_kftlbf:,.2f} kft-lbf")
            c2.metric("Standpipe pressure", f"{r.spp_psi:,.0f} psi")
            st.caption(f"Playhead: {r.t_s / 60:.1f} min of {df.t_s.iloc[-1] / 60:.0f} min  |  case: "
                       f"{ss.get('w_dr_case', 'With disturbances')}  |  ECD {r.ecd_ppg:.2f} ppg  |  flags in last 30 s: {recent}")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=df.t_s / 60, y=df.torque_kftlbf, line=dict(color=WARM, width=1), name="Torque"), 1, 1)
            fig.add_trace(go.Scatter(x=df.t_s / 60, y=df.spp_psi, line=dict(color=COOL, width=1), name="SPP"), 2, 1)
            fig.add_vline(x=r.t_s / 60, line_color=INK, line_dash="dot")
            fig.update_yaxes(title_text="kft-lbf", row=1, col=1)
            fig.update_yaxes(title_text="psi", row=2, col=1)
            fig.update_xaxes(title_text="min", row=2, col=1)
            show(style_fig(fig, 300), key="ops_trend")
        st.page_link(PAGE_OBJS["Drilling Simulator"], label="Open Drilling Simulator", icon=":material/arrow_forward:")
        st.button("Reset drilling playhead", on_click=lambda: (ss.update(sim_idx=0, sim_running=False)), key="ops_reset")


def page_rig_explorer():
    header("3D Rig Explorer", "Procedural drilling-site geometry. Drag to orbit, scroll to zoom, right-drag to pan.")
    ids = list(GEOMETRY)
    ctrl, view = st.columns([1, 3], gap="medium")
    with ctrl:
        names = {e: EQUIP[e]["name"] for e in ids}
        if "w_rig_sel" not in st.session_state:
            st.session_state["w_rig_sel"] = "(none)"
        sel = st.selectbox("Select equipment", ["(none)"] + ids, format_func=lambda e: names.get(e, e), key="w_rig_sel")
        cam = st.selectbox("Camera preset", PRESET_NAMES, key="w_rig_cam")
        labels = toggle("Show labels", "w_rig_labels", True)
        with st.expander("Visibility", expanded=False):
            vis = st.multiselect("Visible equipment", ids, default=ids if "w_rig_vis" not in st.session_state else None,
                                 format_func=lambda e: names[e], key="w_rig_vis")
        st.button("Reset view", on_click=reset_keys, args=({"w_rig_sel": "(none)", "w_rig_cam": "Isometric",
                                                            "w_rig_labels": True, "w_rig_vis": ids},), key="rig_reset")
    with view:
        selected = None if sel == "(none)" else sel
        rend = st.radio("Renderer", ["Realistic (WebGL, shadows)", "Classic (Plotly)"], horizontal=True, key="w_rig_rend", label_visibility="collapsed")
        if rend.startswith("Realistic"):
            show_three(rig_three_html(selected, tuple(sorted(vis)), cam, labels, gfx_name() == "Ultra"), 660)
        else:
            show(cached_rig_figure(selected, tuple(sorted(vis)), cam, labels, 640, gfx_name()), key="rig_fig")
    if selected:
        e = EQUIP[selected]
        st.markdown(f"### {e['name']}")
        a, b = st.columns(2)
        a.markdown(f"**Purpose**  \n{e['purpose']}")
        a.markdown("**Main components**  \n" + "  \n".join(f"- {c}" for c in e["components"]))
        b.markdown("**Operating variables**  \n" + "  \n".join(f"- {c}" for c in e["variables"]))
        b.markdown(f"**Engineering explanation**  \n{e['explanation']}")
    else:
        st.info("Pick an item in **Select equipment** to highlight it and read its purpose, components and operating variables. "
                "Direct click-picking is not available in Plotly 3D, so a selector is used.")


def page_cutaway():
    header("Wellbore Cutaway", "Underground schematic: geology, casing, cement, drill string, annulus and mud flow.")
    ctrl, view = st.columns([1.1, 3], gap="medium")
    with ctrl:
        with st.container(border=True):
            v = slider_grid([
                ("Well depth (ft)", "w_cw_depth", 1000, 11500, 100, 8000),
                ("Casing shoe (ft)", "w_cw_shoe", 500, 11000, 100, 5000),
                ("Open hole (in)", "w_cw_hole", 6.0, 8.75, 0.25, 8.5),
                ("Drill-pipe OD (in)", "w_cw_pod", 3.5, 6.0, 0.25, 5.0),
            ])
            depth, shoe, hole, pod = v["w_cw_depth"], v["w_cw_shoe"], v["w_cw_hole"], v["w_cw_pod"]
            cutaway = toggle("Cutaway (remove front half)", "w_cw_cut", True)
            exag = toggle("Exaggerated vertical scale (x2.5)", "w_cw_exag", False)
            st.markdown("**Components**")
            comps = {"layers": "Geological layers", "casing": "Casing", "cement": "Cement", "string": "Drill string & bit",
                     "annulus": "Annulus fluid", "mud": "Mud flow arrows"}
            tcols = st.columns(2)
            show_set = set()
            for i, (kk, lbl) in enumerate(comps.items()):
                with tcols[i % 2]:
                    if toggle(lbl, f"w_cw_{kk}", True):
                        show_set.add(kk)
            st.button("Reset", on_click=reset_keys, args=({"w_cw_depth": 8000, "w_cw_shoe": 5000, "w_cw_hole": 8.5, "w_cw_pod": 5.0,
                                                           "w_cw_cut": True, "w_cw_exag": False,
                                                           **{f"w_cw_{kk}": True for kk in comps}},), key="cw_reset")
    errs = []
    if pod >= hole:
        errs.append("Drill-pipe OD must be smaller than the open-hole diameter.")
    if shoe > depth:
        errs.append("Casing shoe cannot be deeper than the well depth.")
    with view:
        if errs:
            for e in errs:
                st.error(e)
        else:
            show(cutaway_figure(depth, shoe, hole, pod, cutaway, exag, show_set), key="cw_fig")
    cur = [n for n, t, b, _ in LAYERS if t <= depth < b]
    st.caption(f"Bit currently in the illustrative layer: **{cur[0] if cur else 'n/a'}**. Blue arrows: mud down the pipe; "
               "orange arrows: return flow up the annulus. Layers are illustrative and do not represent actual field geology.")


def page_mud():
    header("Mud Circulation Lab", "Mud Tank > Pump > Standpipe > Drill String > Bit > Annulus > Flowline > Shaker > Tank.")
    ctrl, view = st.columns([1.25, 3], gap="medium")
    with ctrl:
        with st.container(border=True):
            st.markdown("**Fluid and flow**")
            v = slider_grid([
                ("Mud weight (ppg)", "w_mud_mw", 7.0, 18.0, 0.1, 10.0),
                ("Flow rate (gpm)", "w_mud_q", 0, 1200, 10, 520),
                ("TVD (ft)", "w_mud_tvd", 500, 20000, 100, 8000),
                ("Viscosity (cP)", "w_mud_mu", 1.0, 60.0, 0.5, 15.0),
            ])
            st.markdown("**Geometry**")
            v.update(slider_grid([
                ("Hole diameter (in)", "w_mud_dh", 4.0, 17.5, 0.125, 8.5),
                ("Drill-pipe OD (in)", "w_mud_dp", 2.0, 8.0, 0.125, 5.0),
                ("Drill-pipe ID (in)", "w_mud_did", 1.0, 7.5, 0.001, 4.276, dict(fmt="%.3f")),
                ("Bit flow area (in^2)", "w_mud_tfa", 0.2, 2.0, 0.01, 0.75),
            ]))
            v.update(slider_grid([
                ("Resistance multiplier (-)", "w_mud_k", 0.5, 2.0, 0.05, 1.0,
                 dict(help="Dimensionless multiplier on the friction factor (1.0 = nominal; >1 = rougher / more viscous behaviour).")),
            ], 1))
            anim = toggle("Animate flow markers", "w_mud_anim", True)
            st.button("Reset parameters", on_click=reset_keys, args=({"w_mud_mw": 10.0, "w_mud_q": 520, "w_mud_tvd": 8000, "w_mud_dh": 8.5,
                                                                      "w_mud_dp": 5.0, "w_mud_did": 4.276, "w_mud_tfa": 0.75,
                                                                      "w_mud_mu": 15.0, "w_mud_k": 1.0, "w_mud_anim": True},), key="mud_reset")
    mw, q, tvd, mu = v["w_mud_mw"], v["w_mud_q"], v["w_mud_tvd"], v["w_mud_mu"]
    dh, dp, did, tfa, k = v["w_mud_dh"], v["w_mud_dp"], v["w_mud_did"], v["w_mud_tfa"], v["w_mud_k"]
    errs = validate_hydraulics(mw, q, tvd, dh, dp, did, tfa, mu, k)
    if errs:
        with view:
            for e in errs:
                st.error(e)
        return
    h = hydraulics_summary(mw, q, tvd, dh, dp, did, tfa, mu, k)
    a_flow = 8.0  # assumed flowline ID (in) for the schematic marker speed
    v_flow = velocity_ft_s(q, circle_area_in2(a_flow))
    with view:
        show(build_circulation_figure(tvd, h["v_pipe_fts"], h["v_ann_fts"], v_flow, q, anim, height=560), key="mud_fig")
        if anim and q > 0:
            st.caption("Press **Play** inside the chart to animate; marker speed per segment follows the computed velocity.")
        if q == 0:
            st.warning("Zero flow: no circulation, markers static; friction losses are zero and ECD equals mud weight.")
    c = st.columns(5)
    c[0].metric("Hydrostatic Ph", f"{h['hydrostatic_psi']:,.0f} psi")
    c[1].metric("Annular area", f"{h['annular_area_in2']:,.1f} in^2")
    c[2].metric("Annular velocity", f"{h['v_ann_fts']:.2f} ft/s", f"{h['v_ann_ftmin']:.0f} ft/min", delta_color="off")
    c[3].metric("Annular friction dP", f"{h['dp_ann_psi']:,.0f} psi")
    c[4].metric("ECD", f"{h['ecd_ppg']:.2f} ppg", f"{h['ecd_ppg'] - mw:+.2f} vs MW", delta_color="off")
    c = st.columns(5)
    c[0].metric("Q", f"{h['q_ft3s']:.3f} ft^3/s")
    c[1].metric("Pipe velocity", f"{h['v_pipe_fts']:.1f} ft/s")
    c[2].metric("Pipe + bit dP", f"{h['dp_pipe_psi'] + h['dp_bit_psi']:,.0f} psi")
    c[3].metric("Standpipe pressure (est.)", f"{h['spp_psi']:,.0f} psi")
    c[4].metric("Pump hydraulic hp", f"{h['pump_hhp']:,.0f} hp")
    left, right = st.columns(2)
    with left:
        zs = np.linspace(0, tvd, 60)
        ph = hydrostatic_psi(mw, zs)
        pc = ph + annular_pressure_loss_psi(q, mw, dh, dp, zs, mu, k)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ph, y=zs, name="Hydrostatic (static)", line=dict(color=COOL)))
        fig.add_trace(go.Scatter(x=pc, y=zs, name="Annulus while circulating", line=dict(color=WARM)))
        fig.update_yaxes(autorange="reversed", title_text="TVD (ft)")
        fig.update_xaxes(title_text="Annulus pressure (psi)")
        show(style_fig(fig, 320, "Annulus pressure vs depth"), key="mud_prof")
    with right:
        parts = {"Surface lines": h["dp_surface_psi"], "Drill pipe": h["dp_pipe_psi"], "Bit": h["dp_bit_psi"], "Annulus": h["dp_ann_psi"]}
        fig = go.Figure(go.Bar(x=list(parts), y=list(parts.values()), marker_color=[MUTED, COOL, "#2dd4bf", WARM]))
        fig.update_yaxes(title_text="psi")
        show(style_fig(fig, 320, "Circulating pressure losses (simplified)"), key="mud_loss")
    notes = [f"Flow regime in annulus (simplified Re = {h['re_ann']:,.0f}): {h['regime']}."]
    if q > 0 and h["v_ann_ftmin"] < 100:
        notes.append("Annular velocity is below the commonly quoted 100-200 ft/min guideline for hole cleaning (guideline only; depends on hole angle and mud).")
    st.caption(" ".join(notes) + " Hydrostatic pressure depends only on MW and TVD; friction losses depend on flow and appear only while circulating.")


def drill_controls():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slider("WOB (klbf)", "w_dr_wob", 2.0, 50.0, 0.5, 22.0)
        slider("RPM", "w_dr_rpm", 20.0, 220.0, 5.0, 110.0)
    with c2:
        slider("Pump flow rate (gpm)", "w_dr_flow", 0.0, 1200.0, 10.0, 520.0)
        slider("Mud weight (ppg)", "w_dr_mw", 7.0, 18.0, 0.1, 10.0)
    with c3:
        slider("Formation resistance (relative, 1 = reference)", "w_dr_fr", 0.4, 3.0, 0.05, 1.0)
        slider("Friction disturbance (0-1)", "w_dr_fric", 0.0, 1.0, 0.05, 0.3)
    with c4:
        slider("Stick-slip-like disturbance (0-1)", "w_dr_ss", 0.0, 1.0, 0.05, 0.3)
        slider("Run duration (min)", "w_dr_dur", 5.0, 60.0, 5.0, 30.0)
    with st.expander("Start depth and hydraulics geometry"):
        e1, e2, e3 = st.columns(3)
        with e1:
            slider("Start depth (ft)", "w_dr_depth", 1000.0, 20000.0, 100.0, 8000.0)
            slider("Hole diameter (in)", "w_dr_hole", 4.0, 17.5, 0.125, 8.5)
            slider("Drill-pipe OD (in)", "w_dr_pod", 2.0, 8.0, 0.125, 5.0)
        with e2:
            slider("Drill-pipe ID (in)", "w_dr_pid", 1.0, 7.5, 0.001, 4.276, fmt="%.3f")
            slider("Bit total flow area (in^2)", "w_dr_tfa", 0.2, 2.0, 0.01, 0.746)
        with e3:
            slider("Effective viscosity (cP)", "w_dr_mu", 1.0, 60.0, 0.5, 15.0)
            slider("Hydraulic-resistance multiplier", "w_dr_kres", 0.5, 2.0, 0.05, 1.0)
    with st.expander("Rheology, formation, bit wear and pressure window"):
        r1, r2, r3 = st.columns(3)
        with r1:
            st.selectbox("Rheology model", list(RHEO_MODELS), key="w_dr_rheo")
            st.selectbox("Bit type (visual only)", BIT_TYPES, key="w_dr_bit", help="Changes the 3D bit shown; the ROP/wear model does not depend on it.")
            slider("Fann theta600", "w_dr_t600", 20.0, 150.0, 1.0, 60.0)
            slider("Fann theta300", "w_dr_t300", 10.0, 100.0, 1.0, 38.0)
        with r2:
            slider("Lithology effect (0 = uniform, 1 = layered)", "w_dr_lith", 0.0, 1.0, 1.0, 1.0)
            slider("Initial bit wear (0-0.95)", "w_dr_wear0", 0.0, 0.95, 0.05, 0.0)
            slider("Cuttings size (in)", "w_dr_dcut", 0.05, 0.75, 0.05, 0.25)
        with r3:
            slider("Overpressure factor (0-1)", "w_dr_op", 0.0, 1.0, 0.05, 0.0)
            slider("Overpressure top (ft)", "w_dr_optop", 3000.0, 15000.0, 250.0, 9500.0)


def _start_sim():
    st.session_state.sim_running = True
    df_len = None
    p = get_drill_params(st.session_state.get("w_dr_case", "With disturbances") == "With disturbances")
    try:
        df_len = len(cached_sim(freeze(p)))
    except ValueError:
        st.session_state.sim_running = False
        return
    if st.session_state.sim_idx >= df_len - 1:
        st.session_state.sim_idx = 0


def _pause_sim():
    st.session_state.sim_running = False


def _reset_sim():
    st.session_state.sim_running = False
    st.session_state.sim_idx = 0


def page_drilling():
    header("Drilling Simulator", "Deterministic educational model with illustrative coefficients - not field-calibrated.")
    drill_controls()
    b1, b2, b3, b4, b5, b6 = st.columns([1, 1, 1, 2, 2, 2])
    b1.button("Start", on_click=_start_sim, key="dr_start", type="primary")
    b2.button("Pause", on_click=_pause_sim, key="dr_pause")
    b3.button("Reset", on_click=_reset_sim, key="dr_reset")
    with b4:
        case = st.radio("Playback case", ["With disturbances", "Baseline (no disturbances)"], horizontal=True, key="w_dr_case")
    with b5:
        overlay = toggle("Overlay baseline on charts", "w_dr_overlay", True)
    with b6:
        spd = st.select_slider("Playback speed (x real time)", options=[4, 8, 20, 40, 80], value=st.session_state.get("w_dr_speed", 20), key="w_dr_speed")
    b6.button("Restore default controls", on_click=reset_keys, args=(DRILL_DEFAULTS, {"w_dr_case": "With disturbances", "w_dr_overlay": True,
                                                                                       "w_dr_speed": 20, "sim_idx": 0, "sim_running": False}), key="dr_defaults")
    p = get_drill_params(case == "With disturbances")
    pb = get_drill_params(False)
    errs = validate_sim_params(p)
    if errs:
        for e in errs:
            st.error(e)
        st.session_state.sim_running = False
        return
    n = len(cached_sim(freeze(p)))
    st.session_state.sim_idx = int(min(st.session_state.sim_idx, n - 1))

    def live():
        ss = st.session_state
        df = cached_sim(freeze(p))
        base = cached_sim(freeze(pb))
        if ss.sim_running:
            ss.sim_idx = min(len(df) - 1, ss.sim_idx + max(1, int(spd // 2)))  # one tick per second
            if ss.sim_idx >= len(df) - 1:
                ss.sim_running = False
                st.rerun()
        idx = int(ss.sim_idx)
        r = df.iloc[idx]
        st.progress(idx / (len(df) - 1), text=f"Simulated time {r.t_s / 60:.1f} / {df.t_s.iloc[-1] / 60:.0f} min  -  "
                                              f"{'RUNNING' if ss.sim_running else ('FINISHED' if idx >= len(df) - 1 else 'PAUSED' if idx else 'READY')}")
        m = st.columns(6)
        m[0].metric("Depth", f"{r.depth_ft:,.1f} ft", f"+{r.depth_ft - df.depth_ft.iloc[0]:.1f} ft drilled", delta_color="off")
        m[1].metric("ROP", f"{r.rop_ftph:,.1f} ft/h")
        m[2].metric("Torque", f"{r.torque_kftlbf:,.2f} kft-lbf")
        m[3].metric("Bit RPM", f"{r.rpm_bit:,.0f}")
        m[4].metric("SPP", f"{r.spp_psi:,.0f} psi")
        m[5].metric("ECD", f"{r.ecd_ppg:.2f} ppg")
        left, right = st.columns([2, 3], gap="medium")
        with left:
            show(bit_view_figure(df, idx, kind=st.session_state.get("w_dr_bit"), lite=bool(ss.sim_running)), key="dr_bit")
        with right:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.085,
                                subplot_titles=("Depth (ft)", "ROP (ft/h)", "Torque (kft-lbf)", "Standpipe pressure (psi)"))
            for row, col, color in ((1, "depth_ft", COOL), (2, "rop_ftph", GREEN), (3, "torque_kftlbf", WARM), (4, "spp_psi", "#a78bfa")):
                t = df.t_s / 60
                fig.add_trace(go.Scatter(x=t, y=df[col], line=dict(color=color, width=1), opacity=0.25, showlegend=False), row, 1)
                fig.add_trace(go.Scatter(x=t[: idx + 1], y=df[col][: idx + 1], line=dict(color=color, width=1.6), showlegend=False), row, 1)
                if overlay and case == "With disturbances":
                    fig.add_trace(go.Scatter(x=t, y=base[col], line=dict(color="#111827", width=1, dash="dash"),
                                             name="Baseline", showlegend=row == 1), row, 1)
            fig.add_vline(x=r.t_s / 60, line_color=INK, line_dash="dot")
            fig.update_xaxes(title_text="Time (min)", row=4, col=1)
            fig.update_layout(legend=dict(orientation="h", y=1.08))
            show(style_fig(fig, 640), key="dr_ts")
        m2 = st.columns(5)
        m2[0].metric("Formation", str(r.formation))
        m2[1].metric("Bit wear", f"{r.wear * 100:.0f} %")
        m2[2].metric("MSE", f"{r.mse_psi / 1000:,.1f} kpsi")
        m2[3].metric("Cuttings conc.", f"{r.cut_conc * 100:.2f} %")
        m2[4].metric("Pit gain", f"{r.pit_gain_bbl:+.2f} bbl", f"flow out {r.flow_out_gpm - r.flow_set_gpm:+.0f} gpm", delta_color="off")
        if ss.sim_running:
            st.caption("Pressure-window and MSE charts refresh when the run is paused or finished (keeps live playback responsive).")
        else:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                f2 = go.Figure()
                t = df.t_s / 60
                f2.add_trace(go.Scatter(x=t, y=df.pore_ppg, name="Pore pressure", line=dict(color=COOL)))
                f2.add_trace(go.Scatter(x=t, y=df.frac_ppg, name="Fracture gradient", line=dict(color=WARM)))
                f2.add_trace(go.Scatter(x=t, y=df.ecd_cut_ppg, name="ECD + cuttings", line=dict(color=INK, dash="dot")))
                f2.add_trace(go.Scatter(x=t[: idx + 1], y=df.ecd_ppg[: idx + 1], name="ECD", line=dict(color=GREEN, width=2)))
                f2.update_layout(title="Pressure window (ppg equivalent)", yaxis_title="ppg", xaxis_title="Time (min)", legend=dict(orientation="h", y=-0.25))
                show(style_fig(f2, 340), key="dr_win")
            with c2:
                f3 = make_subplots(specs=[[{"secondary_y": True}]])
                f3.add_trace(go.Scatter(x=t, y=df.mse_psi / 1000, name="MSE (kpsi)", line=dict(color=WARM)), secondary_y=False)
                f3.add_trace(go.Scatter(x=t, y=df.wear * 100, name="Bit wear (%)", line=dict(color=COOL)), secondary_y=True)
                f3.update_layout(title="Mechanical specific energy and bit wear", xaxis_title="Time (min)", legend=dict(orientation="h", y=-0.25))
                show(style_fig(f3, 340), key="dr_mse")

    run_live(live, st.session_state.sim_running, 1.0)
    with st.expander("Model documentation and illustrative coefficients"):
        st.markdown(
            "ROP = ROP0 x f_wob x f_rpm x f_mw x f_hyd x f_fr x f_ss x f_fric with ROP0 = 60 ft/h (reference, illustrative).\n\n"
            "- f_wob = (WOB/20)^0.8 / (1 + (WOB/45)^4): power-law gain with a bit-floundering roll-off\n"
            "- f_rpm = (RPM/100)^0.6 / (1 + (RPM/220)^3)\n"
            "- f_mw = 1 / (1 + 0.12 max(0, MW - 9)): chip hold-down penalty\n"
            "- f_hyd = 0.55 + 0.45 min(1, (Q/(50 x D_hole))^0.5): cuttings removal\n"
            "- f_fr = 1 / FR; f_ss = 1 - 0.35 x SS; f_fric = 1 - 0.20 x FRIC\n\n"
            "Torque (kft-lbf) = [0.8 + 0.00015 x depth + 0.11 x WOB x sqrt(FR) x (1 + 1.2 x FRIC) + 0.004 x RPM] x (1 + 0.45 x SS x sin(2 pi t / 8 s)).\n\n"
            "Stick-slip-like disturbance: bit RPM = RPM x (1 + 0.8 x SS x sin(2 pi t / 8 s)), surface RPM shows a 15 % x SS ripple. "
            "SPP/ECD come from the simplified hydraulics model (Mud Circulation Lab). Sensor noise is seeded (1 % default). "
            "These coefficients are assumptions for teaching, not a universal ROP law and not calibrated to any field.")


def step_tool(state: dict, target: float, speed_ftmin: float, tc: float, approach_ft: float, drag_fn=None):
    """Advance a RIH/POOH tool by wall-clock dt x time-compression. Transparent educational rule:
    effective speed = commanded speed x drag factor x approach factor (slows within `approach_ft` of the end point)."""
    now = time.time()
    dt_real = min(max(now - state["last"], 0.0), 2.0) if state["last"] else 0.5
    state["last"] = now
    dt = dt_real * tc
    state["elapsed"] += dt
    goal = target if state["dir"] == "RIH" else 0.0
    remaining = abs(goal - state["depth"])
    if remaining <= 1e-6:
        state["running"] = False
        return
    appr = min(1.0, max(0.1, remaining / approach_ft))
    drag = drag_fn(state["depth"]) if drag_fn else 1.0
    move = speed_ftmin * appr * drag * dt / 60.0
    move = min(move, remaining)
    state["depth"] += move if state["dir"] == "RIH" else -move
    if abs(goal - state["depth"]) <= 1e-6:
        state["depth"] = goal
        state["running"] = False


def _tool_buttons(skey, target, prefix):
    s = st.session_state[skey]
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 2])

    def start():
        s["dir"] = st.session_state[f"w_{prefix}_dir"]
        goal = target if s["dir"] == "RIH" else 0.0
        if abs(goal - s["depth"]) > 1e-6:
            s["running"] = True
            s["last"] = time.time()

    def pause():
        s["running"] = False

    def reset():
        s.update(depth=0.0, elapsed=0.0, running=False, last=0.0)

    c1.button("Start", on_click=start, key=f"{prefix}_start", type="primary")
    c2.button("Pause", on_click=pause, key=f"{prefix}_pause")
    c3.button("Reset", on_click=reset, key=f"{prefix}_reset")
    with c4:
        st.radio("Direction", ["RIH", "POOH"], horizontal=True, key=f"w_{prefix}_dir",
                 help="RIH = run in hole, POOH = pull out of hole")
    with c5:
        st.select_slider("Time compression (sim s per real s)", options=[1, 5, 15, 60], value=st.session_state.get(f"w_{prefix}_tc", 15),
                         key=f"w_{prefix}_tc")


def page_wireline():
    header("Wireline Operations", "Wireline unit, cable reel, lubricator, wellhead and downhole tool - kinematic educational model.")
    ss = st.session_state
    c1, c2, c3 = st.columns(3)
    with c1:
        target = slider("Target depth (ft)", "w_wl_target", 500.0, 20000.0, 100.0, 8000.0)
    with c2:
        speed = slider("Cable speed (ft/min)", "w_wl_speed", 10.0, 300.0, 5.0, 100.0)
    with c3:
        tool_w = slider("Tool string weight in air (lb)", "w_wl_tw", 20.0, 600.0, 10.0, 120.0)
    ss.setdefault("w_wl_dir", "RIH")
    _tool_buttons("wl", target, "wl")
    tc = ss.get("w_wl_tc", 15)
    s = ss["wl"]

    def live():
        if s["running"]:
            step_tool(s, target, speed, tc, approach_ft=100.0)
            if not s["running"]:
                st.rerun()
        depth = min(s["depth"], target) if s["dir"] == "RIH" and not s["running"] else s["depth"]
        left, right = st.columns([3, 1], gap="medium")
        with left:
            show(wireline_figure(depth, target), key="wl_fig")
        with right:
            tension = (0.10 * depth + tool_w) * 0.85
            stt = "RUNNING" if s["running"] else ("AT TARGET" if abs(depth - target) < 1e-6 else "AT SURFACE" if depth <= 0 else "PAUSED")
            st.markdown(badge(stt, "ok" if s["running"] else "warn"), unsafe_allow_html=True)
            st.metric("Current depth", f"{depth:,.1f} ft")
            st.metric("Elapsed simulated time", f"{int(s['elapsed'] // 3600)}h {int(s['elapsed'] % 3600 // 60):02d}m {int(s['elapsed'] % 60):02d}s")
            st.metric("Progress to target", f"{100 * depth / target:.1f} %")
            st.metric("Indicative cable tension", f"{tension:,.0f} lbf", help="(0.10 lb/ft x depth + tool weight) x 0.85 buoyancy; ignores friction, pressure and dynamics (illustrative).")
            st.caption(f"Direction: {s['dir']}")

    run_live(live, s["running"], 0.5)
    st.caption("Educational model: depth changes at the commanded speed, automatically slowed (down to 10 %) within 100 ft of the end point. "
               "Real wireline operations involve additional mechanical (cable strength, jars, friction, dogleg), pressure-control (lubricator, "
               "stuffing box, BOP), electrical, procedural and safety constraints that are not modelled here.")


def page_coiled():
    header("Coiled Tubing Lab", "Reel, gooseneck, injector head, tubing path, wellhead and downhole BHA - transparent educational model.")
    ss = st.session_state
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target = slider("Target depth (ft)", "w_ct_target", 500.0, 20000.0, 100.0, 9000.0)
    with c2:
        speed = slider("Movement speed (ft/min)", "w_ct_speed", 5.0, 150.0, 5.0, 60.0)
    with c3:
        pump = slider("Pump rate (bbl/min)", "w_ct_pump", 0.0, 3.0, 0.1, 1.0)
    with c4:
        lock = slider("Assumed lock-up depth (ft)", "w_ct_lock", 5000.0, 30000.0, 500.0, 18000.0,
                      help="Illustrative parameter: the drag factor falls as depth approaches this value.")
    ss.setdefault("w_ct_dir", "RIH")
    _tool_buttons("ct", target, "ct")
    tc = ss.get("w_ct_tc", 15)
    s = ss["ct"]
    drag = lambda d: float(np.clip(1.0 - 0.55 * (d / lock) ** 2, 0.15, 1.0))  # noqa: E731
    OD, ID, TOTAL, W_FT = 1.75, 1.532, 15000.0, 2.2  # in, in, ft on reel, lb/ft (illustrative)
    CASING_ID, MW, MU = 4.892, 8.6, 3.0

    def live():
        if s["running"]:
            step_tool(s, target, speed, tc, approach_ft=150.0, drag_fn=drag)
            if not s["running"]:
                st.rerun()
        depth = s["depth"]
        q = pump * 42.0
        dp_ct = pipe_pressure_loss_psi(q, MW, ID, TOTAL, MU, 1.15) if q > 0 else 0.0
        dp_an = annular_pressure_loss_psi(q, MW, CASING_ID, OD, depth, MU, 1.0) if q > 0 else 0.0
        left, right = st.columns([3, 1], gap="medium")
        with left:
            show(coiled_tubing_figure(depth, target), key="ct_fig")
        with right:
            stt = "RUNNING" if s["running"] else ("AT TARGET" if abs(depth - target) < 1e-6 else "AT SURFACE" if depth <= 0 else "PAUSED")
            st.markdown(badge(stt, "ok" if s["running"] else "warn"), unsafe_allow_html=True)
            st.metric("Tubing depth", f"{depth:,.1f} ft")
            st.metric("Effective speed factor", f"{drag(depth):.2f}", help="Drag factor = clip(1 - 0.55 (depth/lock-up)^2, 0.15, 1)")
            st.metric("Circulating pressure (est.)", f"{dp_ct + dp_an:,.0f} psi", help="Simplified Newtonian friction through 15,000 ft of 1.75 in CT (x1.15 curvature) plus the annulus to depth.")
            st.metric("Buoyed tubing weight", f"{W_FT * depth * (1 - MW / 65.4):,.0f} lbf", help="2.2 lb/ft x depth x buoyancy factor; ignores friction and buckling.")
            st.metric("Tubing left on reel", f"{max(TOTAL - depth, 0):,.0f} ft")
            st.metric("Elapsed simulated time", f"{int(s['elapsed'] // 3600)}h {int(s['elapsed'] % 3600 // 60):02d}m {int(s['elapsed'] % 60):02d}s")

    run_live(live, s["running"], 0.5)
    st.caption("Educational model, not a validated field simulator. Effective speed = commanded speed x drag factor x approach factor. "
               "Helical buckling, lock-up, fatigue, pressure-control and well-control constraints are not modelled; fluid is 8.6 ppg, 3 cP in 1.75 in OD x 1.532 in ID tubing, 4.892 in casing ID.")


def _plot_ml_run(run_df, pred_if, scores, thr, title):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Torque (kft-lbf)", "SPP (psi)", "Flow (gpm)", "Surface RPM", "Isolation Forest anomaly score"))
    t = run_df["t_s"] / 60
    lab = run_df["label"].values.astype(bool)
    pr = pred_if.astype(bool)
    for r, col, color in ((1, "torque_kftlbf", WARM), (2, "spp_psi", COOL), (3, "flow_gpm", GREEN), (4, "rpm", "#a78bfa")):
        fig.add_trace(go.Scatter(x=t, y=run_df[col], line=dict(color=color, width=1), name="Sensor", showlegend=False), r, 1)
        fig.add_trace(go.Scatter(x=t[lab], y=run_df[col][lab], mode="markers", marker=dict(color=RED, size=5),
                                 name="True synthetic anomaly", showlegend=r == 1), r, 1)
        fig.add_trace(go.Scatter(x=t[pr], y=run_df[col][pr], mode="markers", marker=dict(color=INK, size=8, symbol="x-thin", line=dict(width=1.5, color=INK)),
                                 name="Flagged by detector", showlegend=r == 1), r, 1)
    fig.add_trace(go.Scatter(x=t, y=scores, line=dict(color=MUTED, width=1), showlegend=False), 5, 1)
    fig.add_hline(y=thr, line_color=WARM, line_dash="dash", row=5, col=1, annotation_text="threshold")
    fig.update_xaxes(title_text="Time (min)", row=5, col=1)
    return style_fig(fig, 860, title)


def _cm_fig(m, title):
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred normal", "Pred anomaly"], y=["True normal", "True anomaly"], colorscale=[[0, "#ffffff"], [1, "#7fbfb2"]],
                               text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}", textfont=dict(size=16, color=INK), showscale=False))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 260, title)


def page_ai():
    header("AI Diagnostics", "Isolation Forest proof of concept on SYNTHETIC drilling data - compared with a static-limit baseline.")
    st.warning("All data and anomalies here are synthetic, generated from the simplified simulator. Results do NOT demonstrate real-world fault-detection capability. "
               "Isolation Forest scores are anomaly rankings, not calibrated failure probabilities.")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_est = slider("Trees (n_estimators)", "w_ai_trees", 50, 300, 25, 150)
    with c2:
        pct = slider("Threshold percentile of normal training scores", "w_ai_pct", 95.0, 99.9, 0.1, 98.0)
    with c3:
        zk = slider("Baseline limit (+/- k sigma)", "w_ai_k", 2.0, 5.0, 0.5, 3.0)
    with st.spinner("Generating synthetic runs and training (cached)..."):
        train, test = cached_ml_data()
        det = cached_detector(int(n_est), float(pct))
    sc = det.scores(test)
    pred_if = (sc > det.threshold).astype(int)
    pred_bl = det.baseline_predict(test, zk)
    y = test["label"].values
    m_if, m_bl = eval_binary(y, pred_if), eval_binary(y, pred_bl)
    ev_if, ev_bl = event_recall(test, pred_if), event_recall(test, pred_bl)
    k = st.columns(4)
    k[0].metric("Test samples", f"{len(test):,}", f"{int(y.sum())} synthetic anomalous", delta_color="off")
    k[1].metric("Isolation Forest flagged", f"{m_if['flagged']:,}")
    k[2].metric("Baseline flagged", f"{m_bl['flagged']:,}")
    k[3].metric("Anomaly events detected (IF / baseline)", f"{ev_if[0]}/{ev_if[1]}  |  {ev_bl[0]}/{ev_bl[1]}")
    tab = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 score", "False-positive rate", "True pos / False pos / False neg / True neg"],
        "Isolation Forest": [f"{m_if['precision']:.3f}", f"{m_if['recall']:.3f}", f"{m_if['f1']:.3f}", f"{m_if['fpr']:.4f}",
                             f"{m_if['tp']} / {m_if['fp']} / {m_if['fn']} / {m_if['tn']}"],
        f"Baseline (+/-{zk:g} sigma on torque, SPP, flow)": [f"{m_bl['precision']:.3f}", f"{m_bl['recall']:.3f}", f"{m_bl['f1']:.3f}",
                                                             f"{m_bl['fpr']:.4f}", f"{m_bl['tp']} / {m_bl['fp']} / {m_bl['fn']} / {m_bl['tn']}"],
    })
    st.dataframe(tab, hide_index=True)
    a, b = st.columns(2)
    with a:
        show(_cm_fig(m_if, "Isolation Forest - confusion matrix"), key="cm_if")
    with b:
        show(_cm_fig(m_bl, "Baseline - confusion matrix"), key="cm_bl")
    rows = []
    for typ in ANOM_TYPES:
        msk = test["anom_type"].values == typ
        if msk.any():
            rows.append(dict(anomaly_type=typ, samples=int(msk.sum()), recall_IF=float(pred_if[msk].mean()), recall_baseline=float(pred_bl[msk].mean())))
    st.markdown("**Recall by synthetic anomaly type**")
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    runs = sorted(test["run_id"].unique())
    run = st.selectbox("Test run to plot (independent seed)", runs, key="w_ai_run",
                       format_func=lambda r: f"Run {r} ({int(test[test.run_id == r].label.sum())} anomalous samples)")
    mask = (test["run_id"] == run).values
    show(_plot_ml_run(test[mask].reset_index(drop=True), pred_if[mask], sc[mask], det.threshold, f"Test run {run}"), key="ai_ts")
    with st.expander("Model configuration and evaluation explanation", expanded=False):
        st.markdown(
            f"- **Data**: {train.run_id.nunique()} normal training runs (seeds 1-{train.run_id.nunique()}) and {test.run_id.nunique()} independent test runs "
            "(seeds 1000+, different operating conditions and noise; 2 runs are fully normal). Fixed seeds make results reproducible.\n"
            "- **Anomalies (synthetic, labelled)**: elevated torque, standpipe-pressure deviation (plugging/washout-like), flow deviation, stick-slip-like torque/RPM oscillation.\n"
            f"- **Features** ({', '.join(FEATURES)}): deviation of torque and SPP from the run's own trailing 4-minute median (causal), flow deviation from set-point, 16 s rolling variability, and measured/model-expected ratios. "
            "Labels are never used as features, and training uses normal runs only, so there is no target leakage; evaluation uses runs not seen in training.\n"
            f"- **Isolation Forest**: n_estimators={det.n_estimators}, max_samples=256, random_state={ML_SEED}; trained on {det.n_train_rows:,} normal rows. "
            f"Alarm threshold = {det.pct:g}th percentile of normal training scores ({det.threshold:.4f}).\n"
            "- **Baseline**: static mean +/- k sigma limits on raw torque, SPP and flow from the same training data (flags if any channel exceeds).\n"
            "- **Model selection**: the feature set and the default 98th-percentile threshold were chosen on a separate validation set (seeds 2000+), not on these test runs.\n"
            "- **Caveat**: test anomalies come from the same generator that defines the features, so performance is optimistic relative to field data.")
    st.markdown("### Score the current Drilling Simulator run")
    p, df, idx = get_current_sim()
    if df is None:
        st.info("Drilling Simulator inputs are invalid.")
        return
    feats = build_features(df, p)
    sc2 = det.scores(feats)
    pr2 = sc2 > det.threshold
    pr2[:WARMUP_SAMPLES] = False  # warm-up samples excluded
    st.caption(f"Flagged samples in this simulator run: **{int(pr2.sum())} of {len(df)}** (case: {st.session_state.get('w_dr_case', 'With disturbances')}). "
               "Disturbances set on the Drilling Simulator page are the only thing that can trigger flags here; no labels exist for this run.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Torque (kft-lbf)", "Anomaly score"))
    t = df.t_s / 60
    fig.add_trace(go.Scatter(x=t, y=df.torque_kftlbf, line=dict(color=WARM, width=1), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=t[pr2], y=df.torque_kftlbf[pr2], mode="markers", marker=dict(color=INK, size=6, symbol="x-thin", line=dict(width=1.5, color=INK)), name="Flagged"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=sc2, line=dict(color=MUTED, width=1), showlegend=False), 2, 1)
    fig.add_hline(y=det.threshold, line_color=WARM, line_dash="dash", row=2, col=1)
    show(style_fig(fig, 380), key="ai_sim")


SC_FIELDS = [("wob", "WOB (klbf)", 2.0, 50.0, 0.5), ("rpm", "RPM", 20.0, 220.0, 5.0), ("flow", "Flow rate (gpm)", 0.0, 1200.0, 10.0),
             ("mw", "Mud weight (ppg)", 7.0, 18.0, 0.1), ("fr", "Formation resistance (-)", 0.4, 3.0, 0.05),
             ("fric", "Friction disturbance (0-1)", 0.0, 1.0, 0.05), ("ss", "Stick-slip-like disturbance (0-1)", 0.0, 1.0, 0.05),
             ("start_depth", "Start depth (ft)", 1000.0, 20000.0, 100.0)]
SC_DEFAULTS = {
    "A": dict(wob=22.0, rpm=110.0, flow=520.0, mw=10.0, fr=1.0, fric=0.0, ss=0.0, start_depth=8000.0),
    "B": dict(wob=28.0, rpm=130.0, flow=560.0, mw=10.4, fr=1.0, fric=0.3, ss=0.3, start_depth=8000.0),
}


def page_scenarios():
    header("Scenario Comparison", "Two independently configured simulator runs, compared on the same time axis.")
    cols = st.columns(2)
    for col, tag in zip(cols, ("A", "B")):
        with col, st.container(border=True):
            st.markdown(f"#### Scenario {tag}")
            slider_grid([(label, f"w_sc_{tag}_{name}", lo, hi, step, SC_DEFAULTS[tag][name]) for name, label, lo, hi, step in SC_FIELDS])
    dur = slider("Run duration (min)", "w_sc_dur", 5.0, 60.0, 5.0, 30.0)
    bc = st.columns([1, 1, 4])
    all_defaults = {f"w_sc_{t}_{n}": v for t in ("A", "B") for n, v in SC_DEFAULTS[t].items()}
    all_defaults["w_sc_dur"] = 30.0
    bc[0].button("Reset both", on_click=reset_keys, args=(all_defaults,), key="sc_reset")
    bc[1].button("Copy A to B", on_click=reset_keys, args=({f"w_sc_B_{n}": st.session_state[f"w_sc_A_{n}"] for n, *_ in SC_FIELDS},), key="sc_copy")
    params, sims = {}, {}
    for tag in ("A", "B"):
        p = dict(SIM_DEFAULTS)
        p.update({n: float(st.session_state[f"w_sc_{tag}_{n}"]) for n, *_ in SC_FIELDS})
        p["duration_min"] = float(dur)
        errs = validate_sim_params(p)
        if errs:
            st.error(f"Scenario {tag}: " + " ".join(errs))
            return
        params[tag] = p
        sims[tag] = cached_sim(freeze(p))
    colors = {"A": COOL, "B": WARM}
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.09, horizontal_spacing=0.08,
                        subplot_titles=("Depth (ft)", "ROP (ft/h)", "Torque (kft-lbf)", "Standpipe pressure (psi)", "ECD (ppg)", "Bit RPM"))
    for tag, df in sims.items():
        for i, col in enumerate(("depth_ft", "rop_ftph", "torque_kftlbf", "spp_psi", "ecd_ppg", "rpm_bit")):
            fig.add_trace(go.Scatter(x=df.t_s / 60, y=df[col], name=f"Scenario {tag}", line=dict(color=colors[tag], width=1.3),
                                     legendgroup=tag, showlegend=i == 0), i // 2 + 1, i % 2 + 1)
    fig.update_xaxes(title_text="Time (min)", row=3)
    show(style_fig(fig, 700), key="sc_fig")
    summ = pd.DataFrame({tag: dict(
        final_depth_ft=df.depth_ft.iloc[-1], footage_drilled_ft=df.depth_ft.iloc[-1] - df.depth_ft.iloc[0], mean_rop_ftph=df.rop_ftph.mean(),
        mean_torque_kftlbf=df.torque_kftlbf.mean(), max_torque_kftlbf=df.torque_kftlbf.max(), peak_spp_psi=df.spp_psi.max(),
        final_ecd_ppg=df.ecd_ppg.iloc[-1], final_hydrostatic_psi=df.hyd_psi.iloc[-1]) for tag, df in sims.items()})
    summ["B - A"] = summ["B"] - summ["A"]
    summ["B vs A (%)"] = np.where(summ["A"] != 0, 100 * summ["B - A"] / summ["A"].replace(0, np.nan), np.nan)
    st.markdown("**Output summary**")
    st.dataframe(summ.round(3))
    diff = [dict(parameter=lbl, A=params["A"][n], B=params["B"][n], delta=params["B"][n] - params["A"][n])
            for n, lbl, *_ in SC_FIELDS if params["A"][n] != params["B"][n]]
    st.markdown("**Parameter differences**")
    st.dataframe(pd.DataFrame(diff).round(3), hide_index=True) if diff else st.info("Scenarios have identical parameters.")
    fa, fb = rop_factor_terms(params["A"]), rop_factor_terms(params["B"])
    rel = {k: fb[k] / fa[k] if fa[k] else np.nan for k in fa}
    lbl = {"f_wob": "WOB", "f_rpm": "RPM", "f_mw": "mud weight (chip hold-down)", "f_hyd": "flow rate (hole cleaning)",
           "f_fr": "formation resistance", "f_ss": "stick-slip-like disturbance", "f_fric": "friction disturbance"}
    ranked = sorted([k for k in rel if abs(rel[k] - 1) > 1e-3], key=lambda k: -abs(np.log(rel[k])))
    rop_a, rop_b = sims["A"].rop_ftph.mean(), sims["B"].rop_ftph.mean()
    txt = f"Scenario B's mean ROP is {100 * (rop_b / rop_a - 1):+.1f}% relative to A ({rop_b:.1f} vs {rop_a:.1f} ft/h). "
    if ranked:
        txt += "Model ROP factors that differ (B/A): " + "; ".join(f"{lbl[k]} x{rel[k]:.2f}" for k in ranked) + ". "
    def _pct(b, a):
        return f"{100 * (b / a - 1):+.1f}%" if abs(a) > 1e-9 else "n/a"
    txt += (f"Torque is {_pct(sims['B'].torque_kftlbf.mean(), sims['A'].torque_kftlbf.mean())} and peak SPP "
            f"{_pct(sims['B'].spp_psi.max(), sims['A'].spp_psi.max())}; hydrostatic pressure at final depth differs by "
            f"{summ.loc['final_hydrostatic_psi', 'B - A']:+,.0f} psi. These are outputs of the illustrative model, not field predictions.")
    st.markdown(txt)
    long = pd.concat([d.assign(scenario=t) for t, d in sims.items()], ignore_index=True)
    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("Download time series (CSV)", long.to_csv(index=False).encode(), "petrosite_scenarios_timeseries.csv", "text/csv", key="dl_ts")
    summ_out = summ.copy()
    summ_out.insert(0, "metric", summ_out.index)
    d2.download_button("Download summary (CSV)", summ_out.to_csv(index=False).encode(), "petrosite_scenarios_summary.csv", "text/csv", key="dl_sum")


def run_self_tests() -> list[tuple[str, bool, str]]:
    """Real checks executed on demand (never pre-recorded)."""
    res = []

    def chk(name, ok, detail):
        res.append((name, bool(ok), detail))

    v = hydrostatic_psi(10.0, 10000.0)
    chk("Hydrostatic 10 ppg @ 10,000 ft = 5,200 psi", abs(v - 5200.0) < 1e-6, f"{v:.3f} psi")
    a = annular_area_in2(8.5, 5.0)
    chk("Annular area 8.5 in x 5 in = 37.110 in^2", abs(a - np.pi/4*(8.5**2-5**2)) < 1e-9 and abs(a - 37.1101) < 1e-3, f"{a:.4f} in^2")
    qs = gpm_to_ft3s(600.0)
    chk("600 gpm = 1.3368 ft^3/s", abs(qs - 1.336806) < 1e-5, f"{qs:.6f} ft^3/s")
    vv = velocity_ft_s(600.0, a)
    chk("Annular velocity = Q/A with in^2->ft^2 conversion (5.187 ft/s)", abs(vv - gpm_to_ft3s(600.0) / (a / 144)) < 1e-9 and abs(vv - 5.187) < 5e-3, f"{vv:.4f} ft/s")
    chk("Zero flow gives zero friction and ECD = MW", annular_pressure_loss_psi(0.0, 10.0, 8.5, 5.0, 8000.0, 15.0) == 0.0
        and abs(ecd_ppg(10.0, 0.0, 8000.0) - 10.0) < 1e-12, "dP=0, ECD=10")
    chk("Invalid: hole <= pipe is rejected", len(validate_hydraulics(10, 500, 8000, 5.0, 5.0)) > 0, "validation error returned")
    chk("Invalid: negative flow is rejected", len(validate_hydraulics(10, -1, 8000, 8.5, 5.0)) > 0, "validation error returned")
    h = hydraulics_summary(10.0, 520.0, 8000.0, 8.5, 5.0)
    chk("ECD >= MW with circulation", h["ecd_ppg"] > 10.0, f"ECD = {h['ecd_ppg']:.3f} ppg")
    d1, d2 = simulate_drilling(dict(SIM_DEFAULTS)), simulate_drilling(dict(SIM_DEFAULTS))
    chk("Simulation is reproducible", d1.equals(d2), f"{len(d1)} steps")
    chk("Depth is non-decreasing", bool((np.diff(d1.depth_ft) >= 0).all()), f"{d1.depth_ft.iloc[0]:.1f} -> {d1.depth_ft.iloc[-1]:.1f} ft")
    ex = simulate_drilling({**SIM_DEFAULTS, "ss": 0.8})
    chk("Stick-slip increases torque variability", ex.torque_kftlbf.std() > d1.torque_kftlbf.std(), f"{ex.torque_kftlbf.std():.3f} > {d1.torque_kftlbf.std():.3f}")
    try:
        simulate_drilling({**SIM_DEFAULTS, "hole_in": 4.0, "pipe_od": 5.0})
        chk("Simulator rejects invalid geometry", False, "no error raised")
    except ValueError:
        chk("Simulator rejects invalid geometry", True, "ValueError raised")
    train, test = cached_ml_data()
    det = cached_detector()
    m = eval_binary(test["label"].values, (det.scores(test) > det.threshold).astype(int))
    chk("ML: training data has no anomalies (normal-only)", int(train["label"].sum()) == 0, f"{len(train):,} rows")
    chk("ML: test runs are independent of training runs", set(train.run_id) .isdisjoint(set(test.run_id)), "disjoint seeds")
    chk("ML: detector produces finite metrics", all(np.isfinite([m["precision"], m["recall"], m["f1"], m["fpr"]])), f"F1={m['f1']:.3f}")
    return res


# ---- realistic SRP scene (client-side animation: the browser plays the frames, the server is not involved) -------------
def _pack_items(items):
    """Turn a list of box/arc items (x-z plane geometry with a y thickness) into flat x, y, z arrays and triangle indices."""
    xs, ys, zs, I, J, K = [], [], [], [], [], []
    for it in items:
        off = len(xs)
        if it[0] == "box":
            _, p0, p1, t, y0, y1 = it
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            d = p1 - p0
            ln = float(np.hypot(*d)) or 1e-9
            nrm = np.array([-d[1], d[0]]) / ln * t / 2.0
            corners = [p0 + nrm, p0 - nrm, p1 - nrm, p1 + nrm]
            for y in (y0, y1):
                for c in corners:
                    xs.append(c[0]); ys.append(y); zs.append(c[1])
            f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
            for k in range(4):
                a, b = k, (k + 1) % 4
                f += [(a, b, 4 + b), (a, 4 + b, 4 + a)]
        else:  # arc slab
            _, cx, cz, r0, r1, th0, th1, y0, y1, n = it
            th = np.linspace(th0, th1, n)
            for y in (y0, y1):
                for r in (r0, r1):
                    for t_ in th:
                        xs.append(cx + r * np.cos(t_)); ys.append(y); zs.append(cz + r * np.sin(t_))
            m = n
            idx = lambda yy, rr, k: (yy * 2 + rr) * m + k  # noqa: E731
            f = []
            for k in range(m - 1):
                for yy in (0, 1):  # inner and outer faces handled below
                    pass
                f += [(idx(0, 1, k), idx(0, 1, k + 1), idx(1, 1, k + 1)), (idx(0, 1, k), idx(1, 1, k + 1), idx(1, 1, k)),  # outer surface
                      (idx(0, 0, k), idx(1, 0, k + 1), idx(0, 0, k + 1)), (idx(0, 0, k), idx(1, 0, k), idx(1, 0, k + 1)),  # inner surface
                      (idx(0, 0, k), idx(0, 0, k + 1), idx(0, 1, k + 1)), (idx(0, 0, k), idx(0, 1, k + 1), idx(0, 1, k)),  # side y0
                      (idx(1, 0, k), idx(1, 1, k + 1), idx(1, 0, k + 1)), (idx(1, 0, k), idx(1, 1, k), idx(1, 1, k + 1))]  # side y1
            f += [(idx(0, 0, 0), idx(0, 1, 0), idx(1, 1, 0)), (idx(0, 0, 0), idx(1, 1, 0), idx(1, 0, 0)),
                  (idx(0, 0, m - 1), idx(1, 1, m - 1), idx(0, 1, m - 1)), (idx(0, 0, m - 1), idx(1, 0, m - 1), idx(1, 1, m - 1))]
        for a, b, c in f:
            I.append(a + off); J.append(b + off); K.append(c + off)
    return np.array(xs), np.array(ys), np.array(zs), I, J, K


def _mesh(x, y, z, I, J, K, color, name, opacity=1.0):
    return go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=color, opacity=opacity, flatshading=False, name=name, hoverinfo="skip", showlegend=False,
                     lighting=dict(ambient=0.42, diffuse=0.95, specular=0.7, roughness=0.3, fresnel=0.35), lightposition=dict(x=250, y=-350, z=550))


def _static_srp(mb_fn):
    mb = MeshBuilder()
    mb_fn(mb)
    return mb


def srp_unit_figure(res: dict, p: dict, n_frames: int = 48, height: int = 640, camera: str = "Side"):
    """
    Conventional (Class I) pumping unit built from the computed four-bar geometry, plus a downhole-pump inset (second scene).
    Moving parts are Mesh3d frames played by the browser; static parts are meshes built once.  Proportions follow the kinematic solution
    (beam arms, pitman, crank radius) with a typical real-world size; the downhole inset is not to scale.
    """
    q = gq()
    n_cyl = int(max(10, 10 * q))
    phi_k = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    r_c, pm = 0.17, 1.0
    beta_all = srp_four_bar(np.linspace(0, 2 * np.pi, 720, endpoint=False), r=r_c, pm=pm)
    beta_max, span = float(beta_all.max()), float(beta_all.max() - beta_all.min())
    stroke_ft = p["stroke"] / 12.0
    I = stroke_ft / span  # front beam arm (ft)
    P = I / 1.156  # rear beam arm (ft)
    r_ft, pm_ft = r_c * P, pm * P
    px, pz = I, 21.0  # saddle-bearing position (wellhead at x = 0)
    shaft = np.array([px + P, pz - (pm + r_c) * P])
    z_car0 = 5.0  # carrier-bar height at the bottom of the stroke
    beta = srp_four_bar(phi_k, r=r_c, pm=pm)
    y_ft = I * (beta_max - beta)  # polished-rod rise (ft)

    # ---------------- static scene ----------------
    gnd = MeshBuilder()
    gnd.box(14, 0, -0.05, 76, 38, 0.1)
    pad = MeshBuilder()
    pad.box(12, 0, 0.03, 40, 20, 0.06)
    base = MeshBuilder()
    for yy in (-1.8, 1.8):
        base.box(px + P * 0.5 + 1.0, yy, 0.55, 34.0, 0.7, 1.1)  # skid I-beams
    for xx in np.linspace(px - 6, px + P + 10, 7):
        base.box(xx, 0, 0.5, 0.7, 4.4, 0.8)
    base.box(px + P, 0, 0.9, 6.0, 5.2, 0.6)  # gearbox foundation
    steel = MeshBuilder()  # samson post, gearbox, wellhead hardware
    for sx in (-1, 1):
        steel.cone_between((px - 3.6, sx * 2.0, 1.0), (px, sx * 0.55, pz - 0.7), 0.22, 0.22, n_cyl)  # front legs
        steel.cone_between((px + 3.8, sx * 2.0, 1.0), (px, sx * 0.55, pz - 0.7), 0.22, 0.22, n_cyl)  # rear legs
        steel.cone_between((px - 3.6, sx * 2.0, 1.0), (px + 3.8, sx * 2.0, 1.0), 0.15, 0.15, n_cyl)
    for zz, w in ((6.5, 0.55), (12.0, 0.8)):  # cross-braces
        f = (zz - 1.0) / (pz - 1.7)
        steel.cone_between((px - 3.6 + f * 3.6, -2.0 + f * 1.45, zz), (px - 3.6 + f * 3.6, 2.0 - f * 1.45, zz), 0.12, 0.12, n_cyl)
    steel.box(px, 0, pz - 0.9, 1.6, 2.0, 0.8)  # centre bearing housing
    steel.frustum(px, -1.3, pz, 2.6, 0.55, 0.55, axis="y", n=18)  # saddle bearing / tail bearing pin
    steel.box(shaft[0], 0, shaft[1] - 0.2, 3.6, 2.4, 3.0)  # gearbox
    steel.frustum(shaft[0], -1.6, shaft[1], 3.2, 0.42, 0.42, axis="y", n=14)  # crankshaft
    motor = MeshBuilder()
    mx, mz = px + P + 7.2, 2.4
    motor.box(mx, 0, 1.0, 4.4, 2.4, 0.4)
    motor.box(mx, 0, mz, 3.4, 1.7, 2.1)
    motor.frustum(mx - 1.4, -1.3, mz + 0.2, 0.8, 0.9, 0.9, axis="y", n=16)  # motor sheave
    brk = MeshBuilder()
    brk.frustum(shaft[0] + 0.1, 1.1, shaft[1], 0.5, 1.4, 1.4, axis="y", n=24)  # brake drum
    guard = MeshBuilder()
    guard.box((mx - 1.4 + shaft[0] + 1.2) / 2, -1.6, (mz + shaft[1]) / 2 + 0.3, abs(mx - 1.4 - shaft[0]) - 0.4, 0.12, 2.4)
    wh = MeshBuilder()  # wellhead and stuffing box
    wh.frustum(0, 0, 0.0, 1.6, 0.7, 0.7, n=22)
    wh.frustum(0, 0, 1.6, 0.4, 0.95, 0.95, n=22)
    wh.frustum(0, 0, 2.0, 1.4, 0.45, 0.45, n=18)  # tubing head / tee
    wh.frustum(0, 0, 3.4, 1.3, 0.38, 0.38, n=18)  # stuffing box
    wh.frustum(0, 0, 4.4, 0.2, 0.55, 0.55, n=18)
    wh.cone_between((0.0, 0, 2.7), (-9.0, 0, 2.7), 0.22, 0.22, n_cyl)  # flowline
    for xx in (-1.8, -4.5):
        wh.frustum(xx, 0, 2.7, 0.9, 0.5, 0.5, axis="x", n=14)  # valves
    tank = MeshBuilder()
    tank.frustum(-14.0, 11.0, 0.0, 9.0, 3.6, 3.6, n=int(28 * q))
    tank.frustum(-14.0, 11.0, 9.0, 0.5, 3.0, 0.6, n=int(28 * q))
    wh.cone_between((-9.0, 0, 2.7), (-9.0, 11.0, 2.7), 0.2, 0.2, n_cyl)
    wh.cone_between((-9.0, 11.0, 2.7), (-11.0, 11.0, 2.7), 0.2, 0.2, n_cyl)
    sh = MeshBuilder()
    sh.box(px + 8.0, 0, 0.12, 42.0, 14.0, 0.05)  # soft shadow patch on the pad
    traces = [
        _mesh(*[np.asarray(a) for a in (gnd.x, gnd.y, gnd.z, gnd.i, gnd.j, gnd.k)], "#cfc6ae", "Ground"),
        _mesh(*[np.asarray(a) for a in (pad.x, pad.y, pad.z, pad.i, pad.j, pad.k)], "#a39a86", "Gravel pad"),
        _mesh(*[np.asarray(a) for a in (sh.x, sh.y, sh.z, sh.i, sh.j, sh.k)], "#3b3a36", "Shadow", 0.16),
        _mesh(*[np.asarray(a) for a in (base.x, base.y, base.z, base.i, base.j, base.k)], "#4b5563", "Skid"),
        _mesh(*[np.asarray(a) for a in (steel.x, steel.y, steel.z, steel.i, steel.j, steel.k)], "#2b6cb0", "Samson post and gearbox"),
        _mesh(*[np.asarray(a) for a in (motor.x, motor.y, motor.z, motor.i, motor.j, motor.k)], "#1f7a4d", "Motor"),
        _mesh(*[np.asarray(a) for a in (brk.x, brk.y, brk.z, brk.i, brk.j, brk.k)], "#374151", "Brake"),
        _mesh(*[np.asarray(a) for a in (guard.x, guard.y, guard.z, guard.i, guard.j, guard.k)], "#f59e0b", "Belt guard", 0.22),
        _mesh(*[np.asarray(a) for a in (wh.x, wh.y, wh.z, wh.i, wh.j, wh.k)], "#15803d", "Wellhead and flowline"),
        _mesh(*[np.asarray(a) for a in (tank.x, tank.y, tank.z, tank.i, tank.j, tank.k)], "#9ca3af", "Oil tank"),
    ]
    # V-belts (lines)
    traces.append(_line3d([(mx - 1.4, -1.35, mz + 0.9), (shaft[0] + 1.2, -1.35, shaft[1] + 1.3)], "#111827", 3, "V-belt"))
    traces.append(_line3d([(mx - 1.4, -1.35, mz - 0.5), (shaft[0] + 1.2, -1.35, shaft[1] - 1.1)], "#111827", 3, "V-belt"))

    # ---------------- animated parts ----------------
    def pose(k):
        b = float(beta[k])
        ph = float(phi_k[k])
        front = np.array([px - I * np.cos(b), pz - I * np.sin(b)])
        rear = np.array([px + P * np.cos(b), pz + P * np.sin(b)])
        pin = shaft + r_ft * np.array([np.sin(ph), np.cos(ph)])
        d = np.array([np.sin(ph), np.cos(ph)])
        car_z = z_car0 + float(y_ft[k])
        beam = [("box", front, rear, 1.0, -0.55, 0.55), ("box", front + [0, 0.0], rear, 1.5, -0.12, 0.12)]
        th_f = np.pi + b
        horse = [("arc", px, pz, I - 0.55, I + 0.1, th_f - 0.72, th_f + 0.30, -0.8, 0.8, 16)]
        pit = [("box", rear, pin, 0.4, y0, y0 + 0.35) for y0 in (-1.55, 1.2)]
        eq = [("box", rear + [0, 0.0], rear + [0, 0.01], 0.6, -1.6, 1.6)]
        crank = [("box", shaft - 0.9 * d * r_ft / r_ft * 1.0, pin, 0.65, y0, y0 + 0.4) for y0 in (-1.4, 1.0)]
        cwd = -d
        cw = [("box", shaft + cwd * 0.7, shaft + cwd * 2.5, 2.7, y0, y0 + 0.9) for y0 in (-1.55, 0.65)]
        carrier = [("box", np.array([0.0, car_z]), np.array([0.0, car_z + 0.35]), 1.3, -0.5, 0.5)]
        return dict(front=front, rear=rear, car_z=car_z, steel_b=beam + horse + pit + eq + crank, cw=cw, carrier=carrier)

    p0 = pose(0)
    mA = _pack_items(p0["steel_b"])
    mB = _pack_items(p0["cw"])
    mC = _pack_items(p0["carrier"])
    base_idx = len(traces)
    traces += [_mesh(*mA, "#1d4ed8", "Walking beam, horsehead, pitmans, cranks"), _mesh(*mB, "#b91c1c", "Counterweights"), _mesh(*mC, "#f59e0b", "Carrier bar")]
    # cable bridle + polished rod
    def lines_for(k):
        pk = pose(k)
        cz = pk["car_z"]
        return [(0.0, pz, cz + 0.35), ]
    car_z0 = p0["car_z"]
    cab = go.Scatter3d(x=[0, 0, None, 0, 0], y=[-0.45, -0.45, None, 0.45, 0.45], z=[pz, car_z0 + 0.35, None, pz, car_z0 + 0.35], mode="lines",
                       line=dict(color="#111827", width=3), hoverinfo="skip", showlegend=False)
    pol = go.Scatter3d(x=[0, 0], y=[0, 0], z=[car_z0, 4.4], mode="lines", line=dict(color="#d1d5db", width=7), hoverinfo="skip", showlegend=False)
    traces += [cab, pol]

    # ---------------- downhole inset (second scene) ----------------
    ts = res["ts"]
    plg = np.interp(np.linspace(0, 1, n_frames, endpoint=False), np.linspace(0, 1, len(ts), endpoint=False), ts.plunger_in.values)
    plg = (plg - plg.min()) / max(plg.max() - plg.min(), 1e-9)
    up_phase = np.gradient(plg) > 0  # plunger moving up -> TV closed, SV open
    inf = res["inflow"]
    L_lvl = max(0.0, min(1.0, 1.0 - inf["pi"] / max(p["pr"], 1.0)))  # schematic fluid-level depth fraction from intake pressure
    z_lvl = -10.0 - 62.0 * L_lvl
    dh = []
    cas = MeshBuilder()
    cas.tube(-106.0, 3.0, 4.6, 4.9, n=int(32 * q))
    traces2_static = [cas.trace("Casing", "#94a3b8", 0.22)]
    tub = MeshBuilder()
    tub.tube(-82.0, 3.0, 2.1, 2.35, n=int(26 * q))
    traces2_static.append(tub.trace("Tubing", "#64748b", 0.45))
    oil = MeshBuilder()
    oil.frustum(0, 0, -82.0, 85.0, 2.05, 2.05, n=int(22 * q))
    traces2_static.append(oil.trace("Produced oil in tubing", "#b45309", 0.28))
    ann = MeshBuilder()
    ann.tube(-106.0, z_lvl, 2.4, 4.55, n=int(32 * q))
    traces2_static.append(ann.trace("Annulus fluid", "#38bdf8", 0.30))
    bar = MeshBuilder()
    bar.tube(-95.0, -81.0, 1.55, 1.95, n=int(28 * q))
    traces2_static.append(bar.trace("Pump barrel", "#475569", 0.85))
    tail = MeshBuilder()
    tail.tube(-104.0, -95.0, 0.55, 0.85, n=int(18 * q))
    traces2_static.append(tail.trace("Tail pipe / gas anchor", "#78716c", 0.9))
    sv = MeshBuilder()
    sv.frustum(0, 0, -95.8, 0.8, 1.6, 1.6, n=int(24 * q))
    traces2_static.append(sv.trace("Standing-valve cage", "#334155", 0.95))
    perf_z = np.linspace(-103.5, -105.5, 4)
    traces2_static.append(go.Scatter3d(x=[4.8] * 4 + [-4.8] * 4, y=[0] * 8, z=list(perf_z) * 2, mode="markers",
                                       marker=dict(size=4, color="#f97316", symbol="diamond"), hoverinfo="skip", showlegend=False))
    n0 = len(traces) + len(traces2_static)
    pl0 = -92.0 + 3.0 * plg[0]
    plunger = MeshBuilder()
    plunger.frustum(0, 0, 0, 8.0, 1.35, 1.35, n=int(26 * q))
    px_, py_, pz_ = np.asarray(plunger.x), np.asarray(plunger.y), np.asarray(plunger.z)
    pi_, pj_, pk_ = plunger.i, plunger.j, plunger.k
    plunger_tr = _mesh(px_, py_, pz_ + pl0, pi_, pj_, pk_, "#0e7490", "Plunger")
    rod = go.Scatter3d(x=[0, 0], y=[0, 0], z=[pl0 + 8.0, 3.0 + 3.0 * (plg[0] - 0.5)], mode="lines", line=dict(color="#d1d5db", width=6), hoverinfo="skip", showlegend=False)
    cz = np.arange(-72.0, 3.0, 5.0)
    coup = go.Scatter3d(x=[0] * len(cz), y=[0] * len(cz), z=list(cz), mode="markers", marker=dict(size=3.5, color="#6b7280", symbol="square"), hoverinfo="skip", showlegend=False)
    def valve_traces(k):
        up = bool(up_phase[k])
        pz_k = -92.0 + 3.0 * plg[k]
        tv = go.Scatter3d(x=[0], y=[0], z=[pz_k - 0.2 + (0.0 if up else 1.4)], mode="markers", marker=dict(size=9, color="#dc2626" if up else "#16a34a"), hoverinfo="skip", showlegend=False)
        svb = go.Scatter3d(x=[0], y=[0], z=[-96.4 + (1.4 if up else 0.0)], mode="markers", marker=dict(size=9, color="#16a34a" if up else "#dc2626"), hoverinfo="skip", showlegend=False)
        return tv, svb
    tv0, sv0 = valve_traces(0)
    dyn2 = [plunger_tr, rod, coup, tv0, sv0]

    from plotly.subplots import make_subplots as _ms
    fig = _ms(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], column_widths=[0.64, 0.36], horizontal_spacing=0.01)
    for t_ in traces:
        fig.add_trace(t_, row=1, col=1)
    for t_ in traces2_static + dyn2:
        fig.add_trace(t_, row=1, col=2)
    n_main = len(traces)
    idx_dyn_main = [base_idx, base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4]
    n_static2 = len(traces2_static)
    idx_dyn2 = [n_main + n_static2 + j for j in range(5)]

    frames = []
    for k in range(n_frames):
        pk = pose(k)
        a = _pack_items(pk["steel_b"]); b_ = _pack_items(pk["cw"]); c_ = _pack_items(pk["carrier"])
        cz_ = pk["car_z"]
        cab_k = dict(z=[pz, cz_ + 0.35, None, pz, cz_ + 0.35])
        pol_k = dict(z=[cz_, 4.4])
        pl_k = -92.0 + 3.0 * plg[k]
        tv, svb = valve_traces(k)
        shift = 3.0 * (plg[k] - 0.5)
        data = [go.Mesh3d(x=a[0], y=a[1], z=a[2]), go.Mesh3d(x=b_[0], y=b_[1], z=b_[2]), go.Mesh3d(x=c_[0], y=c_[1], z=c_[2]),
                go.Scatter3d(z=cab_k["z"]), go.Scatter3d(z=pol_k["z"]),
                go.Mesh3d(x=px_, y=py_, z=pz_ + pl_k), go.Scatter3d(z=[pl_k + 8.0, 3.0 + shift]),
                go.Scatter3d(z=list(cz + shift)), tv, svb]
        frames.append(go.Frame(data=data, traces=idx_dyn_main + idx_dyn2, name=str(k)))
    fig.frames = frames
    cams = dict(Side=dict(eye=dict(x=0.12, y=-1.35, z=0.38), center=dict(x=0.02, y=0, z=-0.02)),
                Front=dict(eye=dict(x=-1.7, y=-0.4, z=0.4), center=dict(x=0.0, y=0, z=-0.05)),
                Iso=dict(eye=dict(x=1.25, y=-1.35, z=0.75), center=dict(x=0.0, y=0, z=-0.08)))
    hidden = dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    ann1 = [dict(x=x, y=y, z=z, text=t, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#64748b", ax=ax, ay=ay, font=dict(size=11, color=INK),
                 bgcolor="rgba(255,255,255,0.88)", bordercolor=EDGE, borderwidth=1, borderpad=2)
            for (t, x, y, z, ax, ay) in [
                ("Horsehead", -I + 0.3, 0.0, pz - 0.5, -30, -45), ("Walking beam", px - 2.0, 0.0, pz + 1.0, -10, -50),
                ("Samson post", px - 1.5, 2.0, 11.0, -60, 10), ("Counterweights", shaft[0], 1.6, shaft[1] - 1.8, 30, 75),
                ("Gearbox", shaft[0] + 1.8, 1.2, shaft[1] - 0.6, 75, -10), ("Pitman arms", px + P - 0.4, 1.3, pz - 5.0, 70, -45),
                ("Electric motor", mx, 0.0, mz + 1.2, 30, 55), ("Polished rod", 0.0, 0.0, 5.0, -70, 0),
                ("Wellhead / stuffing box", 0.0, 0.0, 3.4, -80, 30), ("Flowline to tank", -9.0, 0.0, 2.7, -40, -40), ("Oil tank", -14.0, 11.0, 9.5, 0, -30)]]
    ann2 = [dict(x=x, y=0, z=z, text=t, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#64748b", ax=ax, ay=ay, font=dict(size=11, color=INK),
                 bgcolor="rgba(255,255,255,0.88)", bordercolor=EDGE, borderwidth=1, borderpad=2)
            for (t, x, z, ax, ay) in [
                ("Casing", 4.8, -20.0, 45, 0), ("Tubing (oil flows up)", 2.3, -40.0, 60, -10), ("Sucker-rod string", 0.0, -60.0, -75, 0),
                ("Pump barrel", 1.9, -88.0, 70, -20), ("Plunger", 1.4, pl0 + 4.0, -60, -30), ("Travelling valve", 0.0, pl0 - 0.2, -75, 30),
                ("Standing valve", 0.0, -96.4, 70, 30), ("Perforations", 4.8, -104.5, 60, 10), ("Annulus liquid level", 4.5, z_lvl, 60, -10)]]
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=6, b=24), paper_bgcolor="rgba(0,0,0,0)", showlegend=False, uirevision=f"srp-{camera}",
        scene=dict(aspectmode="data", bgcolor="#dfe8ee", xaxis=hidden, yaxis=hidden, zaxis=hidden, camera=dict(**cams[camera], up=dict(x=0, y=0, z=1)), annotations=ann1, dragmode="orbit"),
        scene2=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=3.2), bgcolor="#eef1f4", xaxis=hidden, yaxis=hidden, zaxis=hidden,
                    camera=dict(eye=dict(x=3.4, y=-4.2, z=0.05), center=dict(x=0, y=0, z=0.0), up=dict(x=0, y=0, z=1)), annotations=ann2, dragmode="orbit"))
    dur = int(np.clip(60000.0 / p["spm"] / n_frames, 30, 400))
    fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor=PANEL, font=dict(color=INK),
                      buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                               dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])])
    fig.add_annotation(text="Left: surface unit. Right: downhole pump (not to scale; plunger stroke exaggerated). Green valve = open, red = closed.",
                       xref="paper", yref="paper", x=0.0, y=0.0, yanchor="top", showarrow=False, font=dict(size=11, color=MUTED))
    return fig


@st.cache_data(show_spinner=False, max_entries=12)
def cached_srp_figure(frozen, camera: str, gfx: str):
    p = dict(frozen)
    return srp_unit_figure(srp_simulate(p), {**SRP_DEFAULTS, **p}, camera=camera)


def page_srp():
    header("SRP Lab", "Sucker-rod pump: four-bar kinematics, damped wave-equation rod string, dynamometer cards, torque, loads and Vogel inflow - educational, not validated.")
    sp = dict(SRP_DEFAULTS)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slider("Pump depth (ft)", "w_srp_depth", 2000.0, 12000.0, 250.0, 6000.0)
        slider("Plunger diameter (in)", "w_srp_pd", 0.75, 3.25, 0.25, 1.75)
    with c2:
        slider("Surface stroke length (in)", "w_srp_stroke", 30.0, 168.0, 2.0, 64.0)
        slider("Pumping speed (SPM)", "w_srp_spm", 1.0, 12.0, 0.5, 6.0)
    with c3:
        slider("Reservoir pressure (psi)", "w_srp_pr", 500.0, 4000.0, 50.0, 1800.0)
        slider("Vogel qmax (bpd)", "w_srp_qmax", 20.0, 1500.0, 10.0, 600.0)
    with c4:
        slider("Tubing head pressure (psi)", "w_srp_pt", 0.0, 500.0, 10.0, 100.0)
        slider("Fluid specific gravity", "w_srp_sg", 0.7, 1.1, 0.01, 0.85)
    with st.expander("Rod string, damping, slippage and unit"):
        e1, e2, e3 = st.columns(3)
        with e1:
            slider("Rod taper: 1 in share (%)", "w_srp_ft", 0.0, 100.0, 5.0, 33.0)
            slider("Rod taper: 7/8 in share (%)", "w_srp_fm", 0.0, 100.0, 5.0, 34.0)
            slider("Rod taper: 3/4 in share (%)", "w_srp_fb", 0.0, 100.0, 5.0, 33.0)
        with e2:
            slider("Damping factor (0-0.6)", "w_srp_damp", 0.0, 0.6, 0.05, 0.15)
            slider("Plunger slippage (fraction)", "w_srp_slip", 0.0, 0.3, 0.01, 0.03)
            slider("Minimum pump intake pressure (psi)", "w_srp_pimin", 20.0, 500.0, 10.0, 60.0)
        with e3:
            st.selectbox("Pumping unit (generic sizes)", list(UNIT_SIZES), index=1, key="w_srp_unit")
            slider("Counterbalance effect (lbf, 0 = auto-balance)", "w_srp_cbe", 0.0, 40000.0, 250.0, 0.0)
    ss = st.session_state
    p = dict(depth=float(ss.w_srp_depth), pump_d=float(ss.w_srp_pd), stroke=float(ss.w_srp_stroke), spm=float(ss.w_srp_spm),
             sg=float(ss.w_srp_sg), pt=float(ss.w_srp_pt), pr=float(ss.w_srp_pr), qmax=float(ss.w_srp_qmax),
             pi_min=float(ss.w_srp_pimin), slip=float(ss.w_srp_slip), damp=float(ss.w_srp_damp),
             f_top=float(ss.w_srp_ft), f_mid=float(ss.w_srp_fm), f_bot=float(ss.w_srp_fb),
             cbe=(float(ss.w_srp_cbe) if ss.w_srp_cbe > 0 else -1.0), unit=str(ss.w_srp_unit))
    errs = validate_srp({**SRP_DEFAULTS, **p})
    if errs:
        for e in errs:
            st.error(e)
        return
    res = srp_simulate(p)
    inf = res["inflow"]
    m = st.columns(6)
    m[0].metric("Production", f"{inf['q_prod']:.0f} bpd", f"pump capacity {inf['q_cap']:.0f} bpd", delta_color="off")
    m[1].metric("Pump fill", f"{inf['fill'] * 100:.0f} %", "PUMP-OFF" if inf["pump_off"] else "full", delta_color="off")
    m[2].metric("Peak / min rod load", f"{res['f_max'] / 1000:.1f} / {res['f_min'] / 1000:.1f} klbf")
    m[3].metric("Peak torque", f"{res['torque_peak_kinlb']:.0f} kin-lbf", f"{res['gearbox_pct']:.0f} % of rating", delta_color="off")
    m[4].metric("Polished-rod HP", f"{res['prhp']:.1f} hp", f"hydraulic {res['hhp']:.1f} hp", delta_color="off")
    m[5].metric("Goodman use", f"{res['goodman_pct']:.0f} %", f"max stress {res['s_max_psi'] / 1000:.1f} ksi", delta_color="off")
    if inf["pump_off"]:
        st.warning("Pump-off: the pump capacity exceeds inflow at the minimum intake pressure, so the barrel does not fill. Slow the unit or reduce the plunger/stroke.")
    if res["gearbox_pct"] > 100 or res["structure_pct"] > 100 or res["goodman_pct"] > 100:
        st.error("An equipment or rod rating is exceeded in this configuration (gearbox, structure or Goodman allowable).")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        cam = st.radio("Camera", ["Side", "Front", "Iso"], horizontal=True, key="w_srp_cam")
        show(cached_srp_figure(freeze(p), cam, gfx_name()), key="srp_3d")
    with right:
        c = res["card"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c.pos_in, y=c.load_lbf, name="Surface card", line=dict(color=COOL, width=2)))
        pc = res["pump_card"]
        fig.add_trace(go.Scatter(x=pc.pos_in, y=pc.load_lbf, name="Pump (plunger) card", line=dict(color=WARM, width=2)))
        fig.add_hline(y=res["rod_weight_buoyed"], line_dash="dot", line_color=MUTED, annotation_text="buoyed rod weight")
        fig.update_layout(title="Dynamometer cards", xaxis_title="Position (in)", yaxis_title="Load (lbf)", legend=dict(orientation="h", y=-0.22))
        show(style_fig(fig, 360), key="srp_card")
        ts = res["ts"]
        f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Gearbox torque (kin-lbf)", "Torque factor (in)"))
        f2.add_trace(go.Scatter(x=ts.t_s, y=ts.torque_kinlb, line=dict(color=WARM), showlegend=False), 1, 1)
        f2.add_trace(go.Scatter(x=ts.t_s, y=ts.tf_in_rad, line=dict(color=COOL), showlegend=False), 2, 1)
        f2.add_hline(y=res["torque_rating"], row=1, col=1, line_dash="dot", line_color=RED)
        f2.update_xaxes(title_text="Time in one stroke (s)", row=2, col=1)
        show(style_fig(f2, 300), key="srp_tq")
    t1, t2, t3 = st.tabs(["Inflow and pump", "Rod string", "Model notes"])
    with t1:
        ipr = np.linspace(0, p["pr"], 60)
        fig = go.Figure(go.Scatter(x=vogel_q(ipr, p["pr"], p["qmax"]), y=ipr, name="Vogel IPR", line=dict(color=COOL)))
        fig.add_trace(go.Scatter(x=[inf["q_prod"]], y=[inf["pi"]], mode="markers", marker=dict(size=11, color=WARM), name="Operating point"))
        fig.update_layout(xaxis_title="Rate (bpd)", yaxis_title="Flowing bottom-hole pressure (psi)", legend=dict(orientation="h", y=-0.25))
        show(style_fig(fig, 320), key="srp_ipr")
        st.caption(f"API theoretical displacement = 0.1166 x Ap x S x N = {inf['q_theor']:.0f} bpd; net of slippage {inf['q_cap']:.0f} bpd. "
                   f"Fluid load = Ap x (P_discharge - P_intake) = {res['fo']:.0f} lbf; net lift {res['lnet']:,.0f} ft. Pump assumed set at the perforations (intake = flowing bottom-hole pressure).")
    with t2:
        st.dataframe(pd.DataFrame([dict(rod_in=s[0], length_ft=round(s[1]), weight_lb_ft=s[2], area_in2=s[3]) for s in res["secs"]]), hide_index=True)
        st.caption(f"Buoyed rod weight {res['rod_weight_buoyed']:,.0f} lbf. Peak surface load {res['f_max']:,.0f} lbf "
                   f"({res['structure_pct']:.0f} % of the unit structure rating). Plunger stroke {res['plunger_stroke_in']:.1f} in vs surface {p['stroke']:.0f} in "
                   f"(difference = rod stretch and overtravel). Counterbalance {res['cbe']:,.0f} lbf. Goodman allowable {res['s_allow_psi'] / 1000:.1f} ksi "
                   "(T = 125 ksi, SF 0.9, illustrative).")
    with t3:
        st.markdown(
            "- **Kinematics:** crank-rocker four-bar (crank, pitman, rear beam arm); polished-rod position from the beam angle; torque factor TF = dy/d(phi); torque = TF x (F - CBE).\n"
            "- **Rod string:** 1-D damped wave equation (Gibbs), finite differences, E = 30.5e6 psi, a ~ 16,300 ft/s, CFL time step, mass-proportional damping; last of several cycles reported.\n"
            "- **Valves:** the load transfers at the polished-rod turn-arounds over a short ramp; the gas-compression downstroke uses an isothermal barrel.\n"
            "- **Limits:** no tubing movement, no deviation, no fluid inertia, no friction model beyond damping. Unit ratings are generic. Not for design or sizing decisions.")
    st.download_button("Download SRP card data (CSV)", res["ts"].to_csv(index=False).encode(), "srp_cycle.csv", "text/csv", key="srp_csv")


def page_welldesign():
    header("Well Path & Torque-Drag", "Minimum-curvature trajectory with a soft-string torque & drag model and buckling screen - educational, not validated.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slider("Kick-off point (ft MD)", "w_td_kop", 500.0, 8000.0, 100.0, 2000.0)
        slider("Build rate (deg/100 ft)", "w_td_bur", 0.5, 8.0, 0.5, 3.0)
    with c2:
        slider("Hold inclination (deg)", "w_td_inc", 0.0, 90.0, 1.0, 60.0)
        slider("Azimuth (deg)", "w_td_azi", 0.0, 359.0, 1.0, 45.0)
    with c3:
        slider("Total depth (ft MD)", "w_td_td", 4000.0, 20000.0, 250.0, 10000.0)
        slider("Friction factor", "w_td_mu", 0.05, 0.6, 0.01, 0.25)
    with c4:
        slider("Mud weight (ppg)", "w_td_mw", 7.0, 18.0, 0.1, 10.5)
        slider("WOB for drilling case (klbf)", "w_td_wob", 0.0, 50.0, 1.0, 20.0)
    d1, d2 = st.columns(2)
    d1.selectbox("Drill pipe", list(PIPE_TABLE), index=1, key="w_td_pipe")
    with d2:
        slider("Hole diameter (in)", "w_td_hole", 6.0, 12.25, 0.125, 8.5)
    ss = st.session_state
    p = dict(kop=float(ss.w_td_kop), build_rate=float(ss.w_td_bur), hold_inc=float(ss.w_td_inc), azi=float(ss.w_td_azi), td_md=float(ss.w_td_td),
             mu=float(ss.w_td_mu), pipe=str(ss.w_td_pipe), hole_in=float(ss.w_td_hole))
    errs = validate_td(p)
    if errs:
        for e in errs:
            st.error(e)
        return
    path, res = cached_td(freeze({**p, "mw": float(ss.w_td_mw), "wob": float(ss.w_td_wob)}))
    hk = res["hookload"]
    m = st.columns(6)
    m[0].metric("Pick-up hookload", f"{hk['pickup'] / 1000:.0f} klbf")
    m[1].metric("Slack-off hookload", f"{hk['slackoff'] / 1000:.0f} klbf")
    m[2].metric("Rotating (off bottom)", f"{hk['rotating'] / 1000:.0f} klbf")
    m[3].metric("Surface torque, rotating", f"{res['surface_torque']['rotating'] / 1000:.1f} kft-lbf")
    m[4].metric("Surface torque, drilling", f"{res['surface_torque']['drilling'] / 1000:.1f} kft-lbf")
    m[5].metric("TVD / displacement", f"{path.tvd.iloc[-1]:,.0f} / {np.hypot(path.north.iloc[-1], path.east.iloc[-1]):,.0f} ft")
    if res["helical"]["slack-off"] or res["helical"]["drilling"]:
        st.error("Helical buckling predicted (compression exceeds 2.83 x the sinusoidal threshold). Lock-up risk.")
    elif res["sinusoidal"]["slack-off"] or res["sinusoidal"]["drilling"]:
        st.warning("Sinusoidal buckling predicted somewhere in the string (compression exceeds the sinusoidal threshold).")
    else:
        st.success("No buckling predicted for slack-off or drilling in this soft-string screen.")
    left, right = st.columns([2, 3], gap="medium")
    with left:
        mode = st.radio("Colour well path by", ["pickup", "slackoff", "drilling"], horizontal=True, key="w_td_mode")
        t = res["tension"][mode]
        fig = go.Figure(go.Scatter3d(x=path.east, y=path.north, z=-path.tvd, mode="lines",
                                     line=dict(width=7, color=t / 1000.0, colorscale="RdBu_r", colorbar=dict(title="klbf", len=0.6)),
                                     hovertemplate="MD %{customdata:.0f} ft<extra></extra>", customdata=path.md))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=6, color=INK), hoverinfo="skip"))
        apply_scene(fig, dict(eye=dict(x=1.5, y=-1.6, z=0.6), center=dict(x=0, y=0, z=-0.05), up=dict(x=0, y=0, z=1)), 520, uirev="td",
                    hint="Axis scale equal; colour = string tension (red high, blue compression).")
        show(fig, key="td_3d")
    with right:
        f = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=("Tension (klbf)", "Torque (kft-lbf)", "Inclination (deg) / DLS (deg/100 ft)"))
        for nm, col in (("pickup", WARM), ("slackoff", COOL), ("rotating", GREEN), ("drilling", INK)):
            f.add_trace(go.Scatter(x=path.md, y=res["tension"][nm] / 1000.0, name=nm, line=dict(color=col)), 1, 1)
        for nm, col in (("rotating", GREEN), ("drilling", INK)):
            f.add_trace(go.Scatter(x=path.md, y=res["torque"][nm] / 1000.0, name=nm, line=dict(color=col), showlegend=False), 2, 1)
        f.add_trace(go.Scatter(x=path.md, y=path.inc_deg, name="inclination", line=dict(color=COOL)), 3, 1)
        f.add_trace(go.Scatter(x=path.md, y=path.dls, name="DLS", line=dict(color=WARM)), 3, 1)
        f.update_xaxes(title_text="Measured depth (ft)", row=3, col=1)
        f.update_layout(legend=dict(orientation="h", y=1.08))
        show(style_fig(f, 620), key="td_ts")
    st.download_button("Download trajectory + tensions (CSV)", pd.DataFrame(dict(
        md=path.md, inc_deg=path.inc_deg, tvd=path.tvd, north=path.north, east=path.east, dls=path.dls,
        t_pickup=res["tension"]["pickup"], t_slackoff=res["tension"]["slackoff"], t_rotating=res["tension"]["rotating"],
        t_drilling=res["tension"]["drilling"])).to_csv(index=False).encode(), "well_td.csv", "text/csv", key="td_csv")
    st.caption("Johancsik soft string: N = sqrt((T dphi sin(theta))^2 + (T dtheta + w sin(theta) dL)^2), dT = w cos(theta) dL +/- mu N; torque = mu N r. "
               "No stiffness, tortuosity, hydraulic or rotational effects; buckling uses Lubinski / Dawson-Paslay-type screening thresholds (sinusoidal Fs = 2 sqrt(E I w sin(theta) / r), helical 2.83 Fs).")


@st.cache_data(show_spinner=False, max_entries=32)
def cached_td(frozen):
    p = dict(frozen)
    path = well_path(p["kop"], p["build_rate"], p["hold_inc"], p["azi"], p["td_md"])
    bt = float(bit_torque_ftlbf(p["wob"], p["hole_in"], 0.0))
    return path, torque_drag(path, p["mu"], p["mw"], p["pipe"], p["hole_in"], wob=p["wob"], bit_torque=bt)


def page_wellcontrol():
    header("Well Control", "Shut-in data to kill-mud weight, circulating pressures, MAASP and influx type (driller's-method kill sheet) - educational calculator.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slider("Mud weight (ppg)", "w_wc_mw", 7.0, 18.0, 0.1, 10.0)
        slider("Well TVD (ft)", "w_wc_tvd", 3000.0, 20000.0, 100.0, 10000.0)
    with c2:
        slider("SIDPP (psi)", "w_wc_sidpp", 0.0, 1500.0, 10.0, 520.0)
        slider("SICP (psi)", "w_wc_sicp", 0.0, 2500.0, 10.0, 700.0)
    with c3:
        slider("Pit gain (bbl)", "w_wc_gain", 1.0, 100.0, 1.0, 20.0)
        slider("Slow-circulating-rate pressure (psi)", "w_wc_scr", 100.0, 2000.0, 10.0, 600.0)
    with c4:
        slider("Kill rate (SPM)", "w_wc_spm", 10.0, 60.0, 1.0, 30.0)
        slider("Pump output (bbl/stroke)", "w_wc_out", 0.05, 0.25, 0.005, 0.1, fmt="%.3f")
    with st.expander("Geometry and casing shoe"):
        e1, e2, e3 = st.columns(3)
        with e1:
            slider("Drill-string volume (bbl)", "w_wc_dpv", 50.0, 400.0, 5.0, 150.0)
            slider("Annulus volume (bbl)", "w_wc_annv", 100.0, 1500.0, 10.0, 500.0)
        with e2:
            slider("Casing-shoe TVD (ft)", "w_wc_shoe", 1000.0, 15000.0, 100.0, 6000.0)
            slider("Shoe fracture gradient (ppg EMW)", "w_wc_fg", 9.0, 20.0, 0.1, 14.0)
        with e3:
            slider("Hole diameter (in)", "w_wc_hole", 4.0, 17.5, 0.125, 8.5)
            slider("Drill-pipe OD (in)", "w_wc_pod", 2.0, 8.0, 0.125, 5.0)
    ss = st.session_state
    if ss.w_wc_hole <= ss.w_wc_pod:
        st.error("Hole diameter must exceed the pipe OD.")
        return
    if ss.w_wc_sicp < ss.w_wc_sidpp:
        st.warning("SICP is below SIDPP; with an influx in the annulus SICP is normally >= SIDPP. Check the readings (trapped pressure / float valve).")
    k = kill_sheet(ss.w_wc_mw, ss.w_wc_tvd, ss.w_wc_sidpp, ss.w_wc_sicp, ss.w_wc_gain, ss.w_wc_scr, ss.w_wc_spm, ss.w_wc_dpv, ss.w_wc_annv,
                   ss.w_wc_out, ss.w_wc_shoe, ss.w_wc_fg, ss.w_wc_hole, ss.w_wc_pod)
    m = st.columns(6)
    m[0].metric("Kill mud weight", f"{k['kmw']:.2f} ppg", f"+{k['kmw'] - ss.w_wc_mw:.2f} ppg", delta_color="off")
    m[1].metric("Initial circ. pressure", f"{k['icp']:.0f} psi")
    m[2].metric("Final circ. pressure", f"{k['fcp']:.0f} psi")
    m[3].metric("MAASP", f"{k['maasp']:.0f} psi")
    m[4].metric("Influx", k["influx_type"], f"{k['influx_height_ft']:.0f} ft, {k['influx_grad']:.3f} psi/ft" if not np.isnan(k["influx_grad"]) else "", delta_color="off")
    m[5].metric("Time to bit / to surface", f"{k['time_to_bit_min']:.0f} / {k['time_surface_min']:.0f} min")
    if k["sicp_over_maasp"]:
        st.error("SICP exceeds MAASP: the shoe may fracture before the kill starts. Treat as a critical well-control problem.")
    if k["kmw_over_fg"]:
        st.error("Kill mud weight exceeds the shoe fracture gradient.")
    fig = go.Figure(go.Scatter(x=k["schedule"].strokes, y=k["schedule"].dp_pressure_psi, mode="lines+markers", line=dict(color=COOL, width=3)))
    fig.update_layout(title="Drill-pipe pressure schedule while pumping kill mud to the bit", xaxis_title="Strokes", yaxis_title="Drill-pipe pressure (psi)")
    show(style_fig(fig, 340), key="wc_sched")
    st.dataframe(k["schedule"].round(0), hide_index=True)
    st.caption("KMW = MW + SIDPP/(0.052 TVD); ICP = SIDPP + SCR; FCP = SCR x KMW/MW; MAASP = (FG - MW) x 0.052 x TVD_shoe; influx height = pit gain / annular capacity; "
               "influx gradient = 0.052 MW - (SICP - SIDPP)/h. Ignores gas migration, temperature, choke-line friction and hole geometry changes. A training aid, not a substitute for well-control procedures.")


def esp_figure(res: dict, n_frames: int = 30, height: int = 600):
    """Schematic ESP completion (vertical scale compressed): casing, tubing, pump stages, intake, protector, motor, cable, flow markers."""
    p = res["params"]
    zs = dict(top=0.0, pump_top=-52.0, pump_bot=-76.0, intake=-78.0, prot_top=-80.0, prot_bot=-86.0, mot_top=-86.5, mot_bot=-100.0)
    mb = MeshBuilder()
    mb.tube(-104.0, 4.0, 4.6, 5.0, n=30)  # casing
    cas = mb.trace("Casing", "#94a3b8", 0.25)
    pump = MeshBuilder()
    n_rings = 14
    for k in range(n_rings):
        zt = zs["pump_top"] - (zs["pump_top"] - zs["pump_bot"]) * k / n_rings
        pump.frustum(0, 0, zt - 1.6, 1.6, 2.0, 2.0, n=20)
        pump.frustum(0, 0, zt - 1.8, 0.2, 2.25, 2.25, n=20)  # stage joint ring
    ml = MeshBuilder()
    ml.frustum(0, 0, zs["intake"] - 1.0, 2.0, 2.0, 2.0, n=18)
    prot = MeshBuilder()
    prot.frustum(0, 0, zs["prot_bot"], zs["prot_top"] - zs["prot_bot"], 2.0, 2.0, n=18)
    mot = MeshBuilder()
    mot.frustum(0, 0, zs["mot_bot"], zs["mot_top"] - zs["mot_bot"], 2.0, 2.0, n=20)
    tbg = [_line3d([(0, 0, 4), (0, 0, zs["pump_top"])], "#334155", 8, "Production tubing")]
    cable = _line3d([(2.6, 0, 4), (2.6, 0, zs["mot_top"]), (2.0, 0, zs["mot_top"])], "#f59e0b", 5, "Power cable")
    traces = [cas, pump.trace("Pump stages", "#0e7490"), ml.trace("Intake", "#64748b"), prot.trace("Protector / seal", "#a16207"),
              mot.trace("Motor", "#b91c1c"), cable] + tbg
    base_idx = len(traces)
    path = np.array([(3.2, 0, -96), (3.2, 0, -79), (0, 0, -79), (0, 0, zs["pump_top"]), (0, 0, 4)])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    n_part = 26

    def parts(ph):
        t = ((np.arange(n_part) / n_part + ph) % 1.0) * seg[-1]
        return [np.interp(t, seg, path[:, i]) for i in range(3)]

    px, py, pz = parts(0.0)
    traces.append(go.Scatter3d(x=px, y=py, z=pz, mode="markers", marker=dict(size=4.5, color=COOL), hoverinfo="skip"))
    for txt, (x, z) in {"Wellhead": (7, 2), "Production tubing": (7, -25), "Pump stages": (7, -64), "Intake": (7, -78), "Protector": (7, -83),
                        "Motor": (7, -94), "Cable": (-6, -40)}.items():
        traces.append(go.Scatter3d(x=[x], y=[0], z=[z], mode="text", text=[txt], textposition="middle right" if x > 0 else "middle left",
                                   textfont=dict(color=INK, size=11), hoverinfo="skip"))
    fig = go.Figure(traces)
    apply_scene(fig, dict(eye=dict(x=1.2, y=-1.6, z=0.25), center=dict(x=0, y=0, z=0.0), up=dict(x=0, y=0, z=1)), height, uirev="esp",
                hint="Schematic (vertical scale compressed). Markers show produced fluid entering the intake, passing the stages and rising up the tubing.")
    if res["q"] > 0:
        fig.frames = [go.Frame(data=[go.Scatter3d(x=a, y=b, z=c, mode="markers", marker=dict(size=4.5, color=COOL))], traces=[base_idx], name=str(k))
                      for k, (a, b, c) in enumerate(parts(k / n_frames) for k in range(n_frames))]
        dur = int(np.clip(60000.0 / max(res["q"], 50.0) * 3, 40, 300))
        fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor=PANEL, font=dict(color=INK),
                          buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
                                   dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])])
    return fig


def page_esp():
    header("ESP Lab", "Electric submersible pump: generic normalised pump curves, affinity laws, system curve with Vogel inflow, gas derating, motor and cable sizing - educational, not a vendor catalogue.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slider("Pump setting depth (ft)", "w_esp_depth", 2000.0, 12000.0, 250.0, 6000.0)
        st.selectbox("Pump (generic)", list(ESP_PUMPS), index=1, key="w_esp_pump")
    with c2:
        slider("Stages", "w_esp_stages", 10, 400, 5, 120)
        slider("Frequency (Hz)", "w_esp_freq", 30.0, 70.0, 0.5, 55.0)
    with c3:
        slider("Reservoir pressure (psi)", "w_esp_pr", 500.0, 4500.0, 50.0, 2200.0)
        slider("Vogel qmax (bpd)", "w_esp_qmax", 100.0, 8000.0, 50.0, 2500.0)
    with c4:
        slider("Free-gas fraction at intake (0-1)", "w_esp_gvf", 0.0, 0.5, 0.01, 0.03)
        slider("Tubing head pressure (psi)", "w_esp_pt", 0.0, 500.0, 10.0, 150.0)
    with st.expander("Fluid, tubing, motor and cable"):
        e1, e2, e3 = st.columns(3)
        with e1:
            slider("Fluid specific gravity", "w_esp_sg", 0.7, 1.1, 0.01, 0.9)
            slider("Tubing ID (in)", "w_esp_tid", 1.5, 4.0, 0.001, 2.992, fmt="%.3f")
        with e2:
            slider("Motor voltage (V)", "w_esp_v", 500.0, 4000.0, 50.0, 2000.0)
            st.selectbox("Cable (copper)", list(ESP_CABLES), index=1, key="w_esp_cable")
        with e3:
            slider("Bottom-hole temperature (F)", "w_esp_bht", 100.0, 350.0, 5.0, 200.0)
            slider("Target rate for stage sizing (bpd)", "w_esp_target", 100.0, 6000.0, 50.0, 1500.0)
    ss = st.session_state
    p = dict(depth=float(ss.w_esp_depth), pump=str(ss.w_esp_pump), stages=int(ss.w_esp_stages), freq=float(ss.w_esp_freq), pr=float(ss.w_esp_pr),
             qmax=float(ss.w_esp_qmax), gvf=float(ss.w_esp_gvf), pt=float(ss.w_esp_pt), sg=float(ss.w_esp_sg), tbg_id=float(ss.w_esp_tid),
             motor_v=float(ss.w_esp_v), cable=str(ss.w_esp_cable), bht=float(ss.w_esp_bht))
    errs = validate_esp({**ESP_DEFAULTS, **p})
    if errs:
        for e in errs:
            st.error(e)
        return
    res = cached_esp(freeze(p))
    m = st.columns(6)
    m[0].metric("Production", f"{res['q']:.0f} bpd", f"{res['x_bep'] * 100:.0f} % of BEP rate", delta_color="off")
    m[1].metric("Total head", f"{res['head']:,.0f} ft")
    m[2].metric("Shaft power", f"{res['shaft_hp']:.0f} hp", f"motor {res['motor_hp']} hp, {res['motor_load']:.0f} % load", delta_color="off")
    m[3].metric("Pump stage efficiency", f"{res['eta_stage'] * 100:.0f} %")
    m[4].metric("Motor current", f"{res['amps']:.0f} A", f"cable drop {res['vdrop_pct']:.1f} %", delta_color="off")
    m[5].metric("Surface power", f"{res['kva']:.0f} kVA")
    for fl in res["flags"]:
        st.warning(fl)
    if not res["flags"]:
        st.success("Operating point is inside the recommended range with no flags raised by this screening.")
    left, right = st.columns([2, 3], gap="medium")
    with left:
        show(esp_figure(res), key="esp_3d")
    with right:
        q = np.linspace(20.0, p["qmax"] * 0.999, 120)
        pp = {**ESP_DEFAULTS, **p}
        f = go.Figure()
        f.add_trace(go.Scatter(x=q, y=esp_pump_head(q, pp), name=f"Pump head ({p['freq']:.0f} Hz)", line=dict(color=COOL, width=3)))
        f.add_trace(go.Scatter(x=q, y=esp_pump_head(q, pp, 60.0), name="Pump head (60 Hz)", line=dict(color=COOL, dash="dot")))
        f.add_trace(go.Scatter(x=q, y=esp_system_head(q, pp), name="System head required", line=dict(color=WARM, width=3)))
        qb = ESP_PUMPS[p["pump"]][0] * p["freq"] / 60.0
        f.add_vrect(x0=0.6 * qb, x1=1.3 * qb, fillcolor=GREEN, opacity=0.10, line_width=0, annotation_text="recommended range")
        if res["q"] > 0:
            f.add_trace(go.Scatter(x=[res["q"]], y=[res["head"]], mode="markers", marker=dict(size=12, color=INK), name="Operating point"))
        f.update_layout(title="Pump curve vs system curve", xaxis_title="Rate (bpd)", yaxis_title="Head (ft)", legend=dict(orientation="h", y=-0.25))
        show(style_fig(f, 340), key="esp_curve")
        sw = res["sweep"]
        g = make_subplots(specs=[[{"secondary_y": True}]])
        g.add_trace(go.Scatter(x=sw.freq, y=sw.q, name="Production (bpd)", line=dict(color=COOL)), secondary_y=False)
        g.add_trace(go.Scatter(x=sw.freq, y=sw.shaft_hp, name="Shaft power (hp)", line=dict(color=WARM)), secondary_y=True)
        g.add_vline(x=p["freq"], line_dash="dot", line_color=MUTED)
        g.update_layout(title="Variable-speed drive sweep", xaxis_title="Frequency (Hz)", legend=dict(orientation="h", y=-0.25))
        show(style_fig(g, 300), key="esp_vsd")
    n_req = esp_required_stages(float(ss.w_esp_target), {**ESP_DEFAULTS, **p})
    st.caption(f"Stages needed for {ss.w_esp_target:.0f} bpd at {p['freq']:.0f} Hz: about {n_req}. Pump housing OD {res['od']:.2f} in. "
               "Head per stage follows H/Hbep = 1.3 - 0.3 (q/qbep)^2 scaled by the affinity laws (q ~ f, H ~ f^2, P ~ f^3); power = q H SG / (135,770 eta) (q in bpd, H in ft); "
               "gas derating is an illustrative curve; motor sizing uses 85 % power factor and 85 % motor efficiency; the pump is assumed set at the perforations (intake = flowing bottom-hole pressure). "
               "Not a vendor selection tool.")
    st.download_button("Download frequency sweep (CSV)", res["sweep"].to_csv(index=False).encode(), "esp_sweep.csv", "text/csv", key="esp_csv")


@st.cache_data(show_spinner=False, max_entries=48)
def cached_esp(frozen):
    return esp_simulate(dict(frozen))


OIL_SCALE = [[0.0, "#cfe3f1"], [0.20, "#f1e6c8"], [0.45, "#e0b25a"], [0.70, "#a8651a"], [1.0, "#2a1606"]]  # water -> oil-rich (dark)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_res(frozen):
    return res_simulate(dict(frozen))


def res_structure_figure(res: dict, snap: dict, height: int = 460):
    """3-D structural view: top-reservoir surface coloured by oil saturation, oil-water contact plane, wells."""
    p = res["params"]
    ny, nx = snap["sw"].shape
    xs = np.arange(nx) * p["dx"]
    ys = np.arange(ny) * p["dx"]
    so = 1.0 - snap["sw"]
    z = -res["depth"]
    surf = go.Surface(x=xs, y=ys, z=z, surfacecolor=so, colorscale=OIL_SCALE, cmin=0.0, cmax=1.0 - p["swc"], showscale=True,
                      colorbar=dict(title="Oil sat.", len=0.55, thickness=12), hoverinfo="skip",
                      lighting=dict(ambient=0.6, diffuse=0.8, specular=0.25, roughness=0.6), contours=dict(z=dict(show=True, usecolormap=False, color="#6b7280", width=1, start=float(z.min()), end=float(z.max()), size=max((z.max() - z.min()) / 12, 1.0))))
    owc = go.Surface(x=[xs[0], xs[-1]], y=[ys[0], ys[-1]], z=np.full((2, 2), -res["owc"]), colorscale=[[0, "#38bdf8"], [1, "#38bdf8"]], showscale=False,
                     opacity=0.35, hoverinfo="skip")
    traces = [surf, owc]
    top = float(z.max()) + 500.0
    for nm, ty_, j, i in res["wells"]:
        col = "#16a34a" if ty_ == "producer" else "#2563eb"
        traces.append(go.Scatter3d(x=[xs[i], xs[i]], y=[ys[j], ys[j]], z=[top, float(z[j, i])], mode="lines", line=dict(color=col, width=6), hoverinfo="skip"))
        traces.append(go.Scatter3d(x=[xs[i]], y=[ys[j]], z=[top], mode="markers", marker=dict(size=5, color=col), hoverinfo="skip"))
    fig = go.Figure(traces)
    ann = [dict(x=xs[i], y=ys[j], z=top, text=f"{nm}", showarrow=False, font=dict(size=11, color=INK), bgcolor="rgba(255,255,255,0.88)", bordercolor=EDGE, borderwidth=1, borderpad=2, yshift=12)
           for nm, ty_, j, i in res["wells"]]
    ann.append(dict(x=xs[-1], y=ys[-1], z=-res["owc"], text="Oil-water contact", showarrow=False, font=dict(size=11, color="#075985"), bgcolor="rgba(255,255,255,0.85)", borderpad=2))
    hidden = dict(visible=False, showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=6, b=24), paper_bgcolor="rgba(0,0,0,0)", showlegend=False, uirevision="resstruct",
                      scene=dict(aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=0.6), xaxis=hidden, yaxis=hidden, zaxis=hidden, bgcolor="#eef1f4", annotations=ann,
                                 camera=dict(eye=dict(x=0.95, y=-1.05, z=0.55), center=dict(x=0, y=0, z=-0.15))))
    fig.add_annotation(text="Structure not to scale vertically (exaggerated). Green wells = producers, blue = injectors. Colour = oil saturation.",
                       xref="paper", yref="paper", x=0, y=0, yanchor="top", showarrow=False, font=dict(size=11, color=MUTED))
    return fig


def page_reservoir():
    header("Reservoir Simulation", "Oil reservoir: undersaturated oil with a structure, an oil-water contact and an edge aquifer - primary depletion or peripheral waterflood. Slightly compressible two-phase model; educational, not history-matched.")
    top = st.columns([1.6, 1, 1, 1])
    with top[0]:
        lay = st.selectbox("Development scenario", list(RES_LAYOUTS)[:3], index=1, format_func=lambda k: RES_LAYOUTS[k], key="w_rs_layout")
    with top[1]:
        slider("Simulated years", "w_rs_years", 2.0, 25.0, 1.0, 10.0)
    with top[2]:
        slider("Well rate cap (stb/d per producer)", "w_rs_cap", 50.0, 1500.0, 25.0, 350.0)
    with top[3]:
        slider("Producer BHP (psi)", "w_rs_pwf", 500.0, 3900.0, 50.0, 2400.0)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slider("Mean permeability (mD)", "w_rs_k", 10.0, 1000.0, 10.0, 120.0)
        slider("Heterogeneity (log-sigma)", "w_rs_het", 0.0, 2.0, 0.1, 0.7)
    with c2:
        slider("Oil viscosity (cP)", "w_rs_muo", 0.3, 50.0, 0.1, 1.5)
        slider("Oil compressibility (1e-6 /psi)", "w_rs_co", 4.0, 30.0, 0.5, 12.0)
    with c3:
        slider("Edge-aquifer strength (0 = closed tank)", "w_rs_aq", 0.0, 0.02, 0.001, 0.003, fmt="%.3f")
        slider("Injection rate (rb/d, waterflood)", "w_rs_q", 0.0, 6000.0, 100.0, 1800.0)
    with c4:
        slider("Connate water Swc", "w_rs_swc", 0.05, 0.4, 0.01, 0.20)
        slider("Residual oil Sor", "w_rs_sor", 0.05, 0.4, 0.01, 0.25)
    with st.expander("Structure, pressure and relative permeability"):
        e1, e2, e3 = st.columns(3)
        with e1:
            slider("Structural relief (ft)", "w_rs_relief", 100.0, 1200.0, 50.0, 400.0)
            slider("OWC position (fraction of relief)", "w_rs_owc", 0.3, 1.3, 0.05, 0.8)
            slider("Grid cells (n x n)", "w_rs_n", 12, 40, 2, 30)
        with e2:
            slider("Initial pressure (psi)", "w_rs_pi", 2500.0, 8000.0, 100.0, 4200.0)
            slider("Bubble-point pressure (psi)", "w_rs_pb", 500.0, 4000.0, 100.0, 2000.0)
            slider("Initial Bo (rb/stb)", "w_rs_bo", 1.05, 1.8, 0.05, 1.25)
        with e3:
            slider("Cell size (ft)", "w_rs_dx", 40.0, 250.0, 10.0, 120.0)
            slider("Correlation length (cells)", "w_rs_corr", 0.0, 8.0, 0.5, 3.0)
            slider("Random seed", "w_rs_seed", 1, 99, 1, 11)
            slider("Corey exponent, water", "w_rs_nw", 1.0, 5.0, 0.1, 2.5)
            slider("Corey exponent, oil", "w_rs_no", 1.0, 5.0, 0.1, 2.0)
    ss = st.session_state
    p = dict(layout=lay, nx=int(ss.w_rs_n), ny=int(ss.w_rs_n), dx=float(ss.w_rs_dx), k_md=float(ss.w_rs_k), hetero=float(ss.w_rs_het), corr=float(ss.w_rs_corr),
             seed=int(ss.w_rs_seed), mu_o=float(ss.w_rs_muo), co=float(ss.w_rs_co) * 1e-6, q_inj=float(ss.w_rs_q), years=float(ss.w_rs_years),
             q_cap=float(ss.w_rs_cap), pwf=float(ss.w_rs_pwf), swc=float(ss.w_rs_swc), sor=float(ss.w_rs_sor), nw=float(ss.w_rs_nw), no=float(ss.w_rs_no),
             aquifer=float(ss.w_rs_aq), relief=float(ss.w_rs_relief), owc_frac=float(ss.w_rs_owc), p_i=float(ss.w_rs_pi), p_b=float(ss.w_rs_pb), bo_i=float(ss.w_rs_bo))
    errs = validate_res({**RES_DEFAULTS, **p})
    if errs:
        for e in errs:
            st.error(e)
        return
    with st.spinner("Running the reservoir simulation..."):
        res = cached_res(freeze(p))
    ts = res["ts"]
    pp = {**RES_DEFAULTS, **p}
    flood = lay == "flood"
    m = st.columns(6)
    m[0].metric("Oil in place (OOIP)", f"{res['ooip'] / 1e6:.1f} MMstb")
    m[1].metric("Cumulative oil", f"{ts.cum_oil.iloc[-1] / 1e6:.2f} MMstb", f"{res['final_recovery'] * 100:.1f} % recovery", delta_color="off")
    m[2].metric("Oil rate now", f"{ts.q_oil.iloc[-1]:,.0f} stb/d", f"peak {ts.q_oil.max():,.0f}", delta_color="off")
    m[3].metric("Water cut now", f"{ts.water_cut.iloc[-1] * 100:.0f} %")
    m[4].metric("Average pressure", f"{ts.p_avg.iloc[-1]:,.0f} psi", f"from {pp['p_i']:,.0f}", delta_color="off")
    m[5].metric("Material-balance error", f"{res['mass_balance_err'] * 100:.2f} %")
    if res["below_pb"]:
        st.warning("Pressure fell below the bubble point somewhere: real oil would release gas there (solution-gas drive). This dead-oil model has no free gas, so recovery after that point is not reliable.")
    drive = ("Waterflood plus pressure support" if flood else ("Edge-water drive" if pp["aquifer"] > 0.001 else "Fluid expansion only (closed tank)"))
    st.caption(f"Dominant mechanism: **{drive}**.  Water breakthrough "
               f"{'at %.1f years' % res['bt_years'] if not np.isnan(res['bt_years']) else 'not reached'}"
               f"{', voidage replacement ratio %.2f' % res['vrr'] if flood else ''}.  Pressure support from the aquifer or injectors keeps the oil rate on plateau; without it the rate collapses when pressure reaches the producers' BHP.")
    snaps = res["snaps"]
    k = st.select_slider("Time snapshot", options=list(range(len(snaps))), value=len(snaps) - 1, format_func=lambda i: f"{snaps[i]['t']:.1f} yr", key="w_rs_snap")
    sn = snaps[k]
    left, right = st.columns([3, 2], gap="medium")
    with left:
        show(res_structure_figure(res, sn), key="rs_3d")
    with right:
        f = go.Figure(go.Heatmap(z=1.0 - sn["sw"], colorscale=OIL_SCALE, zmin=0, zmax=1 - pp["swc"], colorbar=dict(title="So", thickness=12)))
        for nm, ty_, j, i in res["wells"]:
            f.add_trace(go.Scatter(x=[i], y=[j], mode="markers+text", text=[nm], textposition="top center", showlegend=False,
                                   marker=dict(size=9, color="#16a34a" if ty_ == "producer" else "#2563eb", line=dict(color="white", width=1))))
        f.add_hline(y=(res["owc"] - pp["z_top"]) / max(pp["relief"], 1) * (pp["ny"] - 1) * 0 + 0, line_width=0)
        f.update_layout(title=f"Oil saturation at {sn['t']:.1f} years (top = up-dip)")
        f.update_yaxes(scaleanchor="x", autorange="reversed")
        show(style_fig(f, 460), key="rs_so")
    a, b, c = st.columns(3, gap="medium")
    with a:
        f = go.Figure(go.Heatmap(z=sn["p"], colorscale="Turbo", colorbar=dict(title="psi", thickness=12)))
        f.update_layout(title="Pressure (psi)")
        f.update_yaxes(scaleanchor="x", autorange="reversed")
        show(style_fig(f, 320), key="rs_p")
    with b:
        f = go.Figure(go.Heatmap(z=np.log10(res["k"]), colorscale="Viridis", colorbar=dict(title="log10 k", thickness=12)))
        f.update_layout(title="Permeability (mD, log10)")
        f.update_yaxes(scaleanchor="x", autorange="reversed")
        show(style_fig(f, 320), key="rs_k")
    with c:
        wt = pd.DataFrame([dict(well=nm, type=ty_, cum_oil_Mstb=round(res["well_cum"][n] / 1e3, 1)) for n, (nm, ty_, j, i) in enumerate(res["wells"])])
        st.markdown("**Wells**")
        st.dataframe(wt, hide_index=True)
        st.caption(f"Grid {pp['nx']} x {pp['ny']}, cell {pp['dx']:.0f} ft, net pay {pp['h']:.0f} ft; OWC at {res['owc']:,.0f} ft; {res['steps']:,} time steps.")
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        g = make_subplots(specs=[[{"secondary_y": True}]])
        g.add_trace(go.Scatter(x=ts.years, y=ts.q_oil, name="Oil rate (stb/d)", line=dict(color="#92400e", width=3)), secondary_y=False)
        g.add_trace(go.Scatter(x=ts.years, y=ts.water_cut * 100, name="Water cut (%)", line=dict(color=COOL)), secondary_y=True)
        g.update_yaxes(rangemode="tozero", secondary_y=False)
        g.update_yaxes(range=[0, max(5.0, float(ts.water_cut.max() * 100) * 1.1)], secondary_y=True)
        g.update_layout(title="Field production", xaxis_title="Years", legend=dict(orientation="h", y=-0.25))
        show(style_fig(g, 320), key="rs_perf")
    with d2:
        g = go.Figure()
        g.add_trace(go.Scatter(x=ts.years, y=ts.p_avg, name="Average reservoir pressure", line=dict(color=INK, width=3)))
        g.add_trace(go.Scatter(x=ts.years, y=ts.p_prod, name="Pressure at producers", line=dict(color=WARM)))
        g.add_hline(y=pp["p_b"], line_dash="dot", line_color=RED, annotation_text="bubble point")
        g.add_hline(y=pp["pwf"], line_dash="dot", line_color=MUTED, annotation_text="producer BHP")
        g.update_layout(title="Pressure depletion", xaxis_title="Years", yaxis_title="psi", legend=dict(orientation="h", y=-0.25))
        show(style_fig(g, 320), key="rs_pres")
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        g = make_subplots(specs=[[{"secondary_y": True}]])
        g.add_trace(go.Scatter(x=ts.years, y=ts.cum_oil / 1e6, name="Cumulative oil (MMstb)", line=dict(color="#92400e", width=3)), secondary_y=False)
        g.add_trace(go.Scatter(x=ts.years, y=ts.recovery * 100, name="Recovery factor (%)", line=dict(color=GREEN)), secondary_y=True)
        g.update_layout(title="Cumulative oil and recovery", xaxis_title="Years", legend=dict(orientation="h", y=-0.25))
        show(style_fig(g, 300), key="rs_cum")
    with e2:
        sgrid = np.linspace(pp["swc"], 1 - pp["sor"], 100)
        krw, kro = corey_krw_kro(sgrid, pp)
        g = make_subplots(specs=[[{"secondary_y": True}]])
        g.add_trace(go.Scatter(x=sgrid, y=kro, name="kro (oil)", line=dict(color="#92400e")), secondary_y=False)
        g.add_trace(go.Scatter(x=sgrid, y=krw, name="krw (water)", line=dict(color=COOL)), secondary_y=False)
        g.add_trace(go.Scatter(x=sgrid, y=fractional_flow(sgrid, pp), name="water fractional flow", line=dict(color=INK, dash="dash")), secondary_y=True)
        g.update_layout(title="Relative permeability and fractional flow", xaxis_title="Water saturation", legend=dict(orientation="h", y=-0.25))
        show(style_fig(g, 300), key="rs_kr")
    st.download_button("Download production history (CSV)", ts.to_csv(index=False).encode(), "reservoir_history.csv", "text/csv", key="rs_csv")
    st.caption("Model: slightly compressible undersaturated oil (Bo(p) = Bo_i exp(co (p_i - p))), connate/aquifer water, Corey relative permeability, IMPES with an implicit pressure solve, harmonic-mean transmissibility, "
               "upstream fractional flow, CFL-limited saturation sub-steps, producers on BHP with a rate cap, injectors on rate, an edge aquifer as a constant-pressure source on the down-dip edge. "
               "No gravity, capillary pressure or free gas; initial pressure is uniform. A 1-D homogeneous incompressible case reproduces the Welge breakthrough (see tests). Not a field-scale or history-matched simulator.")


def page_reference():
    header("Engineering Reference", "Equations, assumptions, units and limitations.")
    st.markdown("### Rheology, cuttings transport and bit model")
    st.latex(r"\text{Bingham: } \frac{dP}{dL}=\max\!\left(\frac{PV\,v}{1500\,d^2}+\frac{YP}{225\,d},\ \frac{\rho^{0.75}v^{1.75}PV^{0.25}}{1800\,d^{1.25}}\right)\ (\text{annulus: }1000,\,200,\,1396)")
    st.latex(r"\text{Power law: } n=3.32\log_{10}\tfrac{\theta_{600}}{\theta_{300}},\quad K=\tfrac{5.11\,\theta_{300}}{511^{n}},\quad Re_c=3470-1370\,n")
    st.latex(r"C=\frac{(1-\phi)\,A_{hole}\,ROP}{F_t\,Q},\quad F_t=\frac{v_a-v_s}{v_a},\quad ECD_{cut}=ECD+C(\rho_{cut}-MW)")
    st.latex(r"MSE=\frac{WOB}{A_b}+\frac{120\pi\,N\,T}{A_b\,ROP},\quad T_{bit}=\frac{\mu\,D\,WOB}{36}")
    st.markdown("Pore pressure, fracture gradient, lithology strengths and bit-wear constants are **illustrative**, not calibrated to any basin. Slip velocity uses a Schiller-Naumann drag law; kick/loss flow rates are 60 / 80 gpm per ppg of under/over-balance (illustrative).")
    st.markdown("### Sucker-rod pump")
    st.latex(r"TF=\frac{dy}{d\varphi},\quad T=TF\,(F-CBE),\quad q_{theor}=0.1166\,A_p\,S\,N\ [\mathrm{bpd}],\quad F_o=A_p\,(P_{dis}-P_{in})")
    st.latex(r"\frac{\partial^2u}{\partial t^2}=a^2\frac{\partial^2u}{\partial z^2}-c\,\frac{\partial u}{\partial t},\quad c=D\,\frac{\pi a}{2L},\quad a\approx16{,}300\ \mathrm{ft/s}")
    st.latex(r"q=q_{max}\left[1-0.2\tfrac{P_{wf}}{P_r}-0.8\left(\tfrac{P_{wf}}{P_r}\right)^2\right],\quad HHP=7.36\times10^{-6}\,q\,L_{net}\,SG,\quad PRHP=\frac{A_{card}\,N}{396000}")
    st.markdown("Explicit finite-difference solution at the CFL limit; the valve load transfers at the polished-rod turn-arounds. Not validated against dynamometer data; no tubing movement or deviation.")
    st.markdown("### Electric submersible pump")
    st.latex(r"H_{stage}=H_{bep}\left(\tfrac{f}{60}\right)^2\left[1.3-0.3\left(\tfrac{q}{q_{bep}f/60}\right)^2\right],\quad hp=\frac{q\,H\,SG}{135{,}770\,\eta},\quad TDH=L+\frac{P_t-P_{wf}(q)}{0.433\,SG}+\Delta H_{fric}")
    st.markdown("Affinity laws (q ~ f, H ~ f^2, P ~ f^3); operating point = intersection of pump and system head by bisection; Hazen-Williams tubing friction; free-gas head derating is an illustrative curve; motor current assumes 85 % power factor and 85 % efficiency. Pump curves are generic, not a vendor catalogue.")
    st.markdown("### Reservoir simulation (oil: depletion and waterflood)")
    st.latex(r"\nabla\!\cdot\!\left(\lambda_t\,k\,\nabla p\right)=\phi\,c_t\,\frac{\partial p}{\partial t}+q,\quad \phi\,\frac{\partial S_w}{\partial t}+\nabla\!\cdot\!\left(f_w\,\mathbf{u}_t\right)=q_w,\quad f_w=\frac{k_{rw}/\mu_w}{k_{rw}/\mu_w+k_{ro}/\mu_o}")
    st.latex(r"B_o(p)=B_{oi}\,e^{c_o(p_i-p)},\quad c_t=c_oS_o+c_wS_w+c_f,\quad RF_{depl}\approx\frac{B_{oi}\,c_{t}\,\Delta p}{B_o}")
    st.markdown("IMPES: implicit pressure (sparse solve), explicit upwind saturation with a CFL limit. Undersaturated slightly compressible oil, structure with an oil-water contact, edge aquifer as a constant-pressure source, producers on BHP with a rate cap, injectors on rate. No gravity, capillarity or free gas. A 1-D homogeneous incompressible run is checked against the Welge breakthrough, and a closed-tank depletion against the compressibility recovery.")
    st.markdown("### Torque & drag, buckling and well control")
    st.latex(r"N=\sqrt{(T\,\Delta\phi\sin\theta)^2+(T\,\Delta\theta+w\sin\theta\,\Delta L)^2},\quad \Delta T=w\cos\theta\,\Delta L\pm\mu N,\quad \tau=\mu N r")
    st.latex(r"F_s=2\sqrt{\frac{EI\,w\sin\theta}{r}},\quad F_{hel}=2.83\,F_s,\quad KMW=MW+\frac{SIDPP}{0.052\,TVD},\quad ICP=SIDPP+SCR,\quad FCP=SCR\frac{KMW}{MW}")
    st.markdown("### Hydraulics (field units)")
    st.latex(r"P_h\,[\mathrm{psi}] = 0.052 \; MW\,[\mathrm{ppg}] \; TVD\,[\mathrm{ft}]")
    st.latex(r"A_{ann}\,[\mathrm{in^2}] = \tfrac{\pi}{4}\,(D_h^2 - D_p^2),\qquad v\,[\mathrm{ft/s}] = \frac{Q\,[\mathrm{ft^3/s}]}{A\,[\mathrm{ft^2}]},\quad A_{ft^2} = A_{in^2}/144")
    st.latex(r"Q\,[\mathrm{ft^3/s}] = Q\,[\mathrm{gpm}] \times 0.133681 / 60")
    st.latex(r"Re = \frac{928\,\rho\,v\,d}{\mu},\quad f = k\,\max\!\left(\frac{16}{Re},\,0.046\,Re^{-0.2}\right),\quad \frac{dP}{dL}\,[\mathrm{psi/ft}] = \frac{f\,\rho\,v^2}{25.8\,d}")
    st.latex(r"\Delta P_{bit} = \frac{8.311\times10^{-5}\,\rho\,Q^2}{C_d^2\,A_n^2}\;(C_d = 0.95),\qquad ECD = MW + \frac{\Delta P_{ann}}{0.052\,TVD}")
    st.markdown("""
- Pipe flow uses d = ID; annulus uses the slot approximation d = D_h - D_p. Newtonian fluid with an *effective* viscosity (cP); no yield stress, no pipe rotation or eccentricity effects.
- k is the dimensionless **hydraulic-resistance multiplier** (1.0 nominal). Surface losses use an assumed k_s Q^2 with k_s = 3e-4.
- Standpipe pressure (SPP) = surface + pipe + bit + annulus losses. Hydrostatic pressure is independent of flow; friction losses exist only while circulating.
- Zero flow returns zero friction, ECD = MW. Validation: positive diameters, hole > pipe OD, pipe ID < OD, non-negative flow.
""")
    st.markdown("### Units")
    st.dataframe(pd.DataFrame({"Quantity": ["Mud weight", "Flow rate", "Depth", "Diameter", "Pressure", "WOB", "Torque", "ROP", "Viscosity"],
                               "Unit": ["ppg", "gpm", "ft", "in", "psi", "klbf", "kft-lbf", "ft/h", "cP"]}), hide_index=True)
    st.markdown("### Drilling simulator (illustrative)")
    st.markdown("See the model documentation expander on the Drilling Simulator page. Coefficients are **assumptions for teaching** and are not calibrated; "
                "the ROP form is not a universal physical law. Stick-slip-like and friction disturbances are synthetic modifiers.")
    st.markdown("### Wireline and coiled tubing")
    st.markdown("Kinematic educational models: speed slows near the end point; coiled tubing adds an assumed drag factor that falls as depth approaches an "
                "assumed lock-up depth. Cable tension / tubing weight indicators are simplified (buoyed weight only). They are not validated field simulators.")
    st.markdown("### Machine learning")
    st.markdown("Isolation Forest on features derived from synthetic simulator runs with labelled injected anomalies (fixed seeds). Normal-only training, independent test runs, "
                "baseline of static +/- k sigma limits. Scores are anomaly rankings, not failure probabilities; synthetic results do not prove real-world capability.")
    st.markdown("### Limitations")
    st.markdown("- Geometry is procedural and schematic (not photorealistic, not to scale; underground sections are shortened or exaggerated and labelled).\n"
                "- Geological layers are illustrative and do not represent actual field geology.\n"
                "- No wellbore trajectory, torque-and-drag, surge/swab, cuttings transport, thermal or non-Newtonian rheology models.\n"
                "- Educational use only; not certified engineering software; no affiliation with or endorsement by any oilfield company.")
    st.markdown("### Built-in self-checks")
    st.caption("Runs the engineering and ML checks live in this session.")
    if st.button("Run self-checks", key="run_tests"):
        with st.spinner("Running..."):
            results = run_self_tests()
        for name, ok, detail in results:
            st.markdown(f"{'PASS' if ok else 'FAIL'} - {name}  `{detail}`")
        st.success(f"{sum(o for _, o, _ in results)}/{len(results)} checks passed") if all(o for _, o, _ in results) else st.error("Some checks failed")


NAV_GROUPS = {
    "Rig & Well": ["3D Rig Explorer", "Wellbore Cutaway", "Well Path & Torque-Drag"],
    "Drilling": ["Mud Circulation Lab", "Drilling Simulator", "Well Control"],
    "Intervention": ["Wireline Operations", "Coiled Tubing Lab"],
    "Production": ["Reservoir Simulation", "SRP Lab", "ESP Lab"],
    "Analytics": ["AI Diagnostics", "Scenario Comparison"],
    "Reference": ["Engineering Reference"],
}
assert sorted(n for v in NAV_GROUPS.values() for n in v) == sorted(PAGES[1:]), "navigation groups must cover every page once"
PAGE_FUNCS = {
    "Operations Center": page_operations, "3D Rig Explorer": page_rig_explorer, "Wellbore Cutaway": page_cutaway,
    "Mud Circulation Lab": page_mud, "Drilling Simulator": page_drilling, "Wireline Operations": page_wireline,
    "Coiled Tubing Lab": page_coiled, "AI Diagnostics": page_ai, "Scenario Comparison": page_scenarios,
    "SRP Lab": page_srp, "ESP Lab": page_esp, "Reservoir Simulation": page_reservoir, "Well Path & Torque-Drag": page_welldesign, "Well Control": page_wellcontrol,
    "Engineering Reference": page_reference,
}


# =============================================================================
# 8. MAIN
# =============================================================================
def reset_everything():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def _guarded(fn):
    """Wrap a page so an unexpected error shows a clear message instead of a blank screen."""
    @functools.wraps(fn)
    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - surface any page failure to the user
            LOG.exception("page failed: %s", getattr(fn, "__name__", "?"))
            st.error(f"This page hit an unexpected error: {exc}")
            st.exception(exc)
    return run


PAGE_OBJS: dict = {}


def main():
    st.set_page_config(page_title="PetroSite 3D", page_icon=":material/oil_barrel:", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<style>
div[data-testid="stMetricDelta"] div { white-space: normal !important; line-height: 1.15; }
div[data-testid="stMetricDelta"] { background: transparent !important; padding: 0 !important; }
div[data-testid="stMetricDelta"] svg { display: none; }
div[data-testid="stMetricValue"] div { white-space: normal !important; overflow-wrap: anywhere; }
</style>""", unsafe_allow_html=True)
    init_state()
    keep_widget_state()
    for i, name in enumerate(PAGES):
        PAGE_OBJS[name] = st.Page(_guarded(PAGE_FUNCS[name]), title=name, url_path=_slug(name), default=(i == 0))
    sections = {"": [PAGE_OBJS["Operations Center"]]}
    for grp, names in NAV_GROUPS.items():
        sections[grp] = [PAGE_OBJS[n] for n in names]
    nav = st.navigation(sections, position="top")
    _g1, _g2 = st.columns([9, 1.3])
    with _g2:
        with st.popover("Graphics", icon=":material/tune:", use_container_width=True):
            st.radio("3D quality", list(GFX_SCALE), index=2, key="w_gfx", horizontal=False,
                     help="Standard = fastest. High = smoother curved parts. Ultra (default) = most detailed and glossiest materials; switch down on weak GPUs.")
            st.caption("Animations run in your browser; quality only changes mesh density.")
    nav.run()
    st.markdown(f"<div class='ps-footer ps-small'><b>PetroSite 3D</b> &nbsp;|&nbsp; {DISCLAIMER}</div>", unsafe_allow_html=True)
    st.button("Reset all application state", on_click=reset_everything, key="reset_all")


if __name__ == "__main__":
    main()
