"""Automated checks for PetroSite 3D. Run:  python -m pytest -q"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402


# ---------------- engineering ----------------
def test_hydrostatic_reference():
    assert app.hydrostatic_psi(10.0, 10000.0) == pytest.approx(5200.0)


def test_annular_area_and_velocity():
    a = app.annular_area_in2(8.5, 5.0)
    assert a == pytest.approx(math.pi / 4 * (8.5**2 - 5.0**2))
    q_ft3s = 600 * 0.133680556 / 60
    assert app.gpm_to_ft3s(600) == pytest.approx(q_ft3s)
    assert app.velocity_ft_s(600, a) == pytest.approx(q_ft3s / (a / 144))


def test_zero_flow_is_safe():
    h = app.hydraulics_summary(10.0, 0.0, 8000.0, 8.5, 5.0)
    assert h["dp_ann_psi"] == 0 and h["spp_psi"] == 0
    assert h["ecd_ppg"] == pytest.approx(10.0)


def test_ecd_exceeds_mw_when_circulating_and_scales_with_flow():
    lo = app.hydraulics_summary(10.0, 300.0, 8000.0, 8.5, 5.0)["ecd_ppg"]
    hi = app.hydraulics_summary(10.0, 700.0, 8000.0, 8.5, 5.0)["ecd_ppg"]
    assert 10.0 < lo < hi


@pytest.mark.parametrize("kw", [
    dict(mw=0, q=500, tvd=8000, dh=8.5, dp=5.0), dict(mw=10, q=-1, tvd=8000, dh=8.5, dp=5.0),
    dict(mw=10, q=500, tvd=0, dh=8.5, dp=5.0), dict(mw=10, q=500, tvd=8000, dh=5.0, dp=5.0),
    dict(mw=10, q=500, tvd=8000, dh=4.0, dp=5.0), dict(mw=10, q=500, tvd=8000, dh=8.5, dp=-1.0),
])
def test_invalid_inputs_rejected(kw):
    assert app.validate_hydraulics(**kw)
    with pytest.raises(ValueError):
        app.hydraulics_summary(kw["mw"], kw["q"], kw["tvd"], kw["dh"], kw["dp"])


def test_pressure_loss_monotonic_in_flow():
    qs = np.array([100, 300, 600, 900.0])
    dp = app.annular_pressure_loss_psi(qs, 10.0, 8.5, 5.0, 8000.0, 15.0)
    assert (np.diff(dp) > 0).all()


# ---------------- simulator ----------------
def test_simulation_reproducible_and_monotonic():
    a, b = app.simulate_drilling(dict(app.SIM_DEFAULTS)), app.simulate_drilling(dict(app.SIM_DEFAULTS))
    pd.testing.assert_frame_equal(a, b)
    assert (np.diff(a.depth_ft) >= 0).all()


def test_baseline_differs_from_disturbed():
    base = app.simulate_drilling({**app.SIM_DEFAULTS, "fric": 0.0, "ss": 0.0})
    dist = app.simulate_drilling({**app.SIM_DEFAULTS, "fric": 0.6, "ss": 0.6})
    assert dist.torque_kftlbf.std() > base.torque_kftlbf.std()
    assert dist.rop_ftph.mean() < base.rop_ftph.mean()


def test_more_formation_resistance_lowers_rop():
    easy = app.simulate_drilling({**app.SIM_DEFAULTS, "fr": 0.8}).rop_ftph.mean()
    hard = app.simulate_drilling({**app.SIM_DEFAULTS, "fr": 2.0}).rop_ftph.mean()
    assert hard < easy


def test_simulator_validation():
    with pytest.raises(ValueError):
        app.simulate_drilling({**app.SIM_DEFAULTS, "hole_in": 4.0, "pipe_od": 5.0})
    with pytest.raises(ValueError):
        app.simulate_drilling({**app.SIM_DEFAULTS, "wob": 0})


# ---------------- ML ----------------
@pytest.fixture(scope="module")
def ml():
    train, test = app.generate_dataset()
    return train, test, app.train_detector(train)


def test_ml_no_leakage_and_independence(ml):
    train, test, _ = ml
    assert int(train["label"].sum()) == 0
    assert set(train.run_id).isdisjoint(set(test.run_id))
    assert not {"label", "anom_type"} & set(app.FEATURES)


def test_ml_reproducible():
    a, _ = app.generate_dataset(n_train=3, n_test=2)
    b, _ = app.generate_dataset(n_train=3, n_test=2)
    pd.testing.assert_frame_equal(a, b)


def test_ml_beats_static_baseline(ml):
    _, test, det = ml
    y = test.label.values
    f1_if = app.eval_binary(y, det.predict(test))["f1"]
    f1_bl = app.eval_binary(y, det.baseline_predict(test, 3.0))["f1"]
    assert f1_if > f1_bl
    assert f1_if > 0.5


def test_ml_false_positive_rate_bounded(ml):
    _, test, det = ml
    assert app.eval_binary(test.label.values, det.predict(test))["fpr"] < 0.08


# ---------------- 3D geometry ----------------
def _ok(fig):
    json.loads(fig.to_json())  # serialisable
    assert len(fig.data) > 0


def test_rig_figure_all_cameras_and_selection():
    for cam in app.PRESET_NAMES:
        _ok(app.rig_figure(camera=cam))
    for eid in app.GEOMETRY:
        _ok(app.rig_figure(selected=eid))


def test_every_equipment_has_required_info():
    for eid in app.GEOMETRY:
        e = app.EQUIP[eid]
        assert all(e[k] for k in ("name", "purpose", "components", "variables", "explanation"))


@pytest.mark.parametrize("q", [0.0, 520.0])
def test_circulation_figure(q):
    h = app.hydraulics_summary(10.0, q, 8000.0, 8.5, 5.0)
    _ok(app.build_circulation_figure(8000.0, h["v_pipe_fts"], h["v_ann_fts"], 5.0, q, True))


@pytest.mark.parametrize("cut,exag", [(True, False), (False, False), (True, True)])
def test_cutaway_figure(cut, exag):
    _ok(app.cutaway_figure(8000, 5000, 8.5, 5.0, cut, exag, {"layers", "casing", "cement", "string", "annulus", "mud"}))
    _ok(app.cutaway_figure(1000, 500, 6.0, 3.5, cut, exag, set()))


def test_service_scenes_at_extremes():
    for d in (0.0, 4000.0, 8000.0):
        _ok(app.wireline_figure(d, 8000.0))
        _ok(app.coiled_tubing_figure(d, 8000.0))


def test_bit_view():
    df = app.simulate_drilling(dict(app.SIM_DEFAULTS))
    for i in (0, len(df) // 2, len(df) - 1):
        _ok(app.bit_view_figure(df, i))


# ---------------- service tool stepping ----------------
def test_step_tool_reaches_target_and_stops():
    st_ = dict(depth=0.0, elapsed=0.0, running=True, dir="RIH", last=0.0)
    import time as _t
    for _ in range(4000):
        st_["last"] = _t.time() - 1.0
        app.step_tool(st_, 500.0, 100.0, 15, 100.0)
        if not st_["running"]:
            break
    assert st_["depth"] == 500.0 and not st_["running"]


def test_self_tests_all_pass():
    assert all(ok for _, ok, _ in app.run_self_tests())


# ---------------- operational status honesty ----------------
def test_idle_detector_status_is_warmup_not_alarm():
    df = app.simulate_drilling(dict(app.SIM_DEFAULTS))
    label, kind, recent, total = app.anomaly_status_for(dict(app.SIM_DEFAULTS), df, 0)
    assert label.startswith("Warming up") and total == 0


def test_undisturbed_default_run_has_no_alarm():
    p = {**app.SIM_DEFAULTS, "fric": 0.0, "ss": 0.0}
    df = app.simulate_drilling(p)
    label, kind, _, _ = app.anomaly_status_for(p, df, len(df) - 1)
    assert kind != "bad"


# ---- Stage A: rheology / simulator realism ----
def test_bingham_and_powerlaw_newtonian_limits():
    import app
    nw = app.make_rheology("Newtonian", 15.0, 60.0, 38.0)
    base = app.pipe_dp_rheo(500.0, 10.0, 4.276, 8000.0, nw, 1.0)
    assert base > 0
    # power-law with n=1 (theta600 = 2*theta300) must collapse to Newtonian-like laminar/turbulent behaviour (same order)
    pl = app.make_rheology("Power law", 15.0, 60.0, 30.0)
    v = app.pipe_dp_rheo(500.0, 10.0, 4.276, 8000.0, pl, 1.0)
    assert 0.3 < v / base < 3.0


def test_sim_monotonic_and_bounded():
    import app
    df = app.simulate_drilling(dict(app.SIM_DEFAULTS))
    assert (df.wear.diff().dropna() >= 0).all()
    assert df.depth_ft.is_monotonic_increasing
    assert (df.rop_ftph > 0).all() and (df.cut_conc >= 0).all()


def test_overpressure_causes_kick_flow():
    import app
    df = app.simulate_drilling({**app.SIM_DEFAULTS, "overpress": 1.0, "op_top": 7000.0, "mw": 9.0})
    assert df.flow_out_gpm.max() > df.flow_set_gpm.iloc[0]
    assert df.pit_gain_bbl.iloc[-1] > 0


# ---- SRP, well path, torque & drag, well control ----
def test_min_curvature_build_arc():
    import app
    p = app.well_path(0.0, 3.0, 90.0, 0.0, 3000.0, step=10.0)
    r = 5729.578 / 3.0
    assert abs(p.tvd.iloc[-1] - r) < 2.0 and abs(p.north.iloc[-1] - r) < 2.0
    assert abs(p.dls.max() - 3.0) < 0.05


def test_vertical_hookload_equals_buoyed_weight_and_no_torque():
    import app
    pv = app.well_path(0, 0, 0, 0, 8000.0)
    r = app.torque_drag(pv, 0.3, 10.0, '5" 19.5 lb/ft', 8.5)
    assert abs(r["hookload"]["pickup"] - r["free_hang"]) < 1.0
    assert r["surface_torque"]["rotating"] == 0.0


def test_horizontal_tangent_drag_is_mu_w_l():
    import app
    md = np.arange(0.0, 3000.0 + 1, 30.0)
    df = pd.DataFrame(dict(md=md, inc_deg=90.0, azi_deg=0.0, tvd=0.0, north=md, east=0.0, dls=0.0))
    r = app.torque_drag(df, 0.3, 10.0, '5" 19.5 lb/ft', 8.5)
    w = r["free_hang"]
    assert abs(r["hookload"]["pickup"] - 0.3 * w) < 1.0
    assert abs(r["hookload"]["slackoff"] + 0.3 * w) < 1.0


def test_kill_sheet_formulas():
    import app
    k = app.kill_sheet(10.0, 10000.0, 520.0, 700.0, 20.0, 600.0, 30.0, 150.0, 500.0, 0.1, 6000.0, 14.0, 8.5, 5.0)
    assert abs(k["kmw"] - (10.0 + 520.0 / (0.052 * 10000.0))) < 1e-9
    assert abs(k["icp"] - 1120.0) < 1e-9
    assert abs(k["fcp"] - 600.0 * k["kmw"] / 10.0) < 1e-9
    assert abs(k["maasp"] - (14.0 - 10.0) * 0.052 * 6000.0) < 1e-9


def test_srp_kinematics_stroke_and_load_limits():
    import app
    phi, y, tf = app.srp_kinematics(64.0)
    assert abs((y.max() - y.min()) * 12.0 - 64.0) < 1e-6
    r = app.srp_simulate({"spm": 2.0, "cycles": 3})
    assert r["f_min"] > 0.9 * r["rod_weight_buoyed"] - 2000.0
    assert r["f_max"] > r["rod_weight_buoyed"] + 0.7 * r["fo"]
    assert abs(r["inflow"]["q_theor"] - 0.1166 * r["inflow"]["ap"] * 64.0 * 2.0) < 1e-6


def test_srp_pump_off_detected_and_vogel_inverse():
    import app
    r = app.srp_simulate({"pr": 1500.0, "qmax": 60.0})
    assert r["inflow"]["pump_off"] and r["inflow"]["fill"] < 1.0
    q = app.vogel_q(800.0, 1800.0, 500.0)
    assert abs(app.vogel_pwf(float(q), 1800.0, 500.0) - 800.0) < 1e-6


# ---- ESP ----
def test_esp_affinity_laws():
    import app
    pump = "Medium (BEP 2,500 bpd)"
    q1, q2 = 1500.0, 1500.0 * 50.0 / 60.0
    h1 = float(app.esp_stage_head(q1, 60.0, pump))
    h2 = float(app.esp_stage_head(q2, 50.0, pump))
    assert abs(h2 / h1 - (50.0 / 60.0) ** 2) < 1e-9
    p1 = float(app.esp_stage_hp(q1, 60.0, pump, 0.9))
    p2 = float(app.esp_stage_hp(q2, 50.0, pump, 0.9))
    assert abs(p2 / p1 - (50.0 / 60.0) ** 3) < 1e-9


def test_esp_operating_point_matches_system_head_and_scales_with_frequency():
    import app
    r = app.esp_simulate({})
    p = r["params"]
    assert abs(float(app.esp_pump_head(r["q"], p)) - float(app.esp_system_head(r["q"], p))) < 1.0
    q = r["sweep"].q.values
    assert np.all(np.diff(q) >= -1e-6)  # production rises with frequency


def test_esp_gas_derate_monotone_and_motor_amps_formula():
    import app
    g = app.esp_gas_derate(np.linspace(0, 0.4, 20))
    assert np.all(np.diff(g) <= 1e-12) and g[0] == 1.0 and g[-1] == 0.0
    r = app.esp_simulate({})
    amps = r["shaft_hp"] * 745.7 / (np.sqrt(3) * 2000.0 * 0.85 * 0.85)
    assert abs(r["amps"] - amps) < 1e-9
    assert r["motor_hp"] >= r["shaft_hp"]


def test_esp_cannot_lift_flagged():
    import app
    r = app.esp_simulate({"stages": 10, "freq": 30.0})
    assert r["q"] == 0.0 and any("cannot lift" in f for f in r["flags"])


# ---- reservoir simulation (oil) ----
def test_waterflood_1d_matches_welge_and_conserves_mass():
    import app
    p = {**app.RES_DEFAULTS, **dict(layout="corner", nx=60, ny=1, hetero=0.0, dx=20.0, pvi_max=0.9, q_inj=300.0, co=0.0, cw=0.0, cf=0.0,
                                    bo_i=1.0, bw=1.0, aquifer=0.0, relief=0.0)}
    r = app._res_run(p)
    welge = app.welge_breakthrough_pvi(p)
    assert abs(r["bt_pvi"] - welge) / welge < 0.12
    assert r["mass_balance_err"] < 0.10  # sharp 1-D shock with saturation clipping at Sor; realistic cases are < 1 %


def test_closed_tank_depletion_matches_compressibility():
    import app
    r = app.res_simulate({"layout": "depletion4", "aquifer": 0.0, "hetero": 0.0, "nx": 16, "ny": 16, "years": 20.0, "owc_frac": 1.5})
    p = {**app.RES_DEFAULTS}
    assert r["mass_balance_err"] < 0.03
    assert 0.0 < r["final_recovery"] < 0.2
    assert r["ts"].p_avg.iloc[-1] < p["p_i"]


def test_oil_reservoir_bounds_and_aquifer_support():
    import app
    base = dict(nx=16, ny=16, years=5.0)
    r0 = app.res_simulate({**base, "aquifer": 0.0})
    r1 = app.res_simulate({**base, "aquifer": 0.01})
    for r in (r0, r1):
        sw = r["snaps"][-1]["sw"]
        assert sw.min() >= 0.2 - 1e-6 and sw.max() <= 1.0
        assert r["mass_balance_err"] < 0.03
    assert r1["ts"].p_avg.iloc[-1] > r0["ts"].p_avg.iloc[-1]
    assert r1["final_recovery"] > r0["final_recovery"]


def test_reservoir_validation():
    import app
    import pytest
    with pytest.raises(ValueError):
        app.res_simulate({"swc": 0.6, "sor": 0.4})
