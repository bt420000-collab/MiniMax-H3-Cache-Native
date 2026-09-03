import math

from h3bc_policy import (
    H3BCConfig,
    forced_refresh_reason,
    normalized_error,
    parse_refresh_steps,
    threshold_multiplier,
    window_phase,
)


def test_config():
    H3BCConfig(0.07).validate()


def test_dynamic_threshold_is_tight_at_edges():
    assert math.isclose(threshold_multiplier(0.0, 0.6, True), 0.6)
    assert math.isclose(threshold_multiplier(0.5, 0.6, True), 1.0)
    assert math.isclose(threshold_multiplier(1.0, 0.6, True), 0.6)


def test_phase():
    assert math.isclose(window_phase(0.9, 0.9, 0.1), 0.0)
    assert math.isclose(window_phase(0.5, 0.9, 0.1), 0.5)
    assert math.isclose(window_phase(0.1, 0.9, 0.1), 1.0)


def test_normalized_error_uses_strictest_guard():
    n = normalized_error(0.04, 0.03, 0.06, 0.08, 0.04, 0.08, True)
    assert math.isclose(n, 0.75)


def test_refresh_step_parser():
    assert parse_refresh_steps("8, 4; 8,12") == (4, 8, 12)
    assert parse_refresh_steps("") == ()


def test_warmup_forces_exact():
    cfg = H3BCConfig(0.04, warmup_steps=4, max_consecutive_hits=1)
    assert forced_refresh_reason(step_index=1, config=cfg, consecutive_hits=0) == "warmup"
    assert forced_refresh_reason(step_index=4, config=cfg, consecutive_hits=0) == "warmup"
    assert forced_refresh_reason(step_index=5, config=cfg, consecutive_hits=0) is None


def test_max_cached_run_forces_refresh():
    cfg = H3BCConfig(0.04, warmup_steps=0, max_consecutive_hits=1)
    assert forced_refresh_reason(step_index=5, config=cfg, consecutive_hits=1) == "max-cached-run"


def test_periodic_and_scheduled_refresh():
    cfg = H3BCConfig(0.04, warmup_steps=0, force_refresh_every=5, force_refresh_steps=(7,))
    assert forced_refresh_reason(step_index=5, config=cfg, consecutive_hits=0) == "periodic"
    assert forced_refresh_reason(step_index=7, config=cfg, consecutive_hits=0) == "scheduled-step"


def test_adaptive_and_external_refresh():
    cfg = H3BCConfig(0.04, warmup_steps=0)
    assert forced_refresh_reason(step_index=5, config=cfg, consecutive_hits=0, force_refresh_next=True) == "adaptive"
    assert forced_refresh_reason(step_index=5, config=cfg, consecutive_hits=0, external_force_refresh=True) == "external"
