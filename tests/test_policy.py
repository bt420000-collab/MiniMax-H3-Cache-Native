import math
from h3bc_policy import H3BCConfig, normalized_error, threshold_multiplier, window_phase


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
