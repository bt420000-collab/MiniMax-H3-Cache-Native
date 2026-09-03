from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_no_forward_monkey_patch():
    text = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert "MiniMaxH3Model._forward =" not in text
    assert "set_model_patch_replace" in text
    assert "WrappersMP.DIFFUSION_MODEL" in text
    assert "WrappersMP.OUTER_SAMPLE" in text


def test_h3bc_marker_and_conflicts():
    text = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert '"h3bc_v2"' in text
    assert '"easycache"' in text
    assert '"cache_dit_turbo"' in text


def test_production_refresh_contract_present():
    text = (ROOT / "nodes.py").read_text(encoding="utf-8")
    policy = (ROOT / "h3bc_policy.py").read_text(encoding="utf-8")
    for marker in ("warmup_steps", "force_refresh_every", "force_refresh_steps", "adaptive_refresh"):
        assert marker in text
        assert marker in policy
    assert "LOSSLESS / REFERENCE" in text
    assert "H3BC_DEBUG" in text
    assert "h3bc_control" in text


def test_off_mode_returns_unmodified_model():
    text = (ROOT / "nodes.py").read_text(encoding="utf-8")
    assert "if mode == MODE_OFF" in text
    assert "return (model,)" in text
