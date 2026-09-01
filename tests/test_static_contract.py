from pathlib import Path


def test_no_forward_monkey_patch():
    text = (Path(__file__).parents[1] / "nodes.py").read_text(encoding="utf-8")
    assert "MiniMaxH3Model._forward =" not in text
    assert "set_model_patch_replace" in text
    assert "WrappersMP.DIFFUSION_MODEL" in text
    assert "WrappersMP.OUTER_SAMPLE" in text


def test_h3bc_marker_and_conflicts():
    text = (Path(__file__).parents[1] / "nodes.py").read_text(encoding="utf-8")
    assert '"h3bc_v2"' in text
    assert '"easycache"' in text
    assert '"cache_dit_turbo"' in text
