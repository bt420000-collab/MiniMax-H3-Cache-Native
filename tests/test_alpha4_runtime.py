import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]


def load_runtime():
    pkg = types.ModuleType("h3bc_alpha4_pkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["h3bc_alpha4_pkg"] = pkg
    comfy = types.ModuleType("comfy")
    pe = types.ModuleType("comfy.patcher_extension")

    class W:
        DIFFUSION_MODEL = "DIFFUSION_MODEL"
        OUTER_SAMPLE = "OUTER_SAMPLE"

    pe.WrappersMP = W
    comfy.patcher_extension = pe
    sys.modules["comfy"] = comfy
    sys.modules["comfy.patcher_extension"] = pe

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, ROOT / file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("h3bc_alpha4_pkg.h3bc_policy", "h3bc_policy.py")
    n = load("h3bc_alpha4_pkg.nodes", "nodes.py")
    rt = load("h3bc_alpha4_pkg.runtime_alpha4", "runtime_alpha4.py")
    rt.apply_alpha4_runtime(n)
    return n


def inplace_block(delta):
    def original(args):
        args["img"].add_(delta)
        return {"img": args["img"]}
    return original


def run_step(cache, patches, sigma, deltas):
    hidden = torch.zeros(6, 4)
    cache.begin_call((torch.zeros(1),), torch.tensor([sigma * 1000.0]), {}, None)
    for i, delta in enumerate(deltas):
        hidden = patches[i]({"img": hidden}, {"original_block": inplace_block(delta)})["img"]
    cache.end_call()
    return hidden


def test_alpha4_repairs_inplace_probe_and_tail_aliasing():
    n = load_runtime()
    cfg = n.H3BCConfig(
        threshold=0.10,
        start_percent=0.0,
        end_percent=1.0,
        max_consecutive_hits=1,
        probe_blocks=1,
        dynamic_threshold=False,
        edge_ratio=1.0,
        error_budget_units=2.0,
        audio_guard_ratio=1.0,
        temporal_guard=False,
        warmup_steps=0,
        adaptive_refresh=False,
    )
    cache = n.H3BCAdaptiveCache(cfg, 1.0, 0.0, 3, False, run_mode="alias-regression")
    patches = [n.make_block_patch(cache, i, 2) for i in range(3)]

    exact = run_step(cache, patches, 0.8, [1.0, 2.0, 3.0])
    ctx = cache.contexts[("default",)]
    assert torch.allclose(exact, torch.full_like(exact, 6.0))
    assert torch.allclose(ctx.previous_probe_residual, torch.full_like(exact, 1.0))
    assert torch.allclose(ctx.tail_residual, torch.full_like(exact, 5.0))

    cached = run_step(cache, patches, 0.7, [1.01, 20.0, 20.0])
    assert cache.cached_steps == 1
    assert torch.allclose(cached, torch.full_like(cached, 6.01), atol=1e-5)
    assert cache.step_records[-1]["normalized"] > 0.0


def test_alpha4_presets_keep_conservative_refresh():
    n = load_runtime()
    assert n.VERSION == "2.0.0-alpha4"
    assert n.PRESETS[n.MODE_SAFE].config.threshold == 0.090
    assert n.PRESETS[n.MODE_BALANCED].config.threshold == 0.100
    assert n.PRESETS[n.MODE_AGGRESSIVE].config.threshold == 0.120
    assert n.PRESETS[n.MODE_SAFE].config.max_consecutive_hits == 1
    assert n.PRESETS[n.MODE_BALANCED].config.max_consecutive_hits == 1
