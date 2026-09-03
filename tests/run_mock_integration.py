import copy, importlib.util, sys, types
from pathlib import Path
import torch

ROOT=Path(__file__).parents[1]
pkg=types.ModuleType("h3bc_pkg"); pkg.__path__=[str(ROOT)]; sys.modules["h3bc_pkg"]=pkg
comfy=types.ModuleType("comfy"); pe=types.ModuleType("comfy.patcher_extension")
class W: DIFFUSION_MODEL="DIFFUSION_MODEL"; OUTER_SAMPLE="OUTER_SAMPLE"
pe.WrappersMP=W; comfy.patcher_extension=pe
sys.modules["comfy"]=comfy; sys.modules["comfy.patcher_extension"]=pe

def load(name,file):
    spec=importlib.util.spec_from_file_location(name, ROOT/file)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
load("h3bc_pkg.h3bc_policy","h3bc_policy.py")
n=load("h3bc_pkg.nodes","nodes.py")

class Layout:
    segments=[(0,4,"audio"),(4,12,"video")]
    signature=(0,2,0,0,0)
class Sampling:
    def percent_to_sigma(self,p): return 1.0-p
class DM:
    def __init__(self): self.blocks=[object() for _ in range(50)]
class Model:
    def __init__(self):
        self.dm=DM(); self.sampling=Sampling(); self.model_options={"transformer_options":{}}; self.patches=[]; self.wrappers=[]
    def get_model_object(self,k): return self.dm if k=="diffusion_model" else self.sampling
    def clone(self):
        x=Model(); x.dm=self.dm; x.sampling=self.sampling; x.model_options=copy.deepcopy(self.model_options); return x
    def set_model_patch_replace(self,fn,*keys): self.patches.append((keys,fn))
    def add_wrapper_with_key(self,*args): self.wrappers.append(args)

DM.__name__="MiniMaxH3Model"; DM.__qualname__="MiniMaxH3Model"
m=Model()
# Legacy alpha1 preset remains accepted with the old widget call signature.
out=n.ApplyMiniMaxH3BC().apply(m,n.LEGACY_BALANCED,0.07,0.1,0.95,2,1,True,0.58,1.25,0.8,True,"disable_dynamic_vbars",False)[0]
assert len(out.patches)==50
assert len(out.wrappers)==2
assert out.model_options["transformer_options"]["prefetch_dynamic_vbars"] is False
# OFF installs nothing and returns the original MODEL identity.
assert n.ApplyMiniMaxH3BC().apply(m,n.MODE_OFF,0.04,0.1,0.95,1,1,True,0.82,1.0,0.8,True,"inherit",False)[0] is m

payload={"layout":Layout()}
base=torch.ones(12,3)
x=(torch.zeros(1),torch.zeros(1))
# Conservative policy: 1 warmup exact, then one cache hit, then mandatory exact refresh.
cfg=n.H3BCConfig(
    0.10,start_percent=0.1,end_percent=0.95,max_consecutive_hits=1,probe_blocks=1,
    dynamic_threshold=False,error_budget_units=2.0,audio_guard_ratio=0.8,temporal_guard=False,
    warmup_steps=1,adaptive_refresh=False,
)
c=n.H3BCAdaptiveCache(cfg,0.9,0.05,50,False,run_mode="test")
c.begin_call(x,torch.tensor([800.0]),{},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(base); assert not c.current.use_cache; c.finish_full_step(base+2); c.end_call()
c.begin_call(x,torch.tensor([700.0]),{},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(base*1.02); assert c.current.use_cache; y=c.finish_cached_step(base*1.02); assert y.shape==base.shape; c.end_call()
c.begin_call(x,torch.tensor([600.0]),{},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(base*1.02); assert not c.current.use_cache; assert "refresh:max-cached-run" in c.reason_counts; c.finish_full_step(base+2); c.end_call()
# External refresh contract can force an exact step.
c.begin_call(x,torch.tensor([500.0]),{"h3bc_control":{"force_refresh":True}},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(base*1.02); assert not c.current.use_cache; assert "refresh:external" in c.reason_counts; c.end_call()
print("H3BC alpha2 mock integration PASS")
