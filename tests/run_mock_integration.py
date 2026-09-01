import copy, importlib.util, sys, types
from pathlib import Path
import torch

ROOT=Path(__file__).parents[1]
# Fake package for relative imports.
pkg=types.ModuleType("h3bc_pkg"); pkg.__path__=[str(ROOT)]; sys.modules["h3bc_pkg"]=pkg
# Minimal comfy API used by nodes.py.
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

DM.__name__="MiniMaxH3Model"
DM.__qualname__="MiniMaxH3Model"
DM.__name__="MiniMaxH3Model"

m=Model()
out=n.ApplyMiniMaxH3BC().apply(m,"H3BC Balanced α — 0.07",0.07,0.1,0.95,2,1,True,0.58,1.25,0.8,True,"disable_dynamic_vbars",False)[0]
assert len(out.patches)==50
assert len(out.wrappers)==2
assert out.model_options["transformer_options"]["prefetch_dynamic_vbars"] is False

cfg=n.H3BCConfig(0.10,start_percent=0.1,end_percent=0.95,max_consecutive_hits=2,probe_blocks=1,dynamic_threshold=False,error_budget_units=2.0,audio_guard_ratio=0.8,temporal_guard=False)
c=n.H3BCAdaptiveCache(cfg,0.9,0.05,50,False)
x=(torch.zeros(1),torch.zeros(1))
payload={"layout":Layout()}
base=torch.ones(12,3)
c.begin_call(x,torch.tensor([800.0]),{},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(base); assert not c.current.use_cache; c.finish_full_step(base+2); c.end_call()
c.begin_call(x,torch.tensor([700.0]),{},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(base*1.02); assert c.current.use_cache; y=c.finish_cached_step(base*1.02); assert y.shape==base.shape; c.end_call()
probe=base*1.02; probe[:4]=1.5
c.begin_call(x,torch.tensor([600.0]),{},payload); c.capture_probe_input(torch.zeros_like(base)); c.decide_after_probe(probe); assert not c.current.use_cache; c.end_call()
print("H3BC mock integration PASS")
