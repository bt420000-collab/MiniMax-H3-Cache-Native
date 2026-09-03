import importlib.util, sys, types
from pathlib import Path
import torch

ROOT=Path(__file__).parents[1]
pkg=types.ModuleType('h3bc_pkg'); pkg.__path__=[str(ROOT)]; sys.modules['h3bc_pkg']=pkg
comfy=types.ModuleType('comfy'); pe=types.ModuleType('comfy.patcher_extension')
class W: DIFFUSION_MODEL='DIFFUSION_MODEL'; OUTER_SAMPLE='OUTER_SAMPLE'
pe.WrappersMP=W; comfy.patcher_extension=pe
sys.modules['comfy']=comfy; sys.modules['comfy.patcher_extension']=pe

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
load('h3bc_pkg.h3bc_policy','h3bc_policy.py')
n=load('h3bc_pkg.nodes','nodes.py')

ctx=n.CacheContext(key=('test',), audio_slice=(0,4), video_slice=(4,12), step_index=1)
p=n.H3BCReferenceProfiler(50,debug=False,sample_tokens=4,sample_channels=2)
x=torch.zeros(12,8)
y=x+1
p.observe_block(context=ctx,block_index=17,block_input=x,block_output=y,timing=p.begin_timing(x))
ctx.step_index=2
y2=x+1.02
p.observe_block(context=ctx,block_index=17,block_input=x,block_output=y2,timing=p.begin_timing(x))
rows=p.aggregate()
row=next(r for r in rows if r['block']==17)
assert row['mean_residual_diff'] is not None
assert 0.019 < row['mean_residual_diff'] < 0.021
assert row['mean_audio_diff'] is not None and row['mean_video_diff'] is not None
print('H3BC reference profiler mock PASS',row)
