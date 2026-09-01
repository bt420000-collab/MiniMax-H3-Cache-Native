import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from h3bc_policy import *
H3BCConfig(0.07).validate()
assert threshold_multiplier(0.5, 0.58, True) > threshold_multiplier(0.0, 0.58, True)
assert normalized_error(0.03,0.02,0.04,0.07,0.056,0.07,True) < 1.0
print("H3BC policy smoke PASS")
