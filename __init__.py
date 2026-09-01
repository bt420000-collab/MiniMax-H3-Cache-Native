"""ComfyUI entrypoint for H3BC.

Importing the source tree outside ComfyUI (for policy/unit tests) is allowed; the
node mappings are populated when the ComfyUI runtime is present.
"""
from importlib.util import find_spec

if find_spec("comfy") is None:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
