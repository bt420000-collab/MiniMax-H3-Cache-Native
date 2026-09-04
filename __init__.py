"""ComfyUI entrypoint for H3BC."""
from importlib.util import find_spec

if find_spec("comfy") is None:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    from . import nodes as _nodes
    from .runtime_alpha4 import apply_alpha4_runtime

    apply_alpha4_runtime(_nodes)
    NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
