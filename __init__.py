__version__ = "1.0.0"

import os
import importlib

cwd_path = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes_list = ["ReduxFineTune"]

for module_name in nodes_list:
    imported_module = importlib.import_module(f".{module_name}", __name__)
    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f'\033[34m[ComfyUI-ReduxFineTune] v{__version__} \033[92mLoaded\033[0m') 