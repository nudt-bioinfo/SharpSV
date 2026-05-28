import importlib.util
import sys
from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent
_PRIVATE_BUNDLE_MODULE_PREFIX = "_sharpsv_bundle"
_LEGACY_INIT_NAME = "m" + "amnet"
_BUNDLE_NATIVE_DIR = _MODULE_DIR / "_bundle" / "native"


def _load_extension(module_name, binary_path):
    spec = importlib.util.spec_from_file_location(module_name, binary_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {binary_path}")

    previous = sys.modules.get(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _load_backend():
    search_roots = [_MODULE_DIR, _BUNDLE_NATIVE_DIR, _MODULE_DIR.parent]
    module_names = [
        _LEGACY_INIT_NAME,
        f"{_PRIVATE_BUNDLE_MODULE_PREFIX}.{_LEGACY_INIT_NAME}",
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for binary_path in sorted(search_root.glob("sharpsv*.so")):
            for module_name in module_names:
                try:
                    return _load_extension(module_name, binary_path)
                except ImportError:
                    continue

        for binary_path in sorted(search_root.glob("mamnet*.so")):
            for module_name in module_names:
                try:
                    return _load_extension(module_name, binary_path)
                except ImportError:
                    continue

    raise ImportError(
        "SharpSV native backend could not be loaded from the package directory "
        "or the packaged native bundle."
    )


_backend = _load_backend()

g_d = _backend.g_d
c_cw = _backend.c_cw
c_cn = _backend.c_cn

__all__ = ["g_d", "c_cw", "c_cn"]
