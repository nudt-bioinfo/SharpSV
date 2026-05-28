import importlib


try:
    _backend = importlib.import_module("sharpsv_backend")
except ImportError as exc:
    raise ImportError(
        "SharpSV native backend is missing. Expected the local 'sharpsv_backend' loader module."
    ) from exc


g_d = _backend.g_d
c_cw = _backend.c_cw
c_cn = _backend.c_cn

__all__ = ["g_d", "c_cw", "c_cn"]
