import ctypes
from pathlib import Path
import numpy as np
_lib = None

def force_roll(model, data, a, b, cap, start, total, n):
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build/slip_control/libslip_control.so'))
        _lib.reflex_force_roll.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.reflex_force_roll.restype = ctypes.c_int
    a = np.ascontiguousarray(a, dtype=np.float64); b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != (6,) or b.shape != (6,) or not np.isfinite(np.r_[a,b,cap]).all(): raise ValueError('Invalid control')
    if not 1.5 <= cap <= 28 or total <= 0 or start < 0 or n < 0 or start+n > total: raise ValueError('Control bounds')
    if np.any(a < model.jnt_range[:6,0]) or np.any(a > model.jnt_range[:6,1]) or np.any(b < model.jnt_range[:6,0]) or np.any(b > model.jnt_range[:6,1]): raise ValueError('Joint bounds')
    status = _lib.reflex_force_roll(model._address, data._address, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), cap, start, total, n)
    if status: raise RuntimeError(f'Simulation diverged: {status}')
