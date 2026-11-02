import ctypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
_lib=None
def roll(model,data,a,b,ga,gb,cap,start,total,n):
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(str(ROOT/'build/official_control/libreflex_control.so'))
        _lib.reflex_roll.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
                                  ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        _lib.reflex_roll.restype=ctypes.c_int
    a=np.ascontiguousarray(a,dtype=np.float64);b=np.ascontiguousarray(b,dtype=np.float64)
    if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite(np.r_[a,b,ga,gb,cap]).all():raise ValueError('Invalid controller arguments')
    if total<=0 or start<0 or n<0 or start+n>total:raise ValueError('Invalid substep interval')
    if not 0<=ga<=.05 or not 0<=gb<=.05 or not 0<cap<=30:raise ValueError('Gripper bounds')
    for i in range(6):
        if a[i]<model.jnt_range[i,0] or a[i]>model.jnt_range[i,1] or b[i]<model.jnt_range[i,0] or b[i]>model.jnt_range[i,1]:raise ValueError('Joint bounds')
    status=_lib.reflex_roll(model._address,data._address,a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ga,gb,cap,start,total,n)
    if status:raise RuntimeError('Native simulation diverged: '+str(status))
