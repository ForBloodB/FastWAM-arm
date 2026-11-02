import ctypes
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];_lib=None
def roll(model,data,a,b,ga,gb,cap,start,total,n,object_geom,left,right,table,budget):
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(str(ROOT/'build/safe_contact/libsafe_contact.so'))
        _lib.safe_contact_roll.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]
        _lib.safe_contact_roll.restype=ctypes.c_int
    a=np.ascontiguousarray(a,dtype=float);b=np.ascontiguousarray(b,dtype=float);stats=np.zeros(5)
    if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite(np.r_[a,b,ga,gb,cap,budget]).all():raise ValueError('Invalid arguments')
    if total<=0 or start<0 or n<=0 or start+n>total or not 0<=ga<=.05 or not 0<=gb<=.05 or not 0<cap<=28:raise ValueError('Bounds')
    if np.any(a<model.jnt_range[:6,0]) or np.any(a>model.jnt_range[:6,1]) or np.any(b<model.jnt_range[:6,0]) or np.any(b>model.jnt_range[:6,1]):raise ValueError('Joint range')
    status=_lib.safe_contact_roll(model._address,data._address,a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ga,gb,cap,start,total,n,object_geom,left,right,table,budget,stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if status:raise RuntimeError('Native simulation diverged:'+str(status))
    return stats
