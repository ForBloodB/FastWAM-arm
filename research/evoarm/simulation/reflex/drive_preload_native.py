import ctypes
from pathlib import Path
import numpy as np
_lib=None

def roll(m,d,a,b,ga,gb,cap,start,total,n,mode,fa,fb,quintic,og,left,right,table,budget):
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build/drive_preload/libdrive_preload.so'))
        ptr=ctypes.POINTER(ctypes.c_double);I=ctypes.c_int;D=ctypes.c_double
        _lib.drive_preload_roll.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ptr,ptr,D,D,D,I,I,I,I,D,D,I,I,I,I,I,D,ptr]
        _lib.drive_preload_roll.restype=I
    a=np.ascontiguousarray(a,dtype=float);b=np.ascontiguousarray(b,dtype=float);stats=np.zeros(7)
    if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite(np.r_[a,b,ga,gb,cap,fa,fb,budget]).all():raise ValueError('Invalid native args')
    if not (0<=ga<=.05 and 0<=gb<=.05 and 0<cap<=28 and 0<=fa<=28 and 0<=fb<=28 and start>=0 and n>0 and start+n<=total):raise ValueError('Native bounds')
    if np.any(a<m.jnt_range[:6,0]) or np.any(a>m.jnt_range[:6,1]) or np.any(b<m.jnt_range[:6,0]) or np.any(b>m.jnt_range[:6,1]):raise ValueError('Joint bounds')
    ptr=ctypes.POINTER(ctypes.c_double)
    rc=_lib.drive_preload_roll(m._address,d._address,a.ctypes.data_as(ptr),b.ctypes.data_as(ptr),ga,gb,cap,start,total,n,int(mode),fa,fb,int(quintic),og,left,right,table,budget,stats.ctypes.data_as(ptr))
    if rc:raise RuntimeError('Native simulation diverged:'+str(rc))
    return stats
