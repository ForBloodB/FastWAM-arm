"""Restricted diagnostic actuator path for explicitly selected J1 or J7 only.

RS06/RS00 private protocol: Seeed-distributed manufacturer manual 251112,
pages 19-25. No zero, save, firmware, or other-joint writes. J7 may temporarily
switch between MIT and CSP while disabled; its original mode is restored.
"""
import math
import struct
import time


def mit_frame(mid,position,kp,kd,tau=0.):
    if mid not in (1,7):raise ValueError('Only J1/J7 diagnostic commands supported')
    if not all(math.isfinite(v) for v in (position,kp,kd,tau)):raise ValueError('Nonfinite MIT target')
    if abs(tau)>.05 or (mid==7 and tau!=0):raise ValueError('Only <=0.05 Nm J1 timeout diagnostic effort')
    kpmax,kdmax=(5000.,100.) if mid==1 else (500.,5.)
    if not -4*math.pi<position<4*math.pi or not 0<=kp<=(200 if mid==1 else 10) or not 0<=kd<=3:
        raise ValueError('Diagnostic MIT envelope exceeded')
    def encode(v,lo,hi):return int((v-lo)/(hi-lo)*65535)
    data=struct.pack('>4H',encode(position,-4*math.pi,4*math.pi),32767,
                     encode(kp,0,kpmax),encode(kd,0,kdmax))
    torque=int((tau/36+1)*32767) if mid==1 else 32767
    return struct.pack('=IB3x8s',0x80000000|(1<<24)|(torque<<8)|mid,8,data)


class DiagnosticMotor:
    def __init__(self,reader,mid):
        if mid not in (1,7):raise ValueError('Explicitly scoped to J1/J7')
        self.reader=reader;self.mid=mid;self.tx=[]

    def _send(self,frame):
        self.reader.sock.send(frame)
        self.tx.append({'ns':time.monotonic_ns(),'frame_hex':frame.hex()})

    def _command(self,kind,data=b'\0'*8):
        self._send(struct.pack('=IB3x8s',0x80000000|(kind<<24)|(253<<8)|self.mid,8,data))

    def disable(self):self._command(4)
    def clear_stall(self):self._command(4,b'\1'+b'\0'*7)
    def enable(self):self._command(3)
    def mit(self,position,kp=50.,kd=3.,tau=0.):self._send(mit_frame(self.mid,position,kp,kd,tau))

    def set_volatile(self,param,value):
        if param==0x700B:
            if not 0<value<=36:raise ValueError('Invalid torque limit')
            dtype='f'
        elif param==0x7028:
            if not isinstance(value,int) or not 0<=value<=100000:raise ValueError('Invalid watchdog count')
            dtype='I'
        elif self.mid==7 and param in (0x7016,0x7017,0x7018,0x7005):
            if param==0x7005:
                if value not in (0,5):raise ValueError('Only MIT/CSP diagnostic modes supported')
                dtype='b'
            else:
                if not math.isfinite(value):raise ValueError('Nonfinite gripper parameter')
                dtype='f'
        else:raise ValueError('Only torque/watchdog volatile writes supported')
        payload=struct.pack('<H',param)+b'\0\0'+struct.pack('<'+dtype,value).ljust(4,b'\0')
        self._command(18,payload)
        actual=self.reader.read(self.mid,param,timeout=.03)
        # loc_ref is reseeded from quantized encoder position while disabled.
        tolerance=.0005 if param==0x7016 else 1e-6
        if not math.isclose(actual,value,rel_tol=1e-5,abs_tol=tolerance):
            raise RuntimeError(f'Write readback mismatch {hex(param)} requested={value} actual={actual}')

    def status(self,max_age_s=.08):
        self.reader.drain()
        status=self.reader.latest_status.get(self.mid)
        if status is None or (time.monotonic_ns()-status['rx_ns'])/1e9>max_age_s:
            raise RuntimeError('Stale or missing motor status')
        return status
