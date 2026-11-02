"""Fixed web MIT gains for the explicitly authorized unloaded wrist baseline."""
import math
import struct


def wrist_mit_frame(joint,position,effort=True,damping_comparison=False):
    if joint not in (5,6) or not math.isfinite(position):
        raise ValueError('Web MIT baseline is scoped to J5/J6')
    limit=1.57 if joint==5 else 3.14
    if not -limit<=position<=limit:raise ValueError('Wrist MIT target outside joint interval')
    if damping_comparison and joint!=6:
        raise ValueError('Damping comparison is scoped to J6')
    kp,kd=(30.,2. if damping_comparison else 1.) if effort else (0.,0.)
    enc=lambda value,lo,hi:int((value-lo)/(hi-lo)*65535)
    payload=struct.pack('>4H',enc(position,-4*math.pi,4*math.pi),32767,
        enc(kp,0.,500.),enc(kd,0.,5.))
    # Manufacturer RS00 ranges; requested feed-forward velocity and torque=0.
    return struct.pack('=IB3x8s',0x80000000|(1<<24)|(32767<<8)|joint,8,payload)
