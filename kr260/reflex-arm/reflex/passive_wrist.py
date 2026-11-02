"""Decode J6/RS00 CAN evidence only; no socket or actuator functions."""
import math
import struct


def decode_j6_frame(raw):
    if len(raw)!=16:
        raise ValueError('Classic SocketCAN frame required')
    cid,dlc,data=struct.unpack('=IB3x8s',raw)
    if not cid&0x80000000 or cid&0x60000000 or dlc!=8:
        return None
    eid=cid&0x1fffffff;kind=(eid>>24)&31
    scale=lambda value,lo,hi:lo+(hi-lo)*value/65535.
    if kind in (1,3,4,18) and eid&255==6:
        result={'kind':kind,'joint':6,'category':'control'}
        if kind==1:
            pos,vel,kp,kd=struct.unpack('>4H',data)
            result.update(position_rad=scale(pos,-4*math.pi,4*math.pi),
                velocity_feedforward_rad_s=scale(vel,-33.,33.),
                torque_feedforward_Nm=scale((eid>>8)&65535,-14.,14.),
                kp=scale(kp,0.,500.),kd=scale(kd,0.,5.))
        elif kind==18:
            param=struct.unpack_from('<H',data)[0]
            value=data[4] if param==0x7005 else struct.unpack_from('<I' if param==0x7028 else '<f',data,4)[0]
            result.update(parameter=hex(param),value=value)
        return result
    if kind in (2,24) and (eid>>8)&255==6:
        pos,vel,torque,temp=struct.unpack('>4H',data)
        return {'kind':kind,'joint':6,'category':'status','fault_mask_raw':(eid>>16)&63,
            'mode_bits_raw':(eid>>22)&3,'position_rad':scale(pos,-4*math.pi,4*math.pi),
            'position_u16':pos,'temperature_C':temp/10.}
    if kind==17 and (eid>>8)&255==6 and not (eid>>16)&255:
        param=struct.unpack_from('<H',data)[0]
        if param in (0x7019,0x701B,0x701A):
            return {'kind':kind,'joint':6,'category':'parameter_reply','parameter':hex(param),
                'value':struct.unpack_from('<f',data,4)[0]}
    return None
