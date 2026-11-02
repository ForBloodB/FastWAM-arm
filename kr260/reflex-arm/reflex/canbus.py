"""Read-only RobStride SocketCAN diagnostics using the manufacturer's protocol.

No enable, motion, zero, configuration-write or save operation is implemented.
Parameter whitelist prevents accidental writes through the diagnostic API.
"""
import math
import socket
import struct
import time

READABLE={0x7019:'f',0x701B:'f',0x701C:'f',0x701A:'f',0x700B:'f',0x7005:'b',
          0x7016:'f',0x7017:'f',0x7018:'f',0x701E:'f',0x701F:'f',0x7020:'f',0x7021:'f',
          0x7024:'f',0x7025:'f',0x7028:'I',0x7029:'B',0x3016:'f'}
FAULT_NAMES=('undervoltage','overcurrent','overtemperature','magnetic_encoder',
             'stall_overload','uncalibrated')


def parse_status_frame(frame):
    """Accept type 2 and type 24 feedback, preserving unverified fault bits raw."""
    if len(frame)!=16: raise ValueError('SocketCAN classic frame required')
    canid,dlc,data=struct.unpack('=IB3x8s',frame)
    eid=canid&0x1fffffff
    kind=(eid>>24)&0x1f
    if not canid&0x80000000 or canid&0x60000000 or dlc!=8 or kind not in (2,24):
        raise ValueError('Not a supported extended status frame')
    position,velocity,torque,temperature=struct.unpack('>4H',data)
    return {'motor_id':(eid>>8)&255,'host_id':eid&255,'type':kind,
            'fault_mask_raw':(eid>>16)&63,'mode_bits_raw':(eid>>22)&3,
            'faults':[name for i,name in enumerate(FAULT_NAMES) if eid&(1<<(16+i))],
            'position_u16':position,'velocity_u16':velocity,'torque_u16':torque,
            'temperature_C':temperature/10.,'can_id_hex':hex(eid)}


def read_request(motor_id,param,host_id=0xFD):
    if not 1<=motor_id<=7 or param not in READABLE or not 0<=host_id<=255:
        raise ValueError('Only whitelisted B601 parameter reads permitted')
    canid=0x80000000|(17<<24)|(host_id<<8)|motor_id
    return struct.pack('=IB3x8s',canid,8,struct.pack('<H',param)+b'\0'*6)


class ReadOnlyCAN:
    def __init__(self,channel='can0',host_id=0xFD):
        if not isinstance(host_id,int) or not 0<=host_id<=255:raise ValueError('Invalid diagnostic host ID')
        self.host_id=host_id
        self.sock=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW)
        self.sock.bind((channel,)); self.sock.settimeout(.1)
        self.log=[]
        self.latest_status={}
        self.last_reply=None
        self.failed=False
        self.capture=None

    def _receive(self):
        raw=self.sock.recv(16)
        ns=time.monotonic_ns()
        if self.capture is not None:self.capture.append({'rx_ns':ns,'frame_hex':raw.hex()})
        try:
            status=parse_status_frame(raw)
            self.latest_status[status['motor_id']]={'rx_ns':ns,**status}
        except ValueError: pass
        return raw,ns

    def drain(self):
        self.sock.setblocking(False)
        try:
            while True:self._receive()
        except BlockingIOError:pass

    def poll(self,duration=.05):
        end=time.monotonic()+duration
        self.sock.settimeout(.002)
        while time.monotonic()<end:
            try:self._receive()
            except socket.timeout:pass
        return self.latest_status

    def read(self,motor_id,param,timeout=.3):
        if self.failed:raise RuntimeError('Read session failed; reopen after timeout')
        self.drain()
        self.sock.settimeout(min(.01,timeout))
        frame=read_request(motor_id,param,self.host_id)
        self.sock.send(frame)
        self.log.append({'tx_ns':time.monotonic_ns(),'motor':motor_id,'type':17,'parameter':hex(param)})
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            try: raw,rx_ns=self._receive()
            except socket.timeout: continue
            canid,dlc,data=struct.unpack('=IB3x8s',raw)
            eid=canid&0x1fffffff
            if not canid&0x80000000 or canid&0x60000000 or dlc!=8: continue
            if (eid>>24)&0x1f!=17 or (eid>>8)&0xff!=motor_id or eid&0xff!=self.host_id: continue
            if struct.unpack('<H',data[:2])[0]!=param: continue
            if (eid>>16)&0xff: raise RuntimeError(f'Parameter read error for motor {motor_id}')
            value=struct.unpack_from('<'+READABLE[param],data,4)[0]
            if not math.isfinite(value): raise ValueError('Nonfinite motor parameter')
            self.last_reply={'rx_ns':rx_ns,'can_id':hex(eid),'data_hex':data.hex(),
                             'motor':motor_id,'parameter':param,'value':value}
            return value
        self.failed=True
        raise TimeoutError(f'Motor {motor_id} parameter {hex(param)} did not respond')

    def close(self): self.sock.close()
