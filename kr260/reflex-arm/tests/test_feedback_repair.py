import math
import struct
import pytest
from reflex.motion import VelocityGuard,can_timeout_raw
from reflex.canbus import parse_status_frame
from reflex.diagnostic_motor import mit_frame


def test_watchdog_units_and_recorded_fault():
    assert can_timeout_raw(100)==2000
    with pytest.raises(ValueError):can_timeout_raw(5)
    frame=struct.pack('=IB3x8s',0x981007FD,8,bytes.fromhex('8004802c7fff012c'))
    assert parse_status_frame(frame)['faults']==['stall_overload']
    # Correctly distinguish Run mode from a fault bit.
    run=struct.pack('=IB3x8s',0x828001FD,8,b'\x80\0'*4)
    assert parse_status_frame(run)['mode_bits_raw']==2
    assert parse_status_frame(run)['faults']==[]


def test_guard_allows_recorded_isolated_spike_but_stops_real_motion():
    g=VelocityGuard()
    for i in range(20):
        # Stationary encoder quantization, raw spike magnitude from CAN capture.
        g.update(i*10_000_000,.0367+(i%2)*.000043,-.231159 if i==8 else 0.)
    for mode in ('encoder','raw'):
        g=VelocityGuard()
        with pytest.raises(ValueError):
            for i in range(20):
                g.update(i*10_000_000,math.radians(i*.1) if mode=='encoder' else 0.,
                         0. if mode=='encoder' else math.radians(10.))
    with pytest.raises(ValueError):g.update(1_000_000_000,0.,0.)
    # Larger J7 estimator noise does not change the encoder speed limit.
    g=VelocityGuard(raw_limit_deg_s=12)
    for i in range(20):g.update(i*10_000_000,0.,math.radians(6 if i%2 else -6))
    with pytest.raises(ValueError,match='Encoder'):
        for i in range(20,30):g.update(i*10_000_000,math.radians((i-19)*.1),0.)


def test_scoped_mit_encoding_gains_and_zero_feedforward():
    for mid,kpm,kdm in ((1,5000,100),(7,500,5)):
        raw=mit_frame(mid,.04,10.,.5)
        cid,dlc,data=struct.unpack('=IB3x8s',raw)
        pos,vel,kp,kd=struct.unpack('>4H',data)
        assert cid&255==mid and (cid>>8)&65535==32767 and dlc==8
        assert abs((pos/65535*8*math.pi-4*math.pi)-.04)<.0004
        assert vel==32767
        assert abs(kp/65535*kpm-10.)<.08
        assert abs(kd/65535*kdm-.5)<.002
    with pytest.raises(ValueError):mit_frame(2,0.,10.,1.)
    with pytest.raises(ValueError):mit_frame(1,float('nan'),10.,1.)


def test_parameter_read_discards_queued_reply(monkeypatch):
    import socket
    import reflex.canbus as cb
    def reply(v):return struct.pack('=IB3x8s',0x910001FD,8,struct.pack('<H2xf',0x7019,v))
    class Fake:
        blocking=True
        queue=[reply(.1)]
        def bind(self,*a):pass
        def settimeout(self,*a):self.blocking=True
        def setblocking(self,value):self.blocking=value
        def recv(self,*a):
            if self.queue:return self.queue.pop(0)
            if not self.blocking:raise BlockingIOError()
            raise socket.timeout()
        def send(self,raw):
            assert (struct.unpack_from('=I',raw)[0]>>24)&31==17
            self.queue.append(reply(.2))
        def close(self):pass
    monkeypatch.setattr(cb.socket,'socket',lambda *a:Fake())
    reader=cb.ReadOnlyCAN()
    assert reader.read(1,0x7019)==pytest.approx(.2)
