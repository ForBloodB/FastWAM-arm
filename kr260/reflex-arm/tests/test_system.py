import json
import struct
import zlib
import numpy as np
import pytest
from reflex.protocol import Packet,Receiver,HEADER
from reflex.safety import SafetyFSM,State,Envelope
from reflex.vision import locate_mask
from reflex.env import GraspSurrogate
from reflex.kinematics import quintic_trajectory


def test_protocol_roundtrip_and_crc():
    p=Packet(1,7,{'mode':'diagnostic'},timestamp_ns=55)
    decoded=Packet.decode(p.encode())
    assert decoded==p
    frame=bytearray(p.encode()); frame[20]^=1
    with pytest.raises(ValueError,match='CRC'): Packet.decode(frame)


def test_replay_order_and_receiver_clock():
    r=Receiver()
    data=Packet(4,1,{},ttl_ms=20,timestamp_ns=999999999999).encode()
    r.accept(data,100)
    with pytest.raises(ValueError): r.accept(data,101)
    with pytest.raises(ValueError): r.accept(Packet(3,1,{}).encode(),102)
    assert not r.expired(100+19_999_999)
    assert r.expired(100+20_000_000)


@pytest.mark.parametrize('bad',[b'',b'x'*5000,b'\0'*60])
def test_bad_frames(bad):
    with pytest.raises(ValueError): Packet.decode(bad)


def test_nonfinite_encoding_rejected():
    with pytest.raises(ValueError): Packet(1,1,{'x':float('nan')}).encode()


def test_untrusted_json_numeric_overflow_and_nesting():
    def frame(body):
        data=HEADER.pack(b'RFAX',1,1,0,1,1,1,100,len(body))+body
        return data+struct.pack('!I',zlib.crc32(data))
    with pytest.raises(ValueError): Packet.decode(frame(b'{"x":1e999}'))
    with pytest.raises(ValueError): Packet.decode(frame(b'{"x":'+b'['*32+b'0'+b']'*32+b'}'))


def test_base_plan_must_be_valid_before_residual():
    with pytest.raises(ValueError,match='Base plan'): Envelope().project([10,0,.1,0,6],[0]*5)


def test_motor_diagnostics_cannot_send_control():
    from reflex.canbus import read_request
    frame=read_request(1,0x7019)
    canid,dlc,data=struct.unpack('=IB3x8s',frame)
    assert canid==0x9100FD01 and dlc==8
    assert data==bytes.fromhex('1970000000000000')
    for param in [0x7006,0x700A,0x0006,0xFFFF]:
        with pytest.raises(ValueError): read_request(1,param)
    # Reading a configuration value is still a type-17 frame, never a write.
    for param in [0x7016,0x700B,0x7018]:
        raw=read_request(7,param)
        cid,dlc,data=struct.unpack('=IB3x8s',raw)
        assert (cid>>24)&31==17 and cid&255==7
        assert data[2:]==bytes(6)


def test_recorded_motor_type24_status_vector():
    from reflex.canbus import parse_status_frame
    frame=struct.pack('=IB3x8s',0x981007FD,8,bytes.fromhex('8004802c7fff012c'))
    result=parse_status_frame(frame)
    assert result['motor_id']==7 and result['host_id']==253
    assert result['type']==24 and result['fault_mask_raw']==16
    assert result['temperature_C']==30. and result['position_u16']==32772


def test_motion_test_amplitude_velocity_and_guards():
    import math
    from reflex.motion import j1_test_target,check_j1_sample
    t=np.linspace(0,6,601)
    q=np.array([j1_test_target(.1,x) for x in t])
    assert max(q)-.1==pytest.approx(math.radians(.5))
    assert q[0]==pytest.approx(q[-1])
    assert np.max(abs(np.diff(q)/np.diff(t)))<math.radians(.27)
    for kwargs in [{'duration':2},{'amplitude_deg':1}]:
        with pytest.raises(ValueError): j1_test_target(0,1,**kwargs)
    check_j1_sample(.1,0,.1,.1,48)
    with pytest.raises(ValueError): check_j1_sample(.13,0,.1,.1,48)
    with pytest.raises(ValueError): check_j1_sample(.1,.1,.1,.1,48)


@pytest.mark.parametrize('fault',['estop','limit','sensor_ok','command_ok'])
def test_safety_faults_latch(fault):
    f=SafetyFSM(100)
    f.tick(0,arm=True)
    assert not f.drive_enabled
    f.tick(1,heartbeat=True,arm=True)
    assert f.drive_enabled
    f.tick(2,**{fault: fault not in ['sensor_ok','command_ok']})
    assert f.state==State.FAULT
    f.tick(3,heartbeat=True,arm=True)
    assert not f.drive_enabled
    f.tick(4,heartbeat=True,reset=True)
    assert f.state==State.DISARMED
    f.tick(5,arm=True)
    assert f.drive_enabled


def test_watchdog_exact_deadline():
    f=SafetyFSM(100)
    f.tick(0,heartbeat=True,arm=True)
    f.tick(99)
    assert f.drive_enabled
    f.tick(100)
    assert f.state==State.FAULT
    f.tick(101,reset=True)
    assert f.state==State.FAULT


def test_residual_projection():
    e=Envelope()
    t=e.project([.3,0,.1,0,6],[999,-999,999,999,999])
    np.testing.assert_allclose(t,[.315,-.015,.11,.15,9.])
    with pytest.raises(ValueError): e.project([0]*5,[np.nan]*5)


def test_rgbd_transform():
    depth=np.ones((10,10))*500
    mask=np.ones_like(depth,bool)
    k=np.array([[100,0,4.5],[0,100,4.5],[0,0,1.]])
    transform=np.eye(4); transform[:3,3]=[.2,.1,0]
    pose=locate_mask(depth,mask,k,transform)
    np.testing.assert_allclose(pose['position_m'],[.2,.1,.5],atol=1e-9)
    depth[:]=0
    with pytest.raises(ValueError): locate_mask(depth,mask,k,transform)


def test_environment_gym_contract_and_seed():
    from stable_baselines3.common.env_checker import check_env
    check_env(GraspSurrogate())
    a,b=GraspSurrogate(),GraspSurrogate()
    x,_=a.reset(seed=4); y,_=b.reset(seed=4)
    np.testing.assert_array_equal(x,y)
    assert a.step(np.zeros(5))[1:]==b.step(np.zeros(5))[1:]
    with pytest.raises(RuntimeError): a.step(np.zeros(5))


def test_trajectory_endpoints_and_continuity():
    t,q=quintic_trajectory([0,0],[.1,-.1],3.,200)
    np.testing.assert_allclose(q[0],[0,0])
    np.testing.assert_allclose(q[-1],[.1,-.1])
    assert np.max(abs(np.diff(q,axis=0)/np.diff(t)[:,None]))<.063


def test_inference_roundtrip_and_replay(tmp_path):
    from stable_baselines3 import SAC
    import torch
    torch.set_num_threads(1)
    env=GraspSurrogate()
    model=SAC('MlpPolicy',env,buffer_size=100,batch_size=8,learning_starts=8,
              policy_kwargs={'net_arch':[16,16]},device='cpu',seed=3)
    model.learn(24)
    obs,_=env.reset(seed=6)
    before=model.predict(obs,deterministic=True)[0]
    model.save(tmp_path/'model')
    loaded=SAC.load(tmp_path/'model.zip',env=GraspSurrogate(),device='cpu')
    np.testing.assert_array_equal(before,loaded.predict(obs,deterministic=True)[0])
    model.save_replay_buffer(tmp_path/'replay.pkl')
    loaded.load_replay_buffer(tmp_path/'replay.pkl')
    assert loaded.replay_buffer.size()==24


def test_udp_bridge_process_and_rejections(tmp_path):
    import socket
    import subprocess
    import sys
    import time
    probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]; probe.close()
    log=tmp_path/'packets.jsonl'
    process=subprocess.Popen([sys.executable,'-m','reflex.bridge','--port',str(port),'--log',str(log)],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    client=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    client.settimeout(.1)
    try:
        reply=None
        for i in range(40):
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            client.sendto(Packet(i+1,1,{'mode':'diagnostic'},timestamp_ns=1).encode(),('127.0.0.1',port))
            try:
                reply=Packet.decode(client.recv(8192)); break
            except socket.timeout: pass
        assert reply is not None
        assert reply.payload['mode']=='NO_ACTUATORS'
        assert reply.payload['state']==0
        client.sendto(b'corrupt',('127.0.0.1',port))
        client.sendto(Packet(1000,1,{'mode':'move'}).encode(),('127.0.0.1',port))
        time.sleep(.1)
    finally:
        client.close(); process.terminate(); process.communicate(timeout=5)
    rows=[json.loads(s) for s in log.read_text().splitlines()]
    assert sum(not r['accepted'] for r in rows)>=2
