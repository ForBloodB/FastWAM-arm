"""Bounded query-only comparison; run while the web reader is stopped."""
import fcntl,json,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'services/leader-follow'),str(root)]
from leader_reader import LeaderReader
from reflex.canbus import ReadOnlyCAN
from motorbridge_smart_servo import FashionStarServo

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);report={}
with open('/tmp/reflex-leader-read.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with FashionStarServo('/dev/ttyUSB0',baudrate=1000000) as sdk:
        samples=[]
        for _ in range(35):
            rows=[]
            for sid in range(7):
                s=sdk.read_angle(sid,True);rows.append(dict(id=sid,raw=s.raw_deg,reliable=s.reliable))
            samples.append(rows);time.sleep(.04)
        report['sdk_samples']=samples
    bus=LeaderReader();samples=[]
    try:
        for _ in range(60):
            samples.append([bus.read(sid) for sid in range(7)]);time.sleep(.04)
    finally:bus.close()
    report['direct_samples']=samples
with open('/tmp/reflex-can-actuation.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bus=ReadOnlyCAN()
    try:
        report['parameters']={i:{hex(p):bus.read(i,p,.05) for p in (0x7019,0x701B,0x701C,0x7005,0x700B,0x7018,0x7028)} for i in range(1,8)}
        bus.poll(.1);report['status']=bus.latest_status
    finally:bus.close()
(out/'readonly_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(parameters=report['parameters'],status=report['status'],
    sdk_reliable_counts=[sum(r[i]['reliable'] for r in report['sdk_samples']) for i in range(7)],
    direct_reliable_last30=[sum(r[i]['reliable'] for r in report['direct_samples'][-30:]) for i in range(7)]),indent=2))
