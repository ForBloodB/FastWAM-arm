"""Final read-only stability check; accepts a fresh output directory only."""
import argparse
import json
from pathlib import Path
import time
from reflex.canbus import ReadOnlyCAN

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
c=ReadOnlyCAN();c.capture=[];states=[];start=time.monotonic()
try:
    for _ in range(100):
        c.poll(.1);states.append({str(k):dict(v) for k,v in c.latest_status.items()})
    probe={str(mid):{hex(p):c.read(mid,p) for p in
        ((0x7019,0x701B,0x701C,0x7005,0x700B,0x7028) if mid!=7 else
         (0x7019,0x701B,0x701C,0x7005,0x700B,0x7028,0x7017,0x7018))} for mid in range(1,8)}
    ok=all(len(s)==7 and all(v['fault_mask_raw']==0 and v['mode_bits_raw']==0 for v in s.values()) for s in states)
    result={'success':ok,'evidence':'real_read_only_final_status','enabled_by_script':False,
        'duration_s':time.monotonic()-start,'status_snapshots':len(states),'probe':probe,'last_status':states[-1]}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'status_snapshots.json').write_text(json.dumps(states))
    (out/'frames.json').write_text(json.dumps(c.capture));print(json.dumps(result))
finally:c.close()
raise SystemExit(0 if ok else 1)
