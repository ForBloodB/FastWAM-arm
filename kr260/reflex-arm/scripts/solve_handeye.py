"""Solve accepted pair files into a candidate, without activating hardware."""
import argparse
import json
from pathlib import Path
from reflex.calibration import eye_in_hand_candidate

p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
files=sorted(Path(a.session).glob('*/pair.json'));samples=[];result={'quality_gate_passed':False}
try:
    metadata=set()
    for path in files:
        sample=json.loads(path.read_text())
        if not sample.get('pair_accepted'):continue
        samples.append(sample)
        metadata.add((sample['urdf_sha256'],sample['board_sha256'],sample['camera_serial'],sample['tool_frame']))
    if len(metadata)>1:raise ValueError('Mixed camera, board or robot geometry within session')
    result=eye_in_hand_candidate(samples)
    result['source_pair_files']=[str(p) for p in files if json.loads(p.read_text()).get('pair_accepted')]
except Exception as exc:result['error']=repr(exc)
(out/'candidate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
raise SystemExit(0 if result['quality_gate_passed'] else 1)
