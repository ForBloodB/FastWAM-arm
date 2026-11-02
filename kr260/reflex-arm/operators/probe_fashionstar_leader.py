"""Query-only FashionStar probe. No unlock, zero/reset, target or CAN API."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.metadata,json,math,time
from motorbridge_smart_servo import FashionStarServo


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',default='/dev/ttyUSB0');p.add_argument('--baudrate',type=int,default=1000000)
    p.add_argument('--seconds',type=float,default=4.);p.add_argument('--out',required=True);args=p.parse_args()
    assert 0<args.seconds<=30 and args.baudrate==1000000
    out=Path(args.out);assert not out.exists()
    ids=list(range(7));rows=[];pings=[]
    with open('/tmp/reflex-leader-read.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with FashionStarServo(args.port,baudrate=args.baudrate) as bus:
            for sid in ids:
                try:pings.append(dict(id=sid,responded=bus.ping(sid)))
                except Exception as exc:pings.append(dict(id=sid,responded=False,error=str(exc)))
            start=time.monotonic()
            while time.monotonic()-start<args.seconds:
                for sid in ids:
                    try:
                        sample=bus.read_angle(sid,multi_turn=True)
                        raw=float(sample.raw_deg);filtered=float(sample.filtered_deg)
                        assert math.isfinite(raw) and math.isfinite(filtered)
                        rows.append(dict(t_s=time.monotonic()-start,id=sid,raw_deg=raw,filtered_deg=filtered,reliable=bool(sample.reliable)))
                    except Exception as exc:rows.append(dict(t_s=time.monotonic()-start,id=sid,error=str(exc)))
                time.sleep(.1)
    available=sorted({r['id'] for r in rows if 'raw_deg' in r})
    summary=[]
    for sid in ids:
        rr=[r for r in rows if r['id']==sid];valid=[r for r in rr if 'raw_deg' in r]
        summary.append(dict(id=sid,samples=len(valid),errors=sum('error' in r for r in rr),
            first_deg=valid[0]['raw_deg'] if valid else None,last_deg=valid[-1]['raw_deg'] if valid else None,
            span_deg=max(r['raw_deg'] for r in valid)-min(r['raw_deg'] for r in valid) if valid else None))
    result=dict(port=args.port,baudrate=args.baudrate,expected_ids=ids,available_ids=available,pings=pings,
        summary=summary,rows=rows,sdk_version=importlib.metadata.version('motorbridge-smart-servo'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transmitted_operations=['ping','read_angle(multi_turn=True)'],motor_action_commands=0,zero_or_counter_writes=0,can_opened=False,
        model_scope='FashionStar protocol and IDs can identify compatible bus; exact arm geometry still needs visual/model confirmation')
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
