"""KR260 telemetry, shadow mapping and explicitly started bounded J1 following."""
import argparse,csv,fcntl,io,json,math,os,sys,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from engine import Engine,Problem
from follow_control import Follow
from leader_reader import LeaderReader


class State:
    def __init__(self,out,hardware=True):
        self.lock=threading.RLock();self.engine=Engine();self.out=out;out.mkdir(parents=True,exist_ok=True)
        self.stop=threading.Event();self.hardware=hardware;self.threads=[]
        self.read_requests=0;self.pause_can=threading.Event();self.recheck=threading.Event();self.follow=Follow(self)
        if (out/'mapping.json').is_file():
            try:
                self.engine.update_mapping(json.loads((out/'mapping.json').read_text()))
                self.engine.event('restored','已恢复上次映射；需要重新记录参考姿态')
            except Exception:self.engine.event('error','上次映射无效，使用默认映射并保持只读')
        if hardware:
            self.threads=[threading.Thread(target=self.leader,daemon=True),threading.Thread(target=self.follower,daemon=True)]
            for t in self.threads:t.start()

    def leader(self):
        while not self.stop.is_set():
            try:
                with open('/tmp/reflex-leader-read.lock','w') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    bus=LeaderReader()
                    try:
                        self.recheck.clear()
                        while not self.stop.is_set() and not self.recheck.is_set():
                            began=time.monotonic();rows=[];errors=[]
                            for sid in range(7):
                                try:
                                    rows.append(bus.read(sid))
                                except Exception as exc:errors.append(f'ID{sid}: {exc}')
                            with self.lock:self.engine.ingest_leader(rows,'; '.join(errors) or None)
                            self.stop.wait(max(0.,.04-(time.monotonic()-began)))
                    finally:bus.close()
            except Exception as exc:
                with self.lock:self.engine.leader_error=str(exc);self.engine.tick()
                self.stop.wait(1.)

    def follower(self):
        from reflex.canbus import ReadOnlyCAN
        while not self.stop.is_set():
            if self.pause_can.is_set():self.stop.wait(.05);continue
            rows=[];errors=[]
            try:
                with open('/tmp/reflex-can-actuation.lock','a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    bus=ReadOnlyCAN()
                    try:
                        for mid in range(1,8):
                            if self.stop.is_set():break
                            pos=bus.read(mid,0x7019,timeout=.04);self.read_requests+=1
                            vel=bus.read(mid,0x701B,timeout=.04);self.read_requests+=1
                            status=bus.latest_status.get(mid,{})
                            rows.append(dict(id=mid,position_deg=math.degrees(pos),velocity_deg_s=math.degrees(vel),
                                mode_bits=status.get('mode_bits_raw'),faults=status.get('faults',[]),temperature_C=status.get('temperature_C')))
                    finally:bus.close()
            except BlockingIOError:errors.append('CAN控制锁正由其他程序使用，暂停从臂查询')
            except Exception as exc:errors.append(str(exc))
            with self.lock:self.engine.ingest_follower(rows,'; '.join(errors) or None)
            self.stop.wait(.55)

    def apply(self,path,data):
        with self.lock:
            if path.startswith('/api/follow/'):
                if path=='/api/follow/start':
                    if not self.hardware:raise Problem('无实体控制后端')
                    self.engine.stop('开始J1联动，已停止目标预览')
                    token=self.follow.start(data)
                    return dict(self.status(),control_token=token)
                if path=='/api/follow/heartbeat':self.follow.heartbeat(data.get('token'))
                elif path=='/api/follow/stop':self.follow.stop()
                else:raise KeyError(path)
                return self.status()
            if self.follow.snapshot()['active']:raise Problem('跟随期间不能修改映射、参考或预览')
            if path=='/api/leader/recheck':
                self.engine.stop('重新检查主臂；请保持小主臂静止1秒')
                self.engine.leader={};self.engine.reference=None;self.recheck.set()
            elif path=='/api/reference':self.engine.capture()
            elif path=='/api/mapping':
                self.engine.update_mapping(data.get('joints'))
                pending=self.out/'mapping.json.tmp';pending.write_text(json.dumps(self.engine.mapping,ensure_ascii=False));pending.replace(self.out/'mapping.json')
            elif path=='/api/preview/start':self.engine.start()
            elif path=='/api/preview/stop':self.engine.stop()
            else:raise KeyError(path)
            snapshot=self.engine.snapshot()
            with (self.out/'actions.jsonl').open('a') as f:
                f.write(json.dumps(dict(time_ns=time.time_ns(),path=path,reference=snapshot['reference'],mapping=self.engine.mapping,motor_commands=0),ensure_ascii=False)+'\n')
            return snapshot

    def status(self):
        with self.lock:
            follow=self.follow.snapshot();snap=self.engine.snapshot()
            snap.update(can_parameter_read_requests=self.read_requests,follow=follow,
                motion_commands=follow['motion_commands'],hardware_follow_enabled=self.follow.ready,
                hardware_follow_reason='J1受限跟随：±1°，0.25°/秒，1.0 Nm；其他轴待逐轴校核。',
                scope='真实主从角度 + 目标预览 + J1单轴实体跟随')
            if follow['active'] and 'target_deg' in follow:
                snap['joints'][0]['target_delta_deg']=follow['target_deg']-follow['follower_reference_deg']
                snap['joints'][0]['target_absolute_deg']=follow['target_deg']
            return snap


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def reply(self,value,status=200,ctype='application/json; charset=utf-8',extra=None):
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode() if not isinstance(value,bytes) else value
            self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path in ['/api/status','/api/health']:
                self.reply(state.status());return
            if self.path=='/api/export.csv':
                snap=state.status();f=io.StringIO();writer=csv.writer(f)
                writer.writerow(['leader_id','follower_id','joint','leader_raw_deg','leader_filtered_deg','follower_actual_deg','preview_delta_deg','preview_absolute_deg','direction','gain','verified','motion_commands'])
                for r in snap['joints']:
                    writer.writerow([r['leader_id'],r['follower_id'],r['name'],(r['leader'] or {}).get('raw_deg'),(r['leader'] or {}).get('filtered_deg'),
                        (r['follower'] or {}).get('position_deg'),r['target_delta_deg'],r['target_absolute_deg'],r['direction'],r['gain'],r['verified'],snap['motion_commands']])
                self.reply(('\ufeff'+f.getvalue()).encode(),ctype='text/csv; charset=utf-8',extra={'Content-Disposition':'attachment; filename="leader-follower-snapshot.csv"'});return
            self.reply({'error':'Not found'},404)
        def do_POST(self):
            if self.headers.get('X-Reflex-Proxy')!='leader-web':self.reply({'error':'Proxy required'},403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise Problem('请求大小无效')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise Problem('需要JSON对象')
                self.reply(state.apply(self.path,data))
            except KeyError:self.reply({'error':'Not found'},404)
            except (ValueError,TypeError) as exc:self.reply({'error':str(exc)},409)
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=9010);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    state=State(args.out);server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(state));server.daemon_threads=True
    import signal
    def shutdown(*_):
        state.follow.stop('服务停止')
        if state.follow.thread:state.follow.thread.join(3.)
        state.stop.set()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    try:server.serve_forever(poll_interval=.2)
    finally:shutdown();server.server_close()
