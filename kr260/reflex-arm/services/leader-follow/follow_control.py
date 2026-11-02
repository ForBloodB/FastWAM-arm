"""Bounded J1 commissioning controller. Other joints have no actuation path.

The selected leader is sampled locally on KR260; HTTP never supplies targets.
J1 uses the RS06 MIT path (200/3, <=1.0 Nm, explicitly approved 2026-10-31). No zero,
mode, persistent parameter or other-joint enable/disable/target writes.
"""
import fcntl,hashlib,json,math,secrets,struct,threading,time
from pathlib import Path
from reflex.canbus import ReadOnlyCAN
from reflex.diagnostic_motor import DiagnosticMotor
from reflex.motion import VelocityGuard

MAX_OFFSET_DEG=1.
SPEED_DEG_S=.25
TORQUE_LIMIT_NM=1.

def capability_hash():
    folder=Path(__file__).parent
    return hashlib.sha256(b''.join((folder/name).read_bytes() for name in ('follow_control.py','leader_reader.py','board_server.py','engine.py'))).hexdigest()


class Target:
    def __init__(self,leader,position,direction=1,gain=1.):
        if direction not in (-1,1) or not .05<=gain<=1.:raise ValueError('J1方向/比例无效')
        self.leader=leader;self.anchor=position;self.target=position;self.direction=direction;self.gain=gain
    def update(self,value,dt):
        if not all(math.isfinite(x) for x in (value,dt)) or not 0<dt<=.05:raise ValueError('控制周期中断')
        if abs(value-self.leader)>45.:raise ValueError('主臂超出本轮45°输入范围')
        desired=self.anchor+math.radians(max(-MAX_OFFSET_DEG,min(MAX_OFFSET_DEG,self.direction*self.gain*(value-self.leader))))
        step=math.radians(SPEED_DEG_S)*dt
        self.target+=max(-step,min(step,desired-self.target));return self.target


class Follow:
    def __init__(self,state):
        self.state=state;self.mutex=threading.RLock();self.thread=None;self.stop_event=threading.Event()
        self.phase='idle';self.token=None;self.lease=0.;self.info={};self.motion_count=0;self.ready=False;self.external_stop=''
        try:
            proof=json.loads((state.out/'j1_driver_verified.json').read_text())
            self.ready=proof.get('passed') is True and proof.get('capability_hash')==capability_hash()
        except (OSError,ValueError):pass
    def snapshot(self):
        with self.mutex:return dict(phase=self.phase,active=self.phase in ('preparing','following','stopping'),
            available_axes=[1],commissioned_axes=[1] if self.ready else [],max_delta_deg=MAX_OFFSET_DEG,speed_deg_s=SPEED_DEG_S,
            torque_limit_Nm=TORQUE_LIMIT_NM,max_session_s=45,lease_timeout_s=1.,motion_commands=self.motion_count,**self.info)
    def start(self,data,pilot=False):
        with self.mutex:
            if self.thread and self.thread.is_alive():raise ValueError('已有跟随会话，请先停止')
            if not pilot and not self.ready:raise ValueError('J1驱动复验尚未通过')
            if type(data.get('axis')) is not int or data['axis']!=1 or data.get('site_confirmed') is not True:raise ValueError('仅开放已确认现场条件的J1单轴测试')
            direction=data.get('direction',1);gain=data.get('gain',.1)
            if pilot and (type(pilot) not in (int,float) or pilot not in (.5,1.)):raise ValueError('CLI复验幅度无效')
            if type(direction) is not int or direction not in (-1,1) or type(gain) not in (int,float) or not math.isfinite(gain) or not .05<=gain<=1.:raise ValueError('J1映射无效')
            self.token=secrets.token_urlsafe(24);self.lease=time.monotonic();self.phase='preparing';self.info={};self.external_stop=''
            self.stop_event.clear();self.thread=threading.Thread(target=self.run,args=(direction,gain,pilot),daemon=True);self.thread.start()
            return self.token
    def heartbeat(self,token):
        with self.mutex:
            if not self.token or not secrets.compare_digest(str(token),self.token):raise ValueError('不是本次跟随控制端')
            self.lease=time.monotonic()
    def stop(self,reason='用户停止跟随'):
        self.external_stop=reason;self.stop_event.set()
    def input(self):
        with self.state.lock:row=dict(self.state.engine.leader.get(0,{}))
        if not row or not row.get('reliable') or time.monotonic()-row.get('sample_time',row.get('received',0))>.2:
            raise RuntimeError('J1主臂反馈过期或不可靠')
        return row['filtered_deg']
    def check_owner(self,pilot=False):
        if self.stop_event.is_set():raise InterruptedError(self.external_stop or '停止')
        if not pilot and time.monotonic()-self.lease>1.:raise InterruptedError('网页心跳超时，已停止J1')
    def run(self,direction,gain,pilot=False):
        c=None;m=None;lock=None;original={};changed=[];enabled=False;rows=[];error=None;reason='';disabled=False
        self.state.pause_can.set();session=f'j1_{time.time_ns()}';start=time.monotonic();tx=[];baseline=[];anchor=None
        try:
            lock=open('/tmp/reflex-can-actuation.lock','a')
            for _ in range(20):
                self.check_owner(pilot)
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.05)
            else:raise RuntimeError('CAN被其他控制器占用')
            c=ReadOnlyCAN();c.capture=[];m=DiagnosticMotor(c,1)
            # Reporting changes only; never enable another joint.
            for j in range(1,8):
                frame=struct.pack('=IB3x8s',0x80000000|(24<<24)|(253<<8)|j,8,bytes([1,2,3,4,5,6,1,0]))
                c.sock.send(frame);tx.append(dict(kind='report_on',joint=j,frame=frame.hex()))
            c.poll(.15)
            if set(c.latest_status)!=set(range(1,8)):raise RuntimeError('需要7轴新鲜状态反馈')
            initial_status={j:dict(s) for j,s in c.latest_status.items()}
            if any(s['fault_mask_raw'] or s['temperature_C']>=45 or s['mode_bits_raw']!=0 for s in initial_status.values()):
                raise RuntimeError('首次J1联动要求7轴无故障且初始失能；不会接管正在运行的电机')
            baseline=[c.read(j,0x7019,.025) for j in range(1,8)];anchor=baseline[0]
            if not -2.5<anchor<2.5:raise RuntimeError('J1超出受限联动起始范围')
            if c.read(1,0x7005,.025)!=0:raise RuntimeError('J1需要已配置的MIT模式，不自动切换')
            if not 42<=c.read(1,0x701C,.025)<=54:raise RuntimeError('电源电压异常')
            for p in (0x700B,0x7028):original[p]=c.read(1,p,.025)
            if original[0x700B]<.1:raise RuntimeError('现有力矩限制不足')
            # Check stability before any enable and record local leader reference.
            self.input()
            for _ in range(15):
                self.check_owner(pilot)
                if abs(c.read(1,0x7019,.025)-anchor)>math.radians(.1):raise RuntimeError('J1静止检查失败')
                time.sleep(.01)
            for p,v in ((0x700B,min(TORQUE_LIMIT_NM,original[0x700B])),(0x7028,2000)):
                changed.append(p);m.set_volatile(p,v)
            target=Target(self.input(),anchor,direction,gain)
            m.mit(anchor,0.,0.);self.check_owner(pilot);enabled=True;m.enable()
            m.mit(anchor,200.,3.)
            # The first physical command is an actual-position hold, not leader absolute position.
            limit=c.read(1,0x700B,.02);watchdog=c.read(1,0x7028,.02)
            if abs(limit-min(TORQUE_LIMIT_NM,original[0x700B]))>1e-5 or watchdog!=2000:raise RuntimeError('使能后限值回读不一致')
            guard=VelocityGuard(limit_deg_s=3.,raw_limit_deg_s=3.)
            with self.mutex:
                self.phase='following';self.info=dict(session=session,pilot=pilot,leader_reference_deg=target.leader,
                    follower_reference_deg=math.degrees(anchor),direction=direction,gain=gain,target_deg=math.degrees(anchor),elapsed_s=0.)
            motion_start=time.monotonic();previous=motion_start-.02;next_tick=motion_start
            i=0
            while True:
                time.sleep(max(0.,next_tick-time.monotonic()));now=time.monotonic();dt=now-previous;previous=now;next_tick=now+.02
                elapsed=now-motion_start;self.check_owner(pilot)
                if elapsed>(18. if pilot else 45.):reason='单轮时长结束';break
                leader=self.input()
                if pilot:
                    # CLI commissioning only, no HTTP route can request this waveform.
                    leader=target.leader+float(pilot)/gain*.5*(1-math.cos(2*math.pi*min(elapsed,16.)/16.))
                qtarget=target.update(leader,dt);m.mit(qtarget,200.,3.)
                self.motion_count+=1
                q=c.read(1,0x7019,.015);v=c.read(1,0x701B,.015);current=c.read(1,0x701A,.015)
                sample=dict(t_s=elapsed,leader_deg=leader,target_deg=math.degrees(qtarget),actual_deg=math.degrees(q),velocity_deg_s=math.degrees(v),current_A=current,gap_s=dt)
                rows.append(sample);guard.update(time.monotonic_ns(),q,v)
                if abs(q-anchor)>math.radians(MAX_OFFSET_DEG+.3) or abs(q-qtarget)>math.radians(.65):raise RuntimeError('J1位置或跟踪误差超限')
                st=c.latest_status.get(1,{})
                if not st or time.monotonic_ns()-st['rx_ns']>100_000_000 or st['mode_bits_raw']!=2 or st['fault_mask_raw'] or st['temperature_C']>=45:
                    raise RuntimeError('J1状态异常')
                # Check one non-commanded encoder per cycle, and all status frames.
                other=2+i%6;otherq=c.read(other,0x7019,.015)
                with self.state.lock:self.state.engine.ingest_follower([dict(id=other,position_deg=math.degrees(otherq),mode_bits=c.latest_status.get(other,{}).get('mode_bits_raw'),faults=[])])
                if abs(otherq-baseline[other-1])>math.radians(.5):raise RuntimeError(f'非指令J{other}位移超过0.5°')
                if any(s['fault_mask_raw'] or (j!=1 and s['mode_bits_raw']!=0) for j,s in c.latest_status.items()):raise RuntimeError('其他轴状态变化')
                if i%10==0:
                    if not 42<=c.read(1,0x701C,.015)<=54:raise RuntimeError('供电异常')
                    with self.state.lock:self.state.engine.ingest_follower([dict(id=1,position_deg=math.degrees(q),velocity_deg_s=math.degrees(v),mode_bits=2,faults=[])])
                with self.mutex:self.info.update(target_deg=math.degrees(qtarget),actual_deg=math.degrees(q),elapsed_s=round(elapsed,2))
                i+=1
        except InterruptedError as exc:reason=str(exc)
        except BaseException as exc:error=str(exc)
        finally:
            with self.mutex:self.phase='stopping'
            # J1 only. Never release a load-bearing joint as a generic stop.
            if m is not None and enabled:
                try:
                    m.disable();c.poll(.06);disabled=m.status()['mode_bits_raw']==0
                    if not disabled:raise RuntimeError('J1停止未确认')
                except BaseException as exc:error=(error or '')+'; '+str(exc)
            elif not enabled:disabled=True
            restore_errors=[]
            if disabled and m is not None:
                for p in changed:
                    try:m.set_volatile(p,int(original[p]) if p==0x7028 else original[p])
                    except BaseException as exc:restore_errors.append(str(exc))
            if restore_errors:error=(error or '')+'; 参数恢复失败: '+'; '.join(restore_errors)
            summary=dict(session=session,pilot=pilot,error=error,reason=reason,enabled=enabled,disabled=disabled,
                capability_hash=capability_hash(),
                configured_torque_limit_Nm=TORQUE_LIMIT_NM,speed_limit_deg_s=SPEED_DEG_S,kp=200.,kd=3.,
                original_parameters={hex(p):v for p,v in original.items()},restored=not restore_errors and disabled,
                baseline_rad=baseline,axis=1,samples=len(rows),motion_commands=len(m.tx) if m else 0,
                elapsed_s=round(time.monotonic()-start,2),other_joint_actuation=0)
            if rows:
                summary.update(actual_excursion_deg=max(r['actual_deg'] for r in rows)-min(r['actual_deg'] for r in rows),
                    max_tracking_error_deg=max(abs(r['actual_deg']-r['target_deg']) for r in rows),
                    return_error_deg=rows[-1]['actual_deg']-math.degrees(anchor),
                    peak_current_A=max(abs(r['current_A']) for r in rows))
            summary['passed']=bool(enabled and disabled and not error and rows and (not pilot or (float(pilot)*.6<summary['actual_excursion_deg']<float(pilot)+.2 and abs(summary['return_error_deg'])<.2 and summary['max_tracking_error_deg']<.2)))
            try:
                directory=self.state.out/session;directory.mkdir()
                (directory/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
                (directory/'samples.json').write_text(json.dumps(rows))
                (directory/'tx.json').write_text(json.dumps(dict(reporting=tx,actuator=m.tx if m else [])))
                (directory/'rx.json').write_text(json.dumps(c.capture if c else []))
            except BaseException as exc:error=(error or '')+'; 保存记录失败 '+str(exc);summary['passed']=False
            if c is not None:c.close()
            if lock is not None:lock.close()
            with self.mutex:
                self.info=dict(summary=summary,message=error or reason or '已停止',session=session)
                self.phase='fault' if error else 'idle';self.token=None
                if pilot==MAX_OFFSET_DEG and summary['passed']:
                    self.ready=True
                    (self.state.out/'j1_driver_verified.json').write_text(json.dumps(summary,indent=2))
            self.state.pause_can.clear()
            with self.state.lock:self.state.engine.event('follow_stop','J1跟随结束：'+str(error or reason or '已停止'))
