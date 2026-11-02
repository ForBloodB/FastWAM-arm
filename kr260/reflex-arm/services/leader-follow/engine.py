"""Read-only/shadow teleoperation state machine. No actuator API exists here."""
from copy import deepcopy
import math,time

NAMES=['底座旋转','肩部','肘部','腕部俯仰','腕部偏航','腕部旋转','夹爪']


class Problem(ValueError):pass


class Engine:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.mode='monitor';self.reference=None;self.revision=0
        self.mapping=[dict(leader_id=i,follower_id=i+1,name=NAMES[i],direction=1,gain=1.,
            enabled=True,verified=False,limit_deg=1.,speed_deg_s=.5) for i in range(7)]
        self.leader={};self.follower={};self.target={};self.last_tick=None;self.events=[]
        self.leader_error='等待主臂数据';self.follower_error='等待从臂反馈'
        self.count=0;self.read_batches=0;self.started=self.clock()
        self.event('ready','服务就绪，当前只读监看')

    def event(self,kind,message):
        self.events.append(dict(t_s=round(self.clock()-self.started,3),kind=kind,message=message))
        self.events=self.events[-80:]

    def ingest_leader(self,rows,error=None):
        now=self.clock()
        for row in rows:
            if row['id'] not in range(7):continue
            if not all(math.isfinite(float(row[k])) for k in ['raw_deg','filtered_deg']):continue
            self.leader[row['id']]=dict(row,received=row.get('sample_time',now))
        self.leader_error=error;self.count+=len(rows);self.read_batches+=1
        self.tick()

    def ingest_follower(self,rows,error=None):
        now=self.clock()
        for row in rows:
            if row['id'] in range(1,8) and math.isfinite(row['position_deg']):
                self.follower[row['id']]=dict(row,received=now)
        self.follower_error=error

    def fresh_leader(self):
        now=self.clock()
        wanted=[r['leader_id'] for r in self.mapping if r['enabled']]
        return bool(wanted) and all(i in self.leader and now-self.leader[i]['received']<=.5 and self.leader[i]['reliable'] for i in wanted)

    def capture(self):
        if self.mode=='preview':raise Problem('请先停止预览，再采集参考姿态')
        if not self.fresh_leader():raise Problem('所选主臂关节需要新鲜且可靠的角度读数')
        now=self.clock()
        self.reference=dict(leader={i:r['filtered_deg'] for i,r in self.leader.items()},
            follower={i:r['position_deg'] for i,r in self.follower.items() if now-r['received']<2.},
            captured_s=round(now-self.started,3),mapping_revision=self.revision)
        self.target={i+1:0. for i in range(7)}
        self.event('reference','已记录软件参考姿态；未设置电机零点')

    def update_mapping(self,rows):
        if not isinstance(rows,list) or len(rows)!=7:raise Problem('需要完整的7行关节映射')
        cleaned=[]
        for i,r in enumerate(rows):
            if not isinstance(r,dict):raise Problem('每行映射需要是JSON对象')
            if r.get('follower_id')!=i+1 or type(r.get('leader_id')) is not int or r['leader_id'] not in range(7):raise Problem('关节ID无效')
            if type(r.get('direction')) is not int or r['direction'] not in [-1,1]:raise Problem('方向必须为+1或-1')
            if any(type(r.get(k)) is not bool for k in ['enabled','verified']):raise Problem('映射开关无效')
            try:gain,limit,speed=[float(r[k]) for k in ['gain','limit_deg','speed_deg_s']]
            except (KeyError,TypeError,ValueError):raise Problem('比例、范围和速度需要数值')
            if not all(math.isfinite(v) for v in [gain,limit,speed]):raise Problem('不能使用非有限数值')
            if not .05<=gain<=6. or not .1<=limit<=10. or not .05<=speed<=2.:raise Problem('预览范围：比例0.05–6，行程0.1–10°，速度0.05–2°/秒')
            cleaned.append(dict(leader_id=r['leader_id'],follower_id=i+1,name=NAMES[i],direction=r['direction'],gain=gain,
                enabled=r['enabled'],verified=r['verified'],limit_deg=limit,speed_deg_s=speed))
        if len({r['leader_id'] for r in cleaned})!=7:raise Problem('每个主臂关节只能映射一次')
        self.stop('映射变更，已停止预览并清除参考姿态')
        self.mapping=cleaned;self.revision+=1;self.reference=None;self.target={}

    def start(self):
        if not self.reference:raise Problem('请先记录当前参考姿态')
        if not self.fresh_leader():raise Problem('主臂反馈不完整、过期或不可靠')
        if self.reference['mapping_revision']!=self.revision:raise Problem('映射已变更，请重采参考姿态')
        if not any(r['enabled'] for r in self.mapping):raise Problem('至少选中一个预览关节')
        self.mode='preview';self.last_tick=self.clock()
        self.event('preview','已开始目标预览；CAN运动发送始终为0')

    def stop(self,message='已停止目标预览'):
        self.mode='monitor';self.last_tick=None;self.event('stop',message)

    def tick(self):
        if self.mode!='preview':return
        now=self.clock()
        if not self.fresh_leader():self.stop('主臂反馈超时或不可靠，预览已停止');return
        dt=now-self.last_tick
        if dt<=0:return
        if dt>.5:self.stop('数据周期中断，预览已停止');return
        for r in self.mapping:
            desired=0.
            if r['enabled']:
                delta=self.leader[r['leader_id']]['filtered_deg']-self.reference['leader'][r['leader_id']]
                if abs(delta)>180.:self.stop('主臂出现角度跳变，需重新采集参考姿态');return
                desired=max(-r['limit_deg'],min(r['limit_deg'],r['direction']*r['gain']*delta))
            old=self.target.get(r['follower_id'],0.);step=r['speed_deg_s']*dt
            self.target[r['follower_id']]=old+max(-step,min(step,desired-old))
        self.last_tick=now

    def snapshot(self):
        self.tick();now=self.clock();reference=deepcopy(self.reference)
        rows=[]
        for m in self.mapping:
            l=self.leader.get(m['leader_id']);f=self.follower.get(m['follower_id'])
            relative=self.target.get(m['follower_id']) if reference else None
            anchor=None if not reference else reference['follower'].get(m['follower_id'])
            rows.append(dict(**m,leader=None if l is None else dict(l,age_ms=round((now-l['received'])*1000,1)),
                follower=None if f is None else dict(f,age_ms=round((now-f['received'])*1000,1)),
                target_delta_deg=relative,target_absolute_deg=None if anchor is None or relative is None else anchor+relative))
        return dict(mode=self.mode,reference=reference,revision=self.revision,joints=rows,
            leader_fresh=self.fresh_leader(),leader_online=sum(now-r['received']<=.5 for r in self.leader.values()),
            leader_reliable=sum(now-r['received']<=.5 and r['reliable'] for r in self.leader.values()),
            follower_online=sum(now-r['received']<=2. for r in self.follower.values()),
            leader_error=self.leader_error,follower_error=self.follower_error,
            samples_read=self.count,leader_batches=self.read_batches,uptime_s=round(now-self.started,2),
            events=deepcopy(self.events[-20:]),motion_commands=0,hardware_follow_enabled=False,
            hardware_follow_reason='实体跟随尚未完成逐轴方向、零位、限位及承重轴停机校核；本服务没有电机使能或运动写入接口。',
            port='/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0',baudrate=1000000,can_channel='can0',
            scope='真实角度只读监看 + 限幅/限速目标预览；不发送实体跟随指令')
