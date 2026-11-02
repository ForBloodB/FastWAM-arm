"""Sensor-conditioned, bounded stage-force residual Monte Carlo control.

The policy has no object truth input. Simulator truth supplies episodic reward
and audit scores only. No new robot joint trajectory or hardware actuator API.
"""
from collections import defaultdict
import hashlib,json
import numpy as np
from .feedback_grasp import FeedbackGraspEnv
from .official_grasp import DRIVE_FORCE,DENSITY,FRICTION

RATIOS=np.array([.65,.8,1.,1.2,1.4])
PHASES=('close','probe_lift','lift','transport_a','transport','lower')

class ResidualPolicy:
    def __init__(self,seed=0,epsilon=.2):
        self.rng=np.random.default_rng(seed);self.epsilon=epsilon;self.q={};self.count={};self.history=[];self.updates=0
    def act(self,key,learn=False):
        q=self.q.get(key,np.zeros(5));n=self.count.get(key,np.zeros(5))
        if learn and self.rng.random()<self.epsilon:a=int(self.rng.integers(5))
        else:
            # Neutral residual wins exact ties; no test reward or case ID used.
            best=np.flatnonzero(q==max(q));a=2 if 2 in best else int(best[0])
        self.history.append((key,a));return a
    def update(self,reward):
        for key,a in self.history:
            if key not in self.q:self.q[key]=np.zeros(5);self.count[key]=np.zeros(5,dtype=int)
            self.count[key][a]+=1
            # Constant step size permits adaptation under deployment shift.
            self.q[key][a]+=.12*(float(reward)-self.q[key][a])
        self.updates+=1;self.history=[]
    def state(self):
        return {'q':{k:v.tolist() for k,v in sorted(self.q.items())},'count':{k:v.tolist() for k,v in sorted(self.count.items())},'updates':self.updates}
    def digest(self):return hashlib.sha256(json.dumps(self.state(),sort_keys=True).encode()).hexdigest()
    def load(self,state):
        self.q={k:np.array(v,float) for k,v in state['q'].items()};self.count={k:np.array(v,int) for k,v in state['count'].items()};self.updates=state['updates'];return self

class ResidualFeedbackEnv(FeedbackGraspEnv):
    def __init__(self,policy=None,learning=False,neutral=False,**kwargs):
        super().__init__(group='vision_contact',**kwargs)
        self.policy=policy or ResidualPolicy();self.learning=learning;self.neutral=neutral
        self.policy_decisions=[];self.latest_seen=None
    def reset(self,**kwargs):
        self.latest_seen=None;self.policy_decisions=[];self.policy.history=[]
        obs,info=super().reset(**kwargs)
        if not self.failure:
            # Same visual analytic nominal baseline in ALL residual conditions.
            mass=np.prod(self.estimated_size)*DENSITY[self.predicted]
            self.drive=float(np.clip(max(DRIVE_FORCE[self.predicted],2*mass*9.81/FRICTION[self.predicted]),1.5,min(28,1.8*self.force_budget)))
            self.start_drive=self.drive
        return obs,info
    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs);self.latest_seen=result.copy();return result
    def policy_context(self,phase):
        # All bins use the sensor whitelist: visual class/size/rise, encoder,
        # commanded cap and delayed fingertip-force measurement.
        seen=self.latest_seen or {};valid=bool(seen.get('valid'))
        rise=(seen['xyz'][2]-self.initial_estimate[2]) if valid else 0.
        force_ratio=float(max(self.sensor_force)/max(self.force_budget,1e-6))
        values=[PHASES.index(phase),int(self.predicted),int(np.prod(self.estimated_size)>.000060),
                int(not valid or rise<.008),int(force_ratio>.6),int(self.drive>self.start_drive*1.1)]
        return ','.join(map(str,values)),{'vision_label':self.predicted,'estimated_volume_m3':float(np.prod(self.estimated_size)),
          'visual_rise_m':float(rise),'vision_valid':valid,'force_ratio':force_ratio,'commanded_drive_N':self.drive}
    def move(self,xyz,yaw,duration,phase,gap,force):
        if phase in PHASES:
            key,obs=self.policy_context(phase);a=2 if self.neutral else self.policy.act(key,self.learning)
            old=self.drive;self.drive=float(np.clip(old*RATIOS[a],1.5,min(28,1.8*self.force_budget)))
            self.policy_decisions.append({'phase':phase,'key':key,'action':a,'ratio':float(RATIOS[a]),'before_N':old,'after_N':self.drive,'observation':obs})
        super().move(xyz,yaw,duration,phase,gap,force)
    def run(self):
        row=super().run();m=row['contact_metrics']
        reward=float(2*row['success']+.25*m['lifted']+.1*m['bilateral']-.8*row['overclamp']-.3*m['slip']-2*row['safety_abort']-.02*row['retries'])
        row.update(controller='stage_force_residual_monte_carlo',reward=reward,policy_decisions=self.policy_decisions,
                   policy_observation_privileged=False,reward_source='simulator_truth_for_training_only',learned_joint_trajectory=False)
        if self.learning:self.policy.update(reward)
        else:self.policy.history=[]
        return row
