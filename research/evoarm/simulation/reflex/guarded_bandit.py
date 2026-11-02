"""Shadow updates, development-table acceptance, and exact model rollback.

This is a simulation-side model-selection mechanism. Terminal success and
native-force labels may score completed attempts and development validation;
they are never actor inputs. Reusing this development table is adaptive model
selection, not an independent safety guarantee. All incurred trials remain.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from .residual_bandit import ResidualBandit

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def guard_metrics(policy,table):
    before=policy.digest();chosen=[]
    for case in table:
        action='zero' if case['x'] is None else policy.act(case['x'],case['allowed'],explore=False)
        chosen.append(case['results'][action])
    assert policy.digest()==before,'Validation changed a candidate or comparator'
    def summary(rows):
        return dict(n=len(rows),success=sum(r['success'] for r in rows),overclamp=sum(r['overclamp'] for r in rows),abort=sum(r['abort'] for r in rows),
                    utility_mean=float(np.mean([r['utility'] for r in rows])),
                    peak_p95_N=float(np.percentile([r['peak_N'] for r in rows],95)),peak_max_N=max(r['peak_N'] for r in rows),
                    ratio_p95=float(np.percentile([r['peak_N']/r['budget_N'] for r in rows],95)),ratio_max=max(r['peak_N']/r['budget_N'] for r in rows),
                    native_overforce_total_s=sum(r['native_overforce_s'] for r in rows))
    return dict(overall=summary(chosen),materials={m:summary([r for r in chosen if r['material']==m]) for m in sorted({r['material'] for r in chosen})})

def acceptance(candidate,incumbent,frozen,min_utility_gain=.01):
    checks={}
    for label,base in [('incumbent',incumbent),('frozen',frozen)]:
        for group in ['overall']+sorted(candidate['materials']):
            c=candidate['overall'] if group=='overall' else candidate['materials'][group]
            b=base['overall'] if group=='overall' else base['materials'][group]
            prefix=f'{label}/{group}/'
            checks[prefix+'success_not_lower']=c['success']>=b['success']
            for metric in ['overclamp','abort','peak_p95_N','peak_max_N','ratio_p95','ratio_max','native_overforce_total_s']:
                checks[prefix+metric+'_not_increased']=c[metric]<=b[metric]+1e-10
    checks['utility_gain_over_incumbent']=candidate['overall']['utility_mean']>=incumbent['overall']['utility_mean']+min_utility_gain
    checks['utility_not_below_frozen']=candidate['overall']['utility_mean']>=frozen['overall']['utility_mean']-1e-12
    return checks


class GuardedBandit:
    algorithm='development_guarded_contextual_bandit_v1'
    def __init__(self,offline,guard_table,block_size=8,epsilon=.03,min_utility_gain=.01):
        if not guard_table or block_size<1 or not 0<=epsilon<=1:raise ValueError('Invalid acceptance configuration')
        self.frozen=offline.clone();self.deployed=offline.clone();self.candidate=offline.clone()
        self.guard_table=deepcopy(guard_table);self.guard_digest=digest(guard_table)
        self.config=dict(block_size=int(block_size),epsilon=float(epsilon),min_utility_gain=float(min_utility_gain))
        self.rng=np.random.default_rng(offline.seed+70000)
        self.experience=[];self.events=[];self.feedback_count=0;self.attempted_gradient_steps=0;self.pending=0
        self.accepted_blocks=0;self.rejected_blocks=0;self.safety_rollbacks=0
        self.proposal_snapshot=None
        self.frozen_metrics=guard_metrics(self.frozen,self.guard_table)
        self.action_names=offline.action_names;self.anchor_digest=offline.anchor_digest

    @property
    def updates(self):return self.feedback_count
    @property
    def online_gradient_steps(self):return self.attempted_gradient_steps
    def model_digest(self):return self.deployed.model_digest()
    def predict(self,x):return self.deployed.predict(x)
    def act(self,x,allowed_names=None,explore=False):
        allowed=list(self.action_names if allowed_names is None else allowed_names)
        if explore and self.rng.random()<self.config['epsilon']:
            # Previous development incidents motivate a conservative exploratory
            # subset. Greedy deployment still uses the common eight-action bank.
            safe=[a for a in allowed if a in ('zero','slow2','hold05_slow2','pitch_pos_regrasp','soft_relief')]
            return safe[int(self.rng.integers(len(safe)))]
        return self.deployed.act(x,allowed,explore=False)

    def _restart_shadow(self):
        state=self.deployed.state()
        # Roll back weights/optimizer/training RNG, retain every actual completed
        # experience including rejected and risky attempts in separate replay.
        state['online_replay']=deepcopy(self.experience[-self.deployed.replay_capacity:])
        self.candidate=ResidualBandit(self.action_names).load(state)
        self.pending=0

    def _validate(self):
        c=guard_metrics(self.candidate,self.guard_table)
        i=guard_metrics(self.deployed,self.guard_table)
        checks=acceptance(c,i,self.frozen_metrics,self.config['min_utility_gain'])
        before=self.deployed.model_digest();proposal=self.candidate.model_digest()
        self.proposal_snapshot=dict(candidate=self.candidate.state(),deployed_before=self.deployed.state())
        accepted=all(checks.values())
        if accepted:self.deployed=self.candidate.clone();self.accepted_blocks+=1
        else:self.rejected_blocks+=1
        event=dict(kind='development_block_validation',completed_feedbacks=self.feedback_count,block_feedbacks=self.pending,
                   accepted=accepted,criteria=checks,candidate_metrics=c,incumbent_metrics=i,frozen_metrics=self.frozen_metrics,
                   deployed_model_before=before,candidate_model=proposal,deployed_model_after=self.deployed.model_digest(),
                   failed_criteria=[k for k,v in checks.items() if not v],guard_table_digest=self.guard_digest,
                   replay_records_retained=len(self.experience),validation_is_adaptive_development=True)
        assert event['deployed_model_after']==(proposal if accepted else before)
        self.events.append(event);self._restart_shadow();return event

    def update(self,x,name,reward,completed_outcome=None):
        record=dict(x=np.asarray(x,dtype=np.float32).astype(float).tolist(),action=name,reward=float(reward))
        update=self.candidate.update(x,name,reward)
        self.experience.append(record);self.feedback_count+=1;self.pending+=1
        self.attempted_gradient_steps+=update['gradient_steps']
        result=dict(shadow_update=update,completed_feedbacks=self.feedback_count,attempted_gradient_steps=self.attempted_gradient_steps,
                    deployed_model_changed=False,acceptance_event=None,safety_reset=None)
        risky=False
        if completed_outcome is not None:
            peak=completed_outcome['fullrate']['peak_force_N'];budget=completed_outcome['physics']['force_budget_N_per_finger']
            risky=bool(completed_outcome['overclamp'] or completed_outcome['safety_abort'] or peak>1.35*budget)
        if risky:
            self.proposal_snapshot=dict(candidate=self.candidate.state(),deployed_before=self.deployed.state())
            before=self.deployed.model_digest();self.deployed=self.frozen.clone();self.safety_rollbacks+=1
            event=dict(kind='completed_risk_rollback',completed_feedbacks=self.feedback_count,
                       deployed_model_before=before,deployed_model_after=self.deployed.model_digest(),frozen_model=self.frozen.model_digest(),
                       fullrate_peak_N=float(peak),budget_N=float(budget),overclamp=bool(completed_outcome['overclamp']),abort=bool(completed_outcome['safety_abort']),
                       incurred_trial_retained=True,experience_records_retained=len(self.experience))
            self.events.append(event);self._restart_shadow();result['safety_reset']=event
        elif self.pending>=self.config['block_size']:
            result['acceptance_event']=self._validate()
        result['deployed_model_changed']=bool(result['acceptance_event'] and result['acceptance_event']['accepted'] or result['safety_reset'] and result['safety_reset']['deployed_model_before']!=result['safety_reset']['deployed_model_after'])
        return result

    def finalize(self):return self._validate() if self.pending else None
    def state(self):
        return dict(algorithm=self.algorithm,config=deepcopy(self.config),frozen=self.frozen.state(),deployed=self.deployed.state(),candidate=self.candidate.state(),
                    guard_table=deepcopy(self.guard_table),guard_table_digest=self.guard_digest,frozen_metrics=deepcopy(self.frozen_metrics),
                    numpy_rng=deepcopy(self.rng.bit_generator.state),experience=deepcopy(self.experience),events=deepcopy(self.events),
                    counters={k:getattr(self,k) for k in ['feedback_count','attempted_gradient_steps','pending','accepted_blocks','rejected_blocks','safety_rollbacks']})
    @classmethod
    def from_state(cls,state):
        if state.get('algorithm')!=cls.algorithm or digest(state['guard_table'])!=state['guard_table_digest']:raise ValueError('Guard checkpoint integrity')
        frozen=ResidualBandit(state['frozen']['config']['action_names']).load(state['frozen'])
        obj=cls(frozen,state['guard_table'],**state['config'])
        if obj.frozen_metrics!=state['frozen_metrics']:raise ValueError('Guard baseline metrics changed')
        obj.deployed=ResidualBandit(obj.action_names).load(state['deployed'])
        obj.candidate=ResidualBandit(obj.action_names).load(state['candidate'])
        if obj.deployed.anchor_digest!=obj.anchor_digest or obj.candidate.anchor_digest!=obj.anchor_digest:raise ValueError('Source anchor changed')
        obj.rng.bit_generator.state=state['numpy_rng'];obj.experience=deepcopy(state['experience']);obj.events=deepcopy(state['events'])
        for k,v in state['counters'].items():setattr(obj,k,int(v))
        if len(obj.experience)!=obj.feedback_count or obj.attempted_gradient_steps!=8*obj.feedback_count:raise ValueError('Completed feedback counters differ')
        return obj
    def clone(self):return type(self).from_state(self.state())
    def digest(self):return digest(self.state())
