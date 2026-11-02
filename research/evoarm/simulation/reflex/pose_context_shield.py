"""Fixed SOURCE-neighborhood action rejection; no current outcome inputs.

This is empirical support, NOT a formal safety certificate. The actor reuses
four constant material-one-hot slots for rendered pose and measured J4; the
material proxy is still used by the frozen controller/motion masks.
"""
from copy import deepcopy
import numpy as np
from .contact_roll_actions import ContactRollActionsEnv,ACTIONS as OLD_ACTIONS
from .contact_preflight_actions import PreflightContactActionsEnv

ACTIONS={k:deepcopy(OLD_ACTIONS[k]) for k in ['zero','tilt_neg55_up6','tilt_neg60_up6']}
SCHEMA='pose_context_15_v1: bias; RGBD x,y,yaw; encoder J4; original volume/probe/opening/force/slip'


def encode(features,probe,q):
    x=np.asarray(features,float).copy()
    if x.shape!=(15,):raise ValueError('Invalid original sensor features')
    x[1]=np.clip((probe['xyz'][0]-.315)/.035,-2.,2.)
    x[2]=np.clip(probe['xyz'][1]/.05,-2.,2.)
    x[3]=np.clip(probe['yaw']/.6,-2.,2.)
    x[4]=np.clip(q[3]/2.,-2.,2.)
    if not np.isfinite(x).all():raise ValueError('Nonfinite pose state')
    return x


def supported_actions(x,allowed,shield):
    out=['zero'];proof={};indices=shield['distance_indices'];v=np.asarray(x)[indices]
    for action in allowed:
        if action=='zero':continue
        items=shield['source_bank'][action]
        distance=np.array([np.linalg.norm(v-np.asarray(r['x'])[indices]) for r in items])
        order=np.argsort(distance)[:shield['k']]
        passed=(len(order)==shield['k'] and distance[order[-1]]<=shield['radius']
                and all(items[i]['safe'] for i in order))
        proof[action]=dict(permitted=bool(passed),neighbors=[dict(scene_id=items[i]['scene_id'],distance=float(distance[i]),
            source_safe_label=items[i]['safe']) for i in order])
        if passed:out.append(action)
    return out,proof


class PoseContextShieldEnv(ContactRollActionsEnv):
    action_specs=ACTIONS
    def __init__(self,source_shield,**kwargs):
        self.source_shield=deepcopy(source_shield);self.shield_events=[];self._preflight_probe=None
        super().__init__(**kwargs)

    def reset(self,**kwargs):
        self.shield_events=[];self._preflight_probe=None
        return super().reset(**kwargs)

    def probe_features(self,closed,probe,close_tcp,close_opening):
        selector=self.rescue_selector;calls=[];self._preflight_probe=probe
        def choose(x,raw):
            encoded=encode(x,probe,self.data.qpos[:6])
            allowed,proof=supported_actions(encoded,raw['available_actions'],self.source_shield)
            # Reject, rather than alter, a primitive's original 215 mm route.
            # Therefore every admitted primitive remains exactly the previously
            # verified native trajectory; no alternative corridor is introduced.
            plans={};valid=['zero']
            for action in allowed:
                if action=='zero':continue
                plan=PreflightContactActionsEnv._preflight(self,action,raw);plans[action]=plan
                first=plan['attempts'][0] if plan.get('attempts') else None
                if first and first['travel_height_m']==.215 and first['permitted']:valid.append(action)
            actor=deepcopy(raw);actor.update(available_actions=valid,feature_schema=SCHEMA,
                original_sensor_features=x.tolist(),source_neighbor_proof=proof,kinematic_preflight=plans)
            request=self.rescue_action if selector is None else selector(encoded.copy(),actor)
            chosen=request if request in valid else 'zero'
            calls.append(dict(encoded=encoded.tolist(),original=x.tolist(),allowed=valid,requested=request,chosen=chosen,
                neighbor_proof=proof,preflight=plans))
            return chosen
        self.rescue_selector=choose
        try:
            result=super().probe_features(closed,probe,close_tcp,close_opening)
            if calls:
                event=calls[0];self.shield_events.append(deepcopy(event));d=self.residual_decision
                d.update(features=event['encoded'],original_sensor_features=event['original'],
                    available_actions=event['allowed'],allowed_actions=event['allowed'],
                    requested_action=event['requested'],masked=event['requested']!=event['chosen'],feature_schema=SCHEMA)
            return result  # Frozen parent's own 15D sensor/controller branch unchanged.
        finally:self.rescue_selector=selector;self._preflight_probe=None

    def run(self):
        row=super().run();row.update(source_shield_events=deepcopy(self.shield_events),policy_feature_schema=SCHEMA,
            source_shield_scope='Empirical source-neighborhood rejection and robot-only route preflight; not a formal guarantee')
        return row
