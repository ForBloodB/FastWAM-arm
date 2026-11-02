"""Preflight the complete tilted pickup/transport BEFORE opening the gripper.

Only robot kinematics, declared destination/table and current RGBD/FK are used.
An unreachable 215 mm travel pose may use a verified 195 mm travel corridor;
the original object-lift success threshold and all force/risk limits remain.
"""
from copy import deepcopy
import numpy as np
from .contact_roll_actions import ContactRollActionsEnv,ACTIONS as OLD
from .contact_repair import local_offset_to_world
from .official_grasp import orientation,DESTINATION

ACTIONS={k:deepcopy(OLD[k]) for k in ['zero','tilt_neg55_up6','tilt_neg60_up10']}
ACTIONS['tilt_neg60_up10']['post_pose_drive_cap_N']=26.6


class PreflightContactActionsEnv(ContactRollActionsEnv):
    action_specs=ACTIONS
    def __init__(self,**kwargs):
        self.preflight_events=[];self.travel_height_m=.215;self._preflight_probe=None
        super().__init__(**kwargs)

    def reset(self,**kwargs):
        self.preflight_events=[];self.travel_height_m=.215;self._preflight_probe=None
        return super().reset(**kwargs)

    def _preflight(self,action,raw):
        s=self.action_specs[action];low,yaw=self._saved_descent
        seen=self._preflight_probe
        delta=np.asarray(seen['xyz'])-self.estimate
        correction=np.clip(delta,[-.006,-.006,-.003],[.006,.006,.003])+local_offset_to_world(s['local_offset_m'],yaw)
        offset=np.clip(self.offset+correction,[-.012,-.012,-.012],[.012,.012,.012])
        pickup=np.asarray(low,float)+offset
        now=self.data.site_xpos[self.tcp].copy();clear=now.copy()
        clear[2]=max(clear[2]+.020,(np.asarray(low)+self.offset)[2]+.035,
                     float(seen['xyz'][2])+self.estimated_size[2]/2+.015)
        dyaw=float(np.clip(seen['yaw']-yaw+s['yaw_delta_rad'],-.25,.25));target_yaw=yaw+dyaw
        old=(self.closing_tilt_rad,self.command_pitch,self._joint_hold_q)
        q0=self.data.qpos[:6].copy();attempts=[]
        try:
            self._joint_hold_q=None
            self.closing_tilt_rad=0.;self.command_pitch=float(self.residual_pitch)
            seed=self.ik(clear,yaw+self.residual_yaw,q0)
            self.closing_tilt_rad=s['closing_tilt_rad'];self.command_pitch=self.residual_pitch+s['pitch_delta_rad']
            initial=seed.copy()
            for height in [.215,.195]:
                reason=None;lowest_margin=float('inf');q=initial.copy();targets=[]
                high=np.asarray(low,float).copy()+offset;high[2]=height+offset[2]+self._initial_offset[2]
                destination=np.r_[DESTINATION,height]+.012*orientation(yaw)[:,0];destination[2]=height
                probe=pickup.copy();probe[2]+=.015
                middle=(high+destination)/2;middle[2]=height
                landing=destination.copy();landing[2]=.135
                waypoints=[('open_align',clear),('pickup',pickup),('probe',probe),
                           ('pickup_travel',high),('transport_mid',middle),('destination_travel',destination),('landing',landing)]
                try:
                    for phase,point in waypoints:
                        q=self.ik(point,target_yaw,q)
                        margin=float(np.min(np.minimum(q-self.ranges[:,0],self.ranges[:,1]-q)))
                        lowest_margin=min(lowest_margin,margin)
                        targets.append(dict(phase=phase,target_m=point.tolist(),joint_margin_rad=margin))
                        if margin<.015:raise RuntimeError('preflight_joint_limit_margin')
                    # The pickup's rendered yaw can change slightly after a
                    # stationary release. Check an explicit small yaw tube at
                    # the high pickup pose, where the previous failure occurred.
                    for dy in [-.05,.05]:self.ik(high,target_yaw+dy,q)
                except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:reason=str(exc)
                attempts.append(dict(travel_height_m=height,permitted=reason is None,reason=reason,
                                     minimum_joint_margin_rad=lowest_margin if np.isfinite(lowest_margin) else None,waypoints=targets))
                if reason is None:
                    return dict(action=action,permitted=True,travel_height_m=height,attempts=attempts,
                        scope='Static whole-corridor robot IK; no object/contact truth')
            return dict(action=action,permitted=False,attempts=attempts,reason='no_verified_travel_corridor')
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            return dict(action=action,permitted=False,attempts=attempts,reason=str(exc))
        finally:self.closing_tilt_rad,self.command_pitch,self._joint_hold_q=old

    def probe_features(self,closed,probe,close_tcp,close_opening):
        original=self.rescue_selector;plans={};filtered=[];request_log=[]
        self._preflight_probe=probe
        def select(x,raw):
            nonlocal filtered
            filtered=['zero']
            for a in raw['available_actions']:
                if a=='zero':continue
                plan=self._preflight(a,raw);plans[a]=plan;self.preflight_events.append(deepcopy(plan))
                if plan['permitted']:filtered.append(a)
            actor=deepcopy(raw);actor['available_actions']=filtered;actor['kinematic_preflight']=deepcopy(plans)
            request=self.rescue_action if original is None else original(x,actor)
            request_log.append(request)
            chosen=request if request in filtered else 'zero'
            if chosen!='zero':self.travel_height_m=plans[chosen]['travel_height_m']
            return chosen
        self.rescue_selector=select
        try:
            result=super().probe_features(closed,probe,close_tcp,close_opening)
            if self.residual_decision is not None and request_log:
                d=self.residual_decision;d['available_actions']=filtered;d['allowed_actions']=filtered
                d['requested_action']=request_log[0];d['masked']=request_log[0]!=d['executed_action']
                d['kinematic_preflight']=deepcopy(plans);d['travel_height_m']=self.travel_height_m
            return result
        finally:self.rescue_selector=original;self._preflight_probe=None

    def move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        target=np.asarray(xyz,float).copy()
        if self.closing_tilt_rad and phase in ['lift','transport_a','transport','retreat','settle']:
            target[2]=self.travel_height_m
        return super().move(target,yaw,duration,phase,gap,force,apply_residual,track)

    def run(self):
        r=super().run();r['kinematic_preflight_events']=deepcopy(self.preflight_events)
        r['verified_travel_height_m']=self.travel_height_m
        r['preflight_scope']='Robot-only whole-corridor IK before opening; all masked original attempts remain in denominator'
        return r
