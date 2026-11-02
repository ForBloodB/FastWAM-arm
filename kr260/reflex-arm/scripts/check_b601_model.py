"""Read-only B601 URDF validation. No ROS or motor API is imported."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reflex.kinematics import URDFModel,quintic_trajectory

p=argparse.ArgumentParser()
p.add_argument('--urdf',default='assets/B601_RS.urdf')
p.add_argument('--out',default='runs/kinematics/summary.json')
a=p.parse_args()
model=URDFModel(a.urdf)
rng=np.random.default_rng(101)
rows=[]
for i in range(20):
    q=np.clip(rng.uniform(-.35,.35,len(model.names)),model.limits[:,0],model.limits[:,1])
    target=model.fk(q)[:3,3]
    result=model.ik_position(target,q+rng.normal(0,.05,len(q)))
    rows.append(result)
t,trajectory=quintic_trajectory(np.zeros(len(model.names)),np.full(len(model.names),.1),3.)
summary={'evidence':'URDF_kinematic_calculation_no_physics_no_motors',
    'urdf':a.urdf,'urdf_sha256':hashlib.sha256(Path(a.urdf).read_bytes()).hexdigest(),
    'joint_names':model.names,'joint_limits_rad':model.limits.tolist(),
    'ik_cases':rows,'ik_converged':sum(r['converged'] for r in rows),
    'trajectory_samples':len(t),'trajectory_max_joint_speed_rad_s':float(np.max(abs(np.diff(trajectory,axis=0)/np.diff(t)[:,None]))),
    'limitations':'Position-only IK; no mesh collisions, orientation constraints, dynamics or grasp verification'}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
