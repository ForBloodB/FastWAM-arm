"""Serialize the unchanged official task; prove XML/binary trajectory parity."""
import hashlib,json,time
from pathlib import Path
import numpy as np,mujoco
from reflex.official_grasp import compile_task_model,CACHE,OfficialGraspEnv
from reflex.official_b601 import verify_vendor
import reflex.official_grasp as task

verify_vendor()
xml=compile_task_model()
if not CACHE.exists():mujoco.mj_saveModel(xml,str(CACHE),None)
binary=mujoco.MjModel.from_binary_path(str(CACHE))
arrays={name:bool(np.array_equal(getattr(xml,name),getattr(binary,name),equal_nan=True)) for name in dir(xml)
        if isinstance(getattr(xml,name),np.ndarray)}
assert all(arrays.values())
original=task.task_model
task.task_model=lambda:xml
a=OfficialGraspEnv();task.task_model=original;b=OfficialGraspEnv()
results=[]
for i in range(4):
    options={'material':task.MATERIALS[i],'condition':'nominal'}
    x=a.reset(seed=5500000000+i,options=options);y=b.reset(seed=5500000000+i,options=options)
    ar=a.step(np.zeros(5));br=b.step(np.zeros(5))
    delta=float(np.max(abs(a.data.qpos-b.data.qpos)))
    assert delta==0 and ar[1]==br[1] and ar[4]['is_success']==br[4]['is_success']
    results.append({'scene':5500000000+i,'max_qpos_difference':delta,'reward_identical':ar[1]==br[1]})
metadata={'mujoco':mujoco.__version__,'sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),
          'bytes':CACHE.stat().st_size,'array_checks':len(arrays),'all_arrays_equal':all(arrays.values()),
          'full_task_checks':results,'geometry_simplified':False,'built_unix':time.time(),'hardware_commands':0}
CACHE.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
