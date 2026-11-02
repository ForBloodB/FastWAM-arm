import hashlib,json,subprocess
from pathlib import Path
import mujoco
root=Path(__file__).resolve().parents[1]
pkg=Path(mujoco.__file__).parent
lib=next(pkg.glob('libmujoco.so.*'))
out=root/'build/official_control';out.mkdir(parents=True,exist_ok=True)
src=root/'reflex/official_control.c';target=out/'libreflex_control.so'
cmd=['cc','-O2','-shared','-fPIC','-I'+str(pkg/'include'),str(src),str(lib),'-Wl,-rpath,'+str(pkg),'-lm','-o',str(target)]
subprocess.run(cmd,check=True)
meta={'command':cmd,'mujoco':mujoco.__version__,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'compiler':subprocess.check_output(['cc','--version'],text=True).splitlines()[0]}
(out/'BUILD.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
