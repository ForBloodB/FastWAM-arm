import subprocess
from pathlib import Path
import mujoco
root=Path(mujoco.__file__).parent
out=Path('build/transition_control');out.mkdir(parents=True,exist_ok=True)
lib=next(root.glob('libmujoco.so*'))
subprocess.run(['gcc','-O3','-shared','-fPIC','reflex/transition_control.c','-I'+str(root/'include'),str(lib),'-Wl,-rpath,'+str(root),'-lm','-o',str(out/'libtransition.so')],check=True)
