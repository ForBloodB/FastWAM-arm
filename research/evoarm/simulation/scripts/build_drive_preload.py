import subprocess
from pathlib import Path
import mujoco
root=Path(mujoco.__file__).parent
out=Path('build/drive_preload');out.mkdir(parents=True,exist_ok=True)
lib=next(root.glob('libmujoco.so*'))
subprocess.run(['gcc','-O3','-shared','-fPIC','reflex/drive_preload_control.c','-I'+str(root/'include'),str(lib),'-Wl,-rpath,'+str(root),'-lm','-o',str(out/'libdrive_preload.so')],check=True)
