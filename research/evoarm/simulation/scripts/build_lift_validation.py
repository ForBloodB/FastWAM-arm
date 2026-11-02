import subprocess
from pathlib import Path
import mujoco
package=Path(mujoco.__file__).parent
out=Path('build/lift_validation');out.mkdir(parents=True,exist_ok=True)
library=next(package.glob('libmujoco.so*'))
subprocess.run(['gcc','-O3','-shared','-fPIC','reflex/lift_validation_control.c',
    '-I'+str(package/'include'),str(library),'-Wl,-rpath,'+str(package),'-lm',
    '-o',str(out/'liblift_validation.so')],check=True)
