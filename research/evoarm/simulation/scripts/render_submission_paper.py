"""First declared development case replay; simulated video, not physical footage."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,subprocess,faulthandler
from copy import deepcopy
from pathlib import Path
import mujoco,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from reflex.submission_paper_task import PaperBoxEnv
from scripts.run_submission_confirmation import isolate
import sys
sys.addaudithook(isolate)
from reflex.official_grasp import DESTINATION
from scripts import run_transition_gate as io

class Video(PaperBoxEnv):
    def __init__(self,**kwargs):
        self.capture=False;self.writer=None;self.ticks=0;self.frame_count=0;self.captured=set()
        super().__init__(**kwargs)
    def sample(self,dt):
        super().sample(dt)
        if not self.capture:return
        self.ticks+=1
        if self.ticks%10:return
        camera=mujoco.MjvCamera();camera.lookat[:]=[.20,0,.25];camera.distance=.95;camera.azimuth=128;camera.elevation=-24
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,width=640,height=480)
        r=self.renderer;r.update_scene(self.data,camera=camera)
        for x,y,sx,sy in ((-.025,0,.0007,.025),(.025,0,.0007,.025),(0,-.025,.025,.0007),(0,.025,.025,.0007)):
            scene=r.scene;g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([sx,sy,.0003]),np.array([DESTINATION[0]+x,DESTINATION[1]+y,.1005]),np.eye(3).ravel(),np.array([.1,1.,.4,1.]))
            scene.ngeom+=1
        im=Image.fromarray(r.render().copy());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,640,72),fill=(15,29,43));draw.rectangle((0,450,640,480),fill=(15,29,43))
        draw.text((12,7),f'SIMULATION | Empty paper box | {self.paper_mode}',font=self.font,fill='white')
        draw.text((12,35),f'{self.phase} | t={self.data.time-self.start_time:.1f}s | peak={max(self.fullrate[:2]):.3f} / 3 N',font=self.font,fill='white')
        draw.text((12,455),f'Shell mass {self.mass*1000:.2f} g | Rigid proxy; no crushing model',font=self.font,fill='white')
        self.writer.stdin.write(np.asarray(im).tobytes());self.frame_count+=1;self.last_frame=im
        if self.phase not in self.captured:
            self.captured.add(self.phase);im.save(self.out/(self.paper_mode+'_'+self.phase+'.png'))

def main():
    faulthandler.dump_traceback_later(45,repeat=True)
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/submission_paper_20261102_v2');a=p.parse_args();root=Path(a.root);out=root/'demo';out.mkdir(exist_ok=False)
    scene=io.read(root/'plan.json')['test'][0];rows=[]
    for mode in ('baseline','gentle','gentle_recolored'):
        env=Video(mode=mode,dt=.0000625);log=(out/(mode+'.log')).open('wb');video=out/(mode+'.mp4')
        try:
            env.reset(seed=scene['seed'],options=deepcopy(scene['options']));env.out=out
            env.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
            env.writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','warning','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
            env.capture=True;r=env.run();env.capture=False;env.writer.stdin.close();assert env.writer.wait()==0
            original=io.read(root/'test_recovered/trials'/(scene['id']+'__'+mode+'__62500.json'))
            for k in ('success','failure_type','fullrate','common_start_digest','duration_s'):assert r[k]==original[k],k
            env.last_frame.save(out/(mode+'_poster.png'))
            rows.append(dict(mode=mode,scene_id=scene['id'],video=video.name,sha256=io.sha(video),frames=env.frame_count,success=r['success'],peak_N=r['fullrate']['peak_force_N'],exact_physics_replay=True,motor_commands=0))
        finally:
            if env.writer is not None and env.writer.poll() is None:
                env.writer.terminate()
                try:env.writer.wait(timeout=5)
                except subprocess.TimeoutExpired:env.writer.kill();env.writer.wait()
            env.close();log.close()
    faulthandler.cancel_dump_traceback_later()
    io.write(out/'complete.json',dict(rows=rows,evidence='Simulation replay only; not a physical robot video'));print(rows)

if __name__=='__main__':main()
