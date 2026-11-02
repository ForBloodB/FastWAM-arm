"""Read Orbbec factory geometry and save a synchronized RGB-D engineering sample."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from pyorbbecsdk import Config,Context,OBSensorType,OBFormat,OBStreamType,Pipeline,AlignFilter

p=argparse.ArgumentParser();p.add_argument('--out',required=True)
p.add_argument('--color-width',type=int,choices=(848,1280),default=1280);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
ctx=Context();devices=ctx.query_devices()
result={'evidence':'real_Orbbec_SDK_factory_parameters','device_count':devices.get_count(),'success':False,
        'camera_to_robot_base_calibrated':False}
pipeline=None;started=False
try:
    if not devices.get_count():raise RuntimeError('No Orbbec camera found')
    device=devices.get_device_by_index(0);info=device.get_device_info()
    result['device']={'name':info.get_name(),'serial':info.get_serial_number(),'firmware':info.get_firmware_version()}
    pipeline=Pipeline(device);config=Config()
    profiles={}
    for sensor in (OBSensorType.COLOR_SENSOR,OBSensorType.DEPTH_SENSOR):
        ps=pipeline.get_stream_profile_list(sensor)
        profiles[str(sensor)]=[str(ps.get_stream_profile_by_index(i)) for i in range(ps.get_count())]
        if sensor==OBSensorType.COLOR_SENSOR and a.color_width==1280:
            profile=ps.get_video_stream_profile(1280,800,OBFormat.YUYV,10)
        else:profile=ps.get_default_video_stream_profile()
        config.enable_stream(profile)
        result[str(sensor)+'_selected']=str(profile)
    result['profiles']=profiles
    print(json.dumps({k:v for k,v in result.items() if k!='profiles'}),flush=True)
    pipeline.start(config);started=True
    align=AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    color=None;depth=None;deadline=time.monotonic()+30
    for i in range(90):
        if time.monotonic()>deadline:break
        frames=pipeline.wait_for_frames(500)
        if frames is None:continue
        if frames.get_color_frame() is None or frames.get_depth_frame() is None:continue
        aligned=align.process(frames)
        if aligned is None:continue
        frames=aligned.as_frame_set()
        color=frames.get_color_frame();depth=frames.get_depth_frame()
        if color is not None and depth is not None and i>=10:break
    if color is None or depth is None:raise RuntimeError('No matched color/depth frames within deadline')
    cp=pipeline.get_camera_param()
    def intrinsic(x):return {k:getattr(x,k) for k in ('width','height','fx','fy','cx','cy')}
    def distortion(x):return {k:getattr(x,k) for k in ('k1','k2','k3','k4','k5','k6','p1','p2') if hasattr(x,k)}
    result['native_factory']={'color_intrinsic':intrinsic(cp.rgb_intrinsic),
        'depth_intrinsic':intrinsic(cp.depth_intrinsic),'color_distortion':distortion(cp.rgb_distortion),
        'depth_distortion':distortion(cp.depth_distortion),
        'depth_to_color_R':np.array(cp.transform.rot).reshape(3,3).tolist(),
        'depth_to_color_t_m':(np.array(cp.transform.transform)*.001).tolist()}
    data=np.frombuffer(color.get_data(),dtype=np.uint8);w=color.get_width();h=color.get_height();fmt=color.get_format()
    if fmt==OBFormat.RGB:rgb=cv2.cvtColor(data.reshape(h,w,3),cv2.COLOR_RGB2BGR)
    elif fmt==OBFormat.BGR:rgb=data.reshape(h,w,3)
    elif fmt==OBFormat.MJPG:rgb=cv2.imdecode(data,cv2.IMREAD_COLOR)
    elif fmt==OBFormat.YUYV:rgb=cv2.cvtColor(data.reshape(h,w,2),cv2.COLOR_YUV2BGR_YUY2)
    else:raise RuntimeError('Unsupported color format '+str(fmt))
    d=np.frombuffer(depth.get_data(),dtype=np.uint16).reshape(depth.get_height(),depth.get_width()).copy()
    scale=float(depth.get_depth_scale())*.001
    cv2.imwrite(str(out/'color.jpg'),rgb);np.save(out/'aligned_depth.npy',d)
    vis=cv2.applyColorMap(cv2.convertScaleAbs(d*scale,alpha=255/2),cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(out/'depth_preview.jpg'),vis)
    result.update(success=True,color_shape=list(rgb.shape),aligned_depth_shape=list(d.shape),
        aligned_depth_scale_m=scale,valid_depth_fraction=float(np.mean(d>0)),
        valid_depth_median_m=float(np.median(d[d>0])*scale) if np.any(d>0) else None,
        color_format=str(fmt))
except BaseException as exc:result['error']=repr(exc)
finally:
    if started:pipeline.stop()
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='profiles'}),flush=True)
raise SystemExit(0 if result['success'] else 1)
