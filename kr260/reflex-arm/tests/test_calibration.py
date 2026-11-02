import cv2
import numpy as np
import pytest
from reflex.calibration import eye_in_hand_candidate,fixed_camera_candidate,board_model,board_pose,camera_model

def transform(r,t):
    a=np.eye(4);a[:3,:3]=cv2.Rodrigues(np.asarray(r,float))[0];a[:3,3]=t;return a

def test_handeye_direction_and_heldout_rejection():
    rng=np.random.default_rng(23)
    x=transform([.2,-.1,.1],[.03,.01,.07]);y=transform([.1,.3,-.2],[.4,.2,.1])
    samples=[]
    for i in range(20):
        g=transform(rng.uniform(-.6,.6,3),rng.uniform(-.2,.2,3));c=np.linalg.inv(x)@np.linalg.inv(g)@y
        samples.append({'T_base_gripper':g.tolist(),'T_camera_board':c.tolist(),'reprojection_rms_px':.1})
    result=eye_in_hand_candidate(samples)
    np.testing.assert_allclose(result['T_gripper_camera'],x,atol=1e-8)
    assert result['quality_gate_passed'] and not result['hardware_execution_enabled']
    np.testing.assert_allclose(fixed_camera_candidate(y,samples[0]['T_camera_board']),
                               np.asarray(samples[0]['T_base_gripper'])@x,atol=1e-9)
    samples[0]['T_camera_board'][0][3]+=.02
    assert not eye_in_hand_candidate(samples)['quality_gate_passed']
    with pytest.raises(ValueError):eye_in_hand_candidate(samples[:5])
    for i,s in enumerate(samples):s['T_base_gripper']=transform([0,0,i*.02],[i*.01,0,0]).tolist()
    with pytest.raises(ValueError,match='axes'):eye_in_hand_candidate(samples)

def test_print_target_decodes_all_corners():
    b=board_model();image=b.generateImage((1000,1400),marginSize=0,borderBits=1)
    image=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_CONSTANT,value=255)
    corners,ids,_,_=cv2.aruco.CharucoDetector(b).detectBoard(image)
    assert len(ids)==24

def test_perspective_board_metric_pose_and_intrinsic_resolution():
    factory={'native_factory':{'color_intrinsic':{'width':1280,'height':800,'fx':620.,'fy':620.,'cx':640.,'cy':400.},
                              'color_distortion':{k:0. for k in ('k1','k2','p1','p2','k3','k4','k5','k6')}}}
    k,d,_=camera_model(factory);assert len(d)==8
    b=board_model();im=b.generateImage((1000,1400),marginSize=0,borderBits=1)
    r=np.array([.15,-.1,.02]);t=np.array([-.06,-.084,.30])
    obj=np.array([[0,0,0],[.12,0,0],[.12,.168,0],[0,.168,0]])
    dst=cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2).astype(np.float32)
    src=np.array([[0,0],[999,0],[999,1399],[0,1399]],np.float32)
    image=cv2.warpPerspective(im,cv2.getPerspectiveTransform(src,dst),(1280,800),borderValue=255)
    result=board_pose(image,factory)
    estimate=np.array(result['T_camera_board'])
    np.testing.assert_allclose(estimate[:3,3],t,atol=.001)
    angle=np.linalg.norm(cv2.Rodrigues(estimate[:3,:3]@cv2.Rodrigues(r)[0].T)[0])
    assert angle<np.deg2rad(1.) and result['corners']==24
    with pytest.raises(ValueError,match='resolution'):board_pose(image[:400],factory)

def test_stationary_capture_pair_rejects_motion_and_camera_outside_window():
    from pathlib import Path
    from reflex.kinematics import URDFModel
    from reflex.handeye_capture import assemble_pair
    model=URDFModel(Path(__file__).resolve().parents[1]/'assets/B601_RS.urdf')
    robot={'success':True,'q_mean_rad':[.04,.01,.01,0,0,0,.002],'q_range_rad':[.0002]*7}
    board={'success':True,'T_camera_board':np.eye(4).tolist(),'reprojection_rms_px':.2}
    window={'camera_started_after_monitor_ready':True,'camera_completed_before_monitor_end':True}
    result=assemble_pair(model,robot,board,window)
    assert result['pair_accepted'] and not result['camera_extrinsic_calibrated']
    np.testing.assert_allclose(result['T_base_gripper'],model.fk(robot['q_mean_rad'][:6]))
    window['camera_completed_before_monitor_end']=False
    with pytest.raises(ValueError,match='window'):assemble_pair(model,robot,board,window)
    window['camera_completed_before_monitor_end']=True
    robot['q_range_rad'][0]=.002
    with pytest.raises(ValueError,match='moved'):assemble_pair(model,robot,board,window)
