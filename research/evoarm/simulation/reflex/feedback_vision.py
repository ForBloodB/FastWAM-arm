"""RGB-D observation adapter, including a bit-exact optional physical PL backend.

Only RGB, depth, calibrated camera transform and declared table/workspace enter
the detector. No simulator object IDs, pose, material, mass or friction inputs.
"""
import json, socket, struct, time
import cv2
import numpy as np

def pixel_labels(rgb, foreground):
    r,g,b=[rgb[:,:,i].astype(np.uint16) for i in range(3)]
    masks=[(r>50)&(r>2*g)&(r>2*b), (b>50)&(b>2*r)&(b>2*g),
           (r>50)&(g>40)&(2*b<r)&(2*b<g)&(r<2*g)&(g<2*r),
           (r>50)&(b>40)&(2*g<r)&(2*g<b)&(r<2*b)&(b<2*r)]
    labels=np.zeros(r.shape,np.uint8)
    for k,mask in enumerate(masks,1):labels[mask & foreground & (labels==0)]=k
    return labels

def recv_exact(sock,n):
    data=bytearray()
    while len(data)<n:
        part=sock.recv(n-len(data))
        if not part:raise ConnectionError('Truncated PL response')
        data.extend(part)
    return bytes(data)

class BoardLabels:
    def __init__(self,host='127.0.0.1',port=18766):
        self.sock=socket.create_connection((host,port),timeout=10);self.sock.settimeout(15);self.calls=[]
    def __call__(self,rgb,foreground):
        h,w=foreground.shape;t0=time.perf_counter_ns()
        payload=np.concatenate([rgb,np.asarray(foreground,np.uint8)[:,:,None]],axis=2).tobytes()
        self.sock.sendall(struct.pack('!4sHHI',b'RV01',w,h,len(payload))+payload)
        n=struct.unpack('!I',recv_exact(self.sock,4))[0]
        if n>8192:raise RuntimeError('Invalid PL response length')
        meta=json.loads(recv_exact(self.sock,n))
        if not meta.get('ok'):raise RuntimeError(meta)
        labels=np.frombuffer(recv_exact(self.sock,h*w),np.uint8).reshape(h,w).copy()
        meta['pc_roundtrip_ms']=(time.perf_counter_ns()-t0)/1e6;self.calls.append(meta)
        if not np.array_equal(labels,pixel_labels(rgb,foreground)):raise RuntimeError('PL/CPU segmentation mismatch')
        return labels
    def close(self):self.sock.close()

def detect(rgb,depth,transform,fovy,backend=pixel_labels,height=None,wanted_label=None,prior_xy=None,prior_size=None):
    t0=time.perf_counter_ns();h,w=depth.shape;f=(h/2)/np.tan(np.deg2rad(fovy)/2)
    v,u=np.indices(depth.shape)
    cam=np.stack([(u-w/2)*depth/f,-(v-h/2)*depth/f,-depth],axis=-1)
    world=cam@transform[:3,:3].T+transform[:3,3]
    fg=np.isfinite(depth)&(depth>0)&(world[:,:,0]>.23)&(world[:,:,0]<.40)&(world[:,:,1]>-.18)&(world[:,:,1]<.16)&(world[:,:,2]>.108)&(world[:,:,2]<.32)
    if prior_xy is not None:fg &= np.linalg.norm(world[:,:,:2]-np.asarray(prior_xy),axis=2)<.038
    labels=backend(rgb,fg);candidates=[]
    for label in ([wanted_label] if wanted_label else range(1,5)):
        count,components,stats,_=cv2.connectedComponentsWithStats((labels==label).astype(np.uint8))
        for k in range(1,count):
            if stats[k,cv2.CC_STAT_AREA]>=12:candidates.append((int(stats[k,cv2.CC_STAT_AREA]),label,components==k))
    if not candidates:return {'valid':False,'pixels':0,'wall_ms':(time.perf_counter_ns()-t0)/1e6},labels
    n,label,mask=max(candidates,key=lambda row:row[0]);points=world[mask]
    top=float(np.percentile(points[:,2],80));surface=points[points[:,2]>=top-.002]
    if len(surface)<12:surface=points
    xy=surface[:,:2];values,basis=np.linalg.eigh(np.cov(xy.T));axis=basis[:,-1]
    yaw=float((np.arctan2(axis[1],axis[0])+np.pi/2)%np.pi-np.pi/2)
    projected=xy@basis;extent=np.ptp(projected,axis=0);mid=(projected.min(axis=0)+projected.max(axis=0))/2;center=mid@basis.T
    observed_height=top-.1 if height is None else height
    size=[float(max(extent)),float(min(extent)),float(observed_height)]
    pose_valid=bool(.012<size[1]<.065 and .02<size[0]<.085 and .02<observed_height<.07)
    if prior_size is not None:pose_valid &= bool(size[0]>.7*prior_size[0] and size[1]>.7*prior_size[1])
    valid=pose_valid if height is None else bool(.02<observed_height<.07 and len(points)>=12)
    return {'valid':valid,'pose_valid':pose_valid,'label':int(label),'xyz':[float(center[0]),float(center[1]),float(top-observed_height/2)],
            'yaw':yaw,'size':size,'pixels':n,'wall_ms':(time.perf_counter_ns()-t0)/1e6},labels
