"""URDF kinematics, limits and position IK without loading vendor motor drivers."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis=np.asarray(axis,float)
    axis=axis/np.linalg.norm(axis)
    x,y,z=axis
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def origin_matrix(element):
    t=np.eye(4)
    if element is not None:
        xyz=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        r,p,y=np.fromstring(element.get('rpy','0 0 0'),sep=' ')
        t[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
        t[:3,3]=xyz
    return t


class URDFModel:
    def __init__(self,path,tip='gripper_end'):
        self.path=Path(path)
        self.tree=ET.parse(path).getroot()
        self.tip=tip
        self.joints={j.get('name'):j for j in self.tree.findall('joint')}
        child_to_joint={j.find('child').get('link'):j for j in self.joints.values()}
        self.chain=[]
        link=tip
        seen=set()
        while link in child_to_joint:
            if link in seen: raise ValueError('URDF cycle')
            seen.add(link)
            j=child_to_joint[link]
            self.chain.append(j)
            link=j.find('parent').get('link')
        self.chain.reverse()
        if not self.chain: raise ValueError(f'Tip frame {tip} not found')
        self.names=[j.get('name') for j in self.chain if j.get('type') not in ('fixed',)]
        self.limits=[]
        for name in self.names:
            j=self.joints[name]
            lim=j.find('limit')
            self.limits.append((-np.pi,np.pi) if j.get('type')=='continuous' else (float(lim.get('lower')),float(lim.get('upper'))))
        self.limits=np.asarray(self.limits)

    def fk(self,q):
        q=np.asarray(q,float)
        if q.shape!=(len(self.names),) or not np.isfinite(q).all(): raise ValueError('Invalid joint vector')
        values=dict(zip(self.names,q))
        t=np.eye(4)
        for j in self.chain:
            t=t@origin_matrix(j.find('origin'))
            if j.get('type')!='fixed':
                a=j.find('axis')
                axis=np.fromstring(a.get('xyz','1 0 0') if a is not None else '1 0 0',sep=' ')
                delta=np.eye(4)
                if j.get('type')=='prismatic': delta[:3,3]=axis*values[j.get('name')]
                else: delta[:3,:3]=rotation(axis,values[j.get('name')])
                t=t@delta
        return t

    def ik_position(self,target,initial=None,max_iter=250,tol=1e-4):
        target=np.asarray(target,float)
        if target.shape!=(3,) or not np.isfinite(target).all(): raise ValueError('Invalid target')
        q=np.zeros(len(self.names)) if initial is None else np.asarray(initial,float).copy()
        q=np.clip(q,self.limits[:,0],self.limits[:,1])
        for i in range(max_iter):
            xyz=self.fk(q)[:3,3]
            error=target-xyz
            if np.linalg.norm(error)<tol: break
            jac=np.column_stack([(self.fk(q+np.eye(len(q))[j]*1e-5)[:3,3]-xyz)/1e-5 for j in range(len(q))])
            dq=jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(3),error)
            q=np.clip(q+np.clip(dq,-.15,.15),self.limits[:,0],self.limits[:,1])
        err=float(np.linalg.norm(target-self.fk(q)[:3,3]))
        return {'q_rad':q.tolist(),'error_m':err,'converged':err<tol,'iterations':i+1}


def quintic_trajectory(start,end,duration,hz=200):
    start,end=np.asarray(start,float),np.asarray(end,float)
    if duration<=0 or hz<=0 or start.shape!=end.shape or not np.isfinite(np.r_[start,end]).all():
        raise ValueError('Invalid trajectory arguments')
    t=np.linspace(0,duration,int(np.ceil(duration*hz))+1)
    u=t/duration
    blend=10*u**3-15*u**4+6*u**5
    return t,start[None,:]+blend[:,None]*(end-start)
