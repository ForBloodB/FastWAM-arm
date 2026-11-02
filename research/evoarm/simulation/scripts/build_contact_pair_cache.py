"""Separate declared rigid tabletop and fingertip contact parameters.

No vendor mesh, object friction range, mass, or actuator change. condim=6
restores the vendor fingertip contact dimension; condim=3 is an ablation.
"""
import argparse,hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
from reflex.official_b601 import MODEL,DERIVED,verify_vendor

def main():
    p=argparse.ArgumentParser();p.add_argument('--dim',type=int,choices=[3,6],required=True);a=p.parse_args()
    verify_vendor();out=Path('build/contact_state');out.mkdir(parents=True,exist_ok=True)
    target=out/f'official_pairs{a.dim}.mjb';assert not target.exists(),target
    tree=ET.parse(MODEL);root=tree.getroot()
    root.find('include').set('file',str(DERIVED/'rs_urdf_aligned_arm.xml'))
    root.find('compiler').set('meshdir',str((DERIVED/root.find('compiler').get('meshdir')).resolve()))
    world=root.find('worldbody')
    for name in ['blue_block','yellow_cylinder','ik_target']:world.remove(world.find(f"body[@name='{name}']"))
    body=world.find("body[@name='red_cube']");body.set('name','task_object');body.find('freejoint').set('name','object_free')
    geom=body.find('geom');geom.set('name','object_geom');geom.set('priority','2')
    contact=root.find('contact')
    ET.SubElement(contact,'pair',{'name':'task_table_pair','geom1':'object_geom','geom2':'task_table_geom',
        'condim':'3','friction':'.25 .25 .005 .0002 .0002','solref':'.003 1','solimp':'.9 .95 .001 .5 2'})
    for side in ['left','right']:
        for part in ['front','mid','rear','carriage','stop']:
            ET.SubElement(contact,'pair',{'name':f'task_{side}_{part}_pair','geom1':'object_geom','geom2':f'gripper_{side}_{part}_collision',
                'condim':str(a.dim),'friction':'.25 .25 .005 .0002 .0002','solref':'.004 1','solimp':'.99 .995 .0005 .5 2'})
    # Inline the included source only to normalize equality reference times.
    # At nominal dt=1 ms, refsafe already clamps 1 ms references to 2 ms.
    inc=root.find('include');root.remove(inc)
    arm=ET.parse(DERIVED/'rs_urdf_aligned_arm.xml').getroot()
    for child in arm:
        if child.tag=='compiler':
            assert child.get('angle')==root.find('compiler').get('angle');continue
        existing=root.find(child.tag)
        if existing is None:root.append(child)
        elif child.tag in ['asset','worldbody','equality','actuator','contact','default']:
            # Include precedes the scene sections; preserve original body,
            # joint, mesh and actuator ordering rather than appending the arm.
            for sub in reversed(list(child)):existing.insert(0,sub)
        else:raise RuntimeError('Unexpected included element '+child.tag)
    normalized=[]
    for eq in root.findall('equality/*'):
        if eq.get('solref'):
            nums=list(map(float,eq.get('solref').split()))
            if 0<nums[0]<.002:
                normalized.append({'name':eq.get('name'),'joint1':eq.get('joint1'),'before':nums.copy()})
                nums[0]=.002;eq.set('solref',' '.join(map(str,nums)))
    xml=ET.tostring(root,encoding='unicode');(out/f'official_pairs{a.dim}.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);mujoco.mj_saveModel(model,str(target),None)
    meta={'created_unix':time.time(),'dim':a.dim,'npair':model.npair,'ngeom':model.ngeom,'nv':model.nv,'nq':model.nq,
          'mujoco':mujoco.__version__,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'equality_refs_normalized':normalized,
          'table_solref':[.003,1.],'table_solimp':[.9,.95,.001,.5,2.],
          'finger_solref':'original class/condition-specific values set at reset, including original soft stress range',
          'friction':'original object tangential mu; original object torsional .005 and rolling .0002',
          'physical_identification':False,'vendor_meshes_changed':False,'force_cap_increased':False}
    (out/f'official_pairs{a.dim}.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
