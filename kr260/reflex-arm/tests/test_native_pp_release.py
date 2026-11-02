import copy
import math
import pytest
from reflex.native_pp_release import resolve_pp_prerequisite,verified_pp_trial


def trial():
    return {'joint':6,'amplitude_motor_deg':2.,'direction':-1,'control_mode':'PP (bounded native position)',
        'completed_profile':True,'passed':True,'tracking_criteria_passed':True,'parameters_restored':True,
        'target_deliberately_left_disabled':True,'no_other_joint_writes':True,'parameter_persistence_writes':0,
        'gain_parameters_changed':False,'source_sha256':{'motion.py':'guard'},
        'post_enable_parameters':{'0x7005':1,'0x7028':2000,'0x700b':.5,'0x7018':.6,
                                 '0x7024':math.radians(.25),'0x7025':math.radians(.25)},
        'final_status':{str(j):{'fault_mask_raw':0,'mode_bits_raw':0 if j in (5,6) else 2} for j in range(1,8)}}


def test_requires_correct_axis_direction_amplitude_and_guard_source():
    assert verified_pp_trial(trial(),6,2.,-1,'guard')
    for args in ((5,2.,-1,'guard'),(6,1.,-1,'guard'),(6,2.,1,'guard'),(6,2.,-1,'different')):
        with pytest.raises(ValueError):verified_pp_trial(trial(),*args)


@pytest.mark.parametrize('change',[{'passed':False},{'tracking_criteria_passed':False},
    {'guard_stop_evidence':{'source':'position_displacement'}},{'parameters_restored':False},
    {'restoration_error':'readback'}, {'gain_parameters_changed':True},{'parameter_persistence_writes':1}])
def test_no_failed_or_changed_trial_releases_collection(change):
    value=trial();value.update(change)
    with pytest.raises(ValueError):verified_pp_trial(value,6,2.,-1,'guard')


def test_no_loose_native_planning_limit_or_enabled_final_state_releases_collection():
    for key,value in (('0x7024',1.),('0x7025',10.),('0x7018',16.),('0x700b',14.),('0x7028',0)):
        data=trial();data['post_enable_parameters'][key]=value
        with pytest.raises(ValueError):verified_pp_trial(data,6,2.,-1,'guard')
    data=trial();data['final_status']['6']['mode_bits_raw']=2
    with pytest.raises(ValueError):verified_pp_trial(data,6,2.,-1,'guard')


def loader(files,requested):
    def load(path):
        requested.append(path)
        if path not in files:raise FileNotFoundError(path)
        return files[path]
    return load


def test_j5_diagnostic_can_use_existing_passing_collection_stage_as_prerequisite():
    # J5's 1-degree stage was collected under ppc names before a 2-degree
    # diagnostic. Both signs must still resolve the expected raw evidence.
    for direction,side in ((1,'pos'),(-1,'neg')):
        data=trial();data.update(joint=5,amplitude_motor_deg=1.,direction=direction)
        path=f'/runs/ppc_j5_{side}_range_1/summary.json';requested=[]
        resolved,proof=resolve_pp_prerequisite(loader({path:data},requested),'/runs',5,1.,direction,'guard')
        assert resolved==path and proof is data
        assert requested==[f'/runs/pp_j5_{side}_range_1/summary.json',path]


def test_existing_diagnostic_evidence_is_preferred_without_reading_fallback():
    data=trial();path='/runs/pp_j6_neg_range_2/summary.json';requested=[]
    assert resolve_pp_prerequisite(loader({path:data},requested),'/runs',6,2.,-1,'guard')==(path,data)
    assert requested==[path]


@pytest.mark.parametrize('change',[{'joint':6},{'amplitude_motor_deg':2.},{'direction':-1},
    {'passed':False},{'tracking_criteria_passed':False},{'gain_parameters_changed':True}])
def test_named_collection_fallback_still_verifies_exact_raw_trial(change):
    data=trial();data.update(joint=5,amplitude_motor_deg=1.,direction=1);data.update(change)
    path='/runs/ppc_j5_pos_range_1/summary.json'
    with pytest.raises(ValueError):
        resolve_pp_prerequisite(loader({path:data},[]),'/runs',5,1.,1,'guard')


def test_failed_positive_two_degree_collection_cannot_unlock_larger_stage():
    data=trial();data.update(joint=5,direction=1,passed=False,tracking_criteria_passed=False)
    with pytest.raises(ValueError):
        resolve_pp_prerequisite(loader({'/runs/ppc_j5_pos_range_2/summary.json':data},[]),'/runs',5,2.,1,'guard')


def test_present_but_failed_diagnostic_does_not_fall_back_to_another_run():
    failed=trial();failed['passed']=False;requested=[]
    files={'/runs/pp_j6_neg_range_2/summary.json':failed,'/runs/ppc_j6_neg_range_2/summary.json':trial()}
    with pytest.raises(ValueError):resolve_pp_prerequisite(loader(files,requested),'/runs',6,2.,-1,'guard')
    assert requested==['/runs/pp_j6_neg_range_2/summary.json']


def test_prerequisite_resolver_rejects_unknown_scope_and_propagates_read_errors():
    for args in ((7,1.,1),(5,.75,1),(5,1.,0)):
        requested=[]
        with pytest.raises(ValueError):resolve_pp_prerequisite(loader({},requested),'/runs',*args,'guard')
        assert requested==[]
    def denied(path):raise PermissionError(path)
    with pytest.raises(PermissionError):resolve_pp_prerequisite(denied,'/runs',5,1.,1,'guard')
    with pytest.raises(FileNotFoundError):resolve_pp_prerequisite(loader({},[]),'/runs',5,1.,1,'guard')
