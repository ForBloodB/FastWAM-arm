"""Evidence checks for staged wrist PP collection, with no hardware APIs."""
import math


def resolve_pp_prerequisite(load,root,joint,amplitude,direction,guard_sha256):
    """Resolve a reviewed raw PP summary, never an operator's success label.

    The collection coordinator uses ``ppc`` names for some stages. Prefer the
    diagnostic ``pp`` evidence and fall back only when that file is absent.
    Present but failed or malformed evidence must fail closed.
    """
    suffixes={.25:'pilot',.5:'range_0p5',1.:'range_1',2.:'range_2',5.:'range_5',10.:'range_10'}
    if joint not in (5,6) or direction not in (-1,1) or amplitude not in suffixes:
        raise ValueError('Unknown native PP prerequisite axis, direction or amplitude')
    suffix=suffixes[amplitude];side='pos' if direction>0 else 'neg'
    prefixes=['pp']+(['ppc'] if amplitude>=1 else [])
    missing=[]
    for prefix in prefixes:
        path=f'{root.rstrip("/")}/{prefix}_j{joint}_{side}_{suffix}/summary.json'
        try:bench=load(path)
        except FileNotFoundError:
            missing.append(path);continue
        verified_pp_trial(bench,joint,amplitude,direction,guard_sha256)
        return path,bench
    raise FileNotFoundError('Native PP prerequisite evidence absent: '+', '.join(missing))


def verified_pp_trial(bench,joint,amplitude,direction,guard_sha256):
    if not (bench.get('joint')==joint and bench.get('amplitude_motor_deg')==amplitude and
        bench.get('direction')==direction and bench.get('control_mode')=='PP (bounded native position)' and
        bench.get('completed_profile') and bench.get('passed') and bench.get('tracking_criteria_passed') and
        not bench.get('error') and not bench.get('guard_stop_evidence') and bench.get('parameters_restored') and
        not bench.get('restoration_error') and bench.get('target_deliberately_left_disabled') and
        bench.get('no_other_joint_writes') and bench.get('parameter_persistence_writes')==0 and
        bench.get('gain_parameters_changed') is False and
        bench.get('source_sha256',{}).get('motion.py')==guard_sha256):
        raise ValueError('Native PP prerequisite lacks passing protected tracking/restoration evidence')
    params=bench.get('post_enable_parameters',{})
    if not (params.get('0x7005')==1 and params.get('0x7028')==2000 and
        math.isclose(params.get('0x700b',0),.5,abs_tol=1e-6) and
        math.isclose(params.get('0x7018',0),.6,abs_tol=1e-6) and
        math.isclose(params.get('0x7024',0),math.radians(.25),abs_tol=1e-7) and
        math.isclose(params.get('0x7025',0),math.radians(.25),abs_tol=1e-7)):
        raise ValueError('Native PP prerequisite planning/current/torque/watchdog limits differ')
    states=bench.get('final_status',{})
    target=states.get(str(joint),states.get(joint,{}))
    if len(states)!=7 or any(s.get('fault_mask_raw',1) for s in states.values()) or target.get('mode_bits_raw')!=0:
        raise ValueError('Native PP prerequisite final motor states do not verify a disabled fault-free target')
    return True
