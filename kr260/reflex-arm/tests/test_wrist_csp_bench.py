"""Exercise the actual bench entry point with a fake CAN bus and fake clock.

Checks target isolation, disable-before-mode-change, and abort/restoration when
another joint moves. No CAN socket or scheduler mutation is allowed in tests.
"""
import importlib
import json
from pathlib import Path
import runpy
import struct
import tempfile
import types
import unittest
from unittest.mock import patch


class BenchIsolationTests(unittest.TestCase):
    def run_bench(self, mid, neighbour_drift=False, release_shift=0., gripper_options=None, tracking_scale=1.,original_speed_kp=None,reset_limits_on_enable=False,velocity_rad_s=0.):
        clock=types.SimpleNamespace(t=100.)
        calls=[];instances=[]
        def sleep(seconds):clock.t+=max(0.,seconds)
        def ns():return int(clock.t*1e9)

        class Bus:
            def __init__(self):
                self.sock=self;self.failed=False;self.capture=[];self.mode={j:2 for j in range(1,8)}
                self.q={j:(.02 if j==7 else 0.) for j in range(1,8)};self.latest_status={};self.enable_time=None;self.released=False
                self.params={0x700B:36. if mid in (2,3) else 14.,0x7028:0,
                    0x7018:57. if mid in (2,3) else 16.,0x7017:33.,0x7016:0.,0x7005:0,
                    0x7024:1.,0x7025:10.,
                    0x701F:(5. if mid==6 else 8.) if original_speed_kp is None else original_speed_kp}
                self.pp_target=dict(self.q);self.last_t=clock.t
                instances.append(self);self.refresh()
            def refresh(self):
                dt=max(0.,clock.t-self.last_t);self.last_t=clock.t
                if self.mode[mid]==2 and self.params[0x7005]==1:
                    error=self.pp_target[mid]-self.q[mid];step=self.params[0x7024]*dt
                    self.q[mid]+=max(-step,min(step,error))
                for j in range(1,8):
                    self.latest_status[j]={'rx_ns':ns(),'mode_bits_raw':self.mode[j],
                        'fault_mask_raw':0,'temperature_C':30.,'motor_id':j}
            def poll(self, seconds=.05):sleep(seconds);self.refresh()
            def drain(self):self.refresh()
            def read(self,j,param,timeout=.3):
                clock.t+=.00005;self.refresh()
                if param==0x7019:
                    if neighbour_drift and j==3 and self.enable_time and clock.t-self.enable_time>.25:return .02
                    return self.q[j]
                if param==0x701B:return velocity_rad_s
                if param==0x701A:return 0.
                if param==0x701C:return 48.
                return self.params[param]
            def send(self,frame):
                canid,_,payload=struct.unpack('=IB3x8s',frame);kind=(canid>>24)&31;j=canid&255
                calls.append((kind,j,payload))
                if kind==4:
                    self.mode[j]=0
                    if not self.released:
                        self.q[j]+=release_shift;self.params[0x7016]=self.q[j];self.released=True
                elif kind==3:
                    self.mode[j]=2;self.enable_time=clock.t;self.anchor=self.q[j]
                    if reset_limits_on_enable:self.params[0x7017]=33.
                    if reset_limits_on_enable and self.params[0x7005]==1:self.params[0x7024]=1.
                elif kind==18:
                    param=struct.unpack_from('<H',payload)[0]
                    dtype='b' if param==0x7005 else ('I' if param==0x7028 else 'f')
                    value=struct.unpack_from('<'+dtype,payload,4)[0]
                    if param in (0x7005,0x7018,0x7017,0x700B,0x7028,0x701F,0x7024,0x7025):
                        assert self.mode[j]==0, 'Mode/limit change while enabled'
                    self.params[param]=value
                    if param==0x7016:
                        if self.params[0x7005]==1:self.pp_target[j]=value
                        elif self.mode[j]==2:self.q[j]=self.anchor+tracking_scale*(value-self.anchor)
                elif kind==1:
                    import math
                    pos,vel,kp,kd=struct.unpack('>4H',payload)
                    if kp and self.mode[j]==2:
                        target=pos/65535*8*math.pi-4*math.pi
                        self.q[j]=self.anchor+tracking_scale*(target-self.anchor)
                self.refresh()
            def close(self):pass

        canbus=importlib.import_module('reflex.canbus')
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lock=root/'lock';lock.touch();out=root/'out'
            import os
            original_open=os.open
            def open_lock(path,*args,**kwargs):
                return original_open(lock if path=='/tmp/reflex-can-actuation.lock' else path,*args,**kwargs)
            with patch.object(canbus,'ReadOnlyCAN',Bus),patch('os.open',open_lock), \
                 patch('os.sched_setaffinity'),patch('os.sched_setscheduler'), \
                 patch('os.sched_getscheduler',return_value=0),patch('os.sched_getparam',return_value=os.sched_param(0)), \
                 patch('os.sched_getaffinity',return_value={0,1,2,3}), \
                 patch('time.monotonic',side_effect=lambda:clock.t),patch('time.monotonic_ns',side_effect=ns), \
                 patch('time.sleep',side_effect=sleep),patch('signal.signal'), \
                 patch('sys.argv',[str(script),'--joint',str(mid),'--out',str(out),'--execute-authorized-test']+
                       (['--load-support-verified'] if mid in (2,3,4) else [])+(gripper_options or [])), \
                 patch('builtins.print'):
                with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
            result=json.loads((out/'summary.json').read_text())
            instances[-1].test_samples=json.loads((out/'samples.json').read_text())
        return result,calls,done.exception.code,instances[-1]

    def test_completed_profiles_only_write_selected_low_load_joint(self):
        for joint in (2,3,4,5,6,7):
            with self.subTest(joint=joint):
                result,calls,code,bus=self.run_bench(joint)
                self.assertEqual(code,0)
                self.assertTrue(result['passed'])
                self.assertEqual({j for _,j,_ in calls},{joint})
                self.assertTrue(set(k for k,_,_ in calls).issubset({1,3,4,18}))
                self.assertEqual(bus.params[0x7005],0)
                self.assertEqual(bus.params[0x7018],57. if joint in (2,3) else 16.)
                self.assertTrue(result['initial_enable_state_restored'])

    def test_other_joint_movement_aborts_and_leaves_target_disabled(self):
        result,calls,code,bus=self.run_bench(5,True)
        self.assertEqual(code,1)
        self.assertIn('Uncommanded joint displacement',result['error'])
        self.assertEqual(bus.mode[5],0)
        self.assertEqual({j for _,j,_ in calls},{5})
        self.assertTrue(result['parameters_restored'])

    def test_small_release_shift_reanchors_without_zeroing_or_old_target_replay(self):
        result,calls,code,bus=self.run_bench(6,release_shift=.0012)
        self.assertEqual(code,0)
        self.assertAlmostEqual(result['anchor_motor_rad'],.0012)
        self.assertEqual({j for _,j,_ in calls},{6})
        self.assertNotIn(6,{kind for kind,_,_ in calls})
        self.assertAlmostEqual(result['final_error_motor_deg'],0.,places=4)

    def test_conservative_reverse_gripper_excursion(self):
        result,calls,code,bus=self.run_bench(7,gripper_options=['--gripper-amplitude','0.5','--gripper-direction','-1'])
        self.assertEqual(code,0)
        self.assertEqual(result['direction'],-1)
        self.assertTrue(result['passed'])
        self.assertEqual({j for _,j,_ in calls},{7})

    def test_j4_reduced_gain_is_volatile_and_restored(self):
        result,calls,code,bus=self.run_bench(4,gripper_options=['--j4-speed-kp','4'])
        self.assertEqual(code,0)
        self.assertEqual(result['temporary_speed_kp'],4.)
        self.assertEqual(bus.params[0x701F],8.)

    def test_j6_half_gain_is_scoped_volatile_and_keeps_motion_guards(self):
        result,calls,code,bus=self.run_bench(6,gripper_options=['--j6-speed-kp','2.5'],original_speed_kp=5.)
        self.assertEqual(code,0)
        self.assertEqual(result['temporary_speed_kp'],2.5)
        self.assertEqual(bus.params[0x701F],5.)
        self.assertEqual(result['current_limit_A'],.6)
        self.assertEqual(result['speed_limit_motor_deg_s'],.25)
        self.assertEqual({j for _,j,_ in calls},{6})
        self.assertEqual(result['parameter_persistence_writes'],0)

    def test_j6_default_and_already_repaired_device_both_use_verified_gain(self):
        for native in (5.,2.5):
            result,calls,code,bus=self.run_bench(6,original_speed_kp=native)
            self.assertEqual(code,0)
            self.assertEqual(result['temporary_speed_kp'],2.5)
            self.assertAlmostEqual(result['post_enable_parameters']['0x701f'],2.5)
            self.assertEqual(bus.params[0x701F],native)

    def test_web_mit_baseline_matches_ui_gains_without_csp_or_speed_gain_writes(self):
        for joint in (5,6):
            result,calls,code,bus=self.run_bench(joint,gripper_options=['--web-mit-profile',
                '--wrist-duration','12','--wrist-direction','-1'])
            self.assertEqual(code,0)
            self.assertTrue(result['passed'])
            self.assertEqual(result['mit_kp'],30.)
            self.assertEqual(result['mit_kd'],1.)
            self.assertTrue(result['target_deliberately_left_disabled'])
            self.assertEqual(bus.mode[joint],0)
            writes={struct.unpack_from('<H',p)[0] for k,_,p in calls if k==18}
            self.assertEqual(writes,{0x700B,0x7028})
            self.assertEqual({j for _,j,_ in calls},{joint})
            self.assertIsNone(result['current_limit_A'])
            self.assertIsNone(result['speed_limit_motor_deg_s'])
            self.assertTrue(result['parameters_restored'])
            active=[p for k,_,p in calls if k==1 and struct.unpack('>4H',p)[2]]
            for p in active:
                _,vel,kp,kd=struct.unpack('>4H',p)
                self.assertEqual((vel,kp,kd),(32767,int(30/500*65535),int(1/5*65535)))

    def test_web_mit_baseline_rejects_expansion_before_opening_can(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        for args in (['--joint','6','--wrist-amplitude','.5','--wrist-duration','12'],
                     ['--joint','7'],['--joint','6','--j6-speed-kp','2.5'],
                     ['--joint','6','--web-mit-range-check','--wrist-amplitude','2','--wrist-duration','40','--repetitions','3']):
            with patch('sys.argv',[str(script),'--out','/unused','--web-mit-profile','--execute-authorized-test']+args), \
                patch('reflex.canbus.ReadOnlyCAN') as bus,patch('sys.stderr'):
                with self.assertRaises(SystemExit):runpy.run_path(str(script),run_name='__main__')
                bus.assert_not_called()

    def test_web_mit_range_characterization_keeps_gains_effort_and_original_tracking_criteria(self):
        result,calls,code,bus=self.run_bench(6,gripper_options=['--web-mit-profile','--web-mit-range-check',
            '--wrist-amplitude','.5','--wrist-duration','12'])
        self.assertEqual(code,0)
        self.assertEqual(result['torque_limit_Nm'],.5)
        self.assertEqual(result['software_current_limit_A'],.75)
        self.assertEqual(result['tracking_criteria']['absolute_return_error_max_motor_deg'],.05)
        self.assertAlmostEqual(result['tracking_criteria']['tracking_rmse_max_motor_deg'],.15)
        self.assertTrue(result['target_deliberately_left_disabled'])
        self.assertEqual({j for _,j,_ in calls},{6})

    def test_web_mit_capture_event_keeps_twelve_seconds_of_bounded_hold(self):
        result,calls,code,bus=self.run_bench(6,gripper_options=['--web-mit-profile','--web-mit-range-check',
            '--wrist-amplitude','1','--wrist-duration','20','--wrist-peak-hold','20','--wrist-return-hold','8',
            '--announce-peak','--peak-settle-seconds','8'])
        self.assertEqual(code,0)
        self.assertGreater(result['peak_capture_event']['remaining_hold_s'],11.9)
        self.assertLessEqual(result['peak_capture_event']['remaining_hold_s'],12.01)
        self.assertTrue(result['target_deliberately_left_disabled'])

    def test_j6_damping_comparison_changes_only_frame_damping(self):
        result,calls,code,bus=self.run_bench(6,gripper_options=['--web-mit-profile','--mit-damping-comparison',
            '--wrist-duration','12','--wrist-peak-hold','20','--wrist-return-hold','8'])
        self.assertEqual(code,0)
        self.assertEqual(result['control_mode'],'MIT (damping comparison)')
        self.assertEqual((result['mit_kp'],result['mit_kd']),(30.,2.))
        active=[struct.unpack('>4H',payload) for kind,j,payload in calls if kind==1 and struct.unpack('>4H',payload)[2]]
        self.assertTrue(active)
        self.assertEqual({item[3] for item in active},{int(2./5.*65535)})
        writes=[struct.unpack_from('<H',payload)[0] for kind,j,payload in calls if kind==18]
        self.assertEqual(set(writes),{0x700B,0x7028})
        self.assertEqual({j for _,j,_ in calls},{6})
        self.assertTrue(result['target_deliberately_left_disabled'])
        self.assertTrue(result['parameters_restored'])
        self.assertTrue(all(abs(row['estimated_MIT_torque_Nm'])<=.5 for row in bus.test_samples))

    def test_damping_comparison_rejects_unreviewed_joint_range_and_rate_before_can(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        for options in (['--joint','5','--web-mit-profile'],['--joint','6'],
                        ['--joint','6','--web-mit-profile','--web-mit-range-check','--wrist-amplitude','5','--wrist-duration','80'],
                        ['--joint','6','--web-mit-profile','--web-mit-range-check','--wrist-amplitude','2','--wrist-duration','40'],
                        ['--joint','6','--web-mit-profile','--repetitions','3']):
            with patch('sys.argv',[str(script),'--out','/unused','--mit-damping-comparison','--execute-authorized-test']+options), \
                 patch('reflex.canbus.ReadOnlyCAN') as bus,patch('sys.stderr'):
                with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
                self.assertEqual(done.exception.code,2)
                bus.assert_not_called()

    def test_damping_comparison_preserves_estimated_effort_stop_and_restoration(self):
        result,calls,code,bus=self.run_bench(6,velocity_rad_s=.255,gripper_options=[
            '--web-mit-profile','--mit-damping-comparison','--wrist-duration','12'])
        self.assertEqual(code,1)
        self.assertFalse(result['completed_profile'])
        self.assertIn('estimated effort',result['error'])
        self.assertTrue(result['parameters_restored'])
        self.assertTrue(result['target_deliberately_left_disabled'])
        self.assertEqual({j for _,j,_ in calls},{6})

    def test_native_pp_sets_correct_planning_registers_and_restores_without_gain_write(self):
        for joint,direction in ((6,1),(6,-1),(5,1),(5,-1)):
            result,calls,code,bus=self.run_bench(joint,gripper_options=['--native-pp-profile','--wrist-duration','20',
                '--wrist-direction',str(direction),'--wrist-peak-hold','20','--wrist-return-hold','8'])
            self.assertEqual(code,0)
            self.assertEqual(result['control_mode'],'PP (bounded native position)')
            self.assertTrue(result['tracking_criteria_passed'])
            self.assertFalse(result['gain_parameters_changed'])
            self.assertEqual(result['native_pp_velocity_deg_s'],.25)
            self.assertEqual(result['native_pp_acceleration_deg_s2'],.25)
            self.assertEqual({j for _,j,_ in calls},{joint})
            writes={struct.unpack_from('<H',payload)[0] for kind,j,payload in calls if kind==18}
            self.assertEqual(writes,{0x7005,0x700B,0x7028,0x7018,0x7024,0x7025,0x7016})
            self.assertEqual(bus.params[0x7024],1.)
            self.assertEqual(bus.params[0x7025],10.)
            self.assertEqual(bus.params[0x7005],0)
            self.assertEqual(bus.params[0x7018],16.)
            self.assertTrue(result['parameters_restored'])
            self.assertTrue(result['target_deliberately_left_disabled'])

    def test_native_pp_enable_reset_or_neighbour_motion_still_aborts(self):
        options=['--native-pp-profile','--wrist-duration','20','--wrist-peak-hold','20','--wrist-return-hold','8']
        for cause in ('reset','neighbour'):
            result,calls,code,bus=self.run_bench(6,gripper_options=options,
                reset_limits_on_enable=cause=='reset',neighbour_drift=cause=='neighbour')
            self.assertEqual(code,1)
            self.assertFalse(result['completed_profile'])
            self.assertTrue(result['parameters_restored'])
            self.assertTrue(result['target_deliberately_left_disabled'])
            self.assertEqual(bus.params[0x7024],1.)
            self.assertEqual(bus.params[0x7025],10.)
            self.assertEqual({j for _,j,_ in calls},{6})

    def test_native_pp_rejects_other_axes_mixed_modes_gains_or_expansion_before_can(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        for options in (['--joint','7'],['--joint','6','--web-mit-profile'],
                        ['--joint','6','--j6-speed-kp','2.5'],['--joint','6','--repetitions','3'],
                        ['--joint','6','--wrist-amplitude','5','--wrist-duration','160']):
            with patch('sys.argv',[str(script),'--out','/unused','--native-pp-profile','--execute-authorized-test']+options), \
                 patch('reflex.canbus.ReadOnlyCAN') as bus,patch('sys.stderr'):
                with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
                self.assertEqual(done.exception.code,2)
                bus.assert_not_called()

    def test_native_pp_rejects_changed_effort_limits_before_creating_can(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        for mode,scope in (('--native-pp-profile',['--wrist-duration','20']),
                ('--native-pp-calibration',['--wrist-amplitude','1','--wrist-duration','40',
                    '--wrist-peak-hold','20','--announce-peak','--peak-settle-seconds','8'])):
            for effort in (['--j5-current','0.7'],['--j5-torque-limit','0.6'],
                    ['--j5-current','0.7','--j5-torque-limit','0.6']):
                with patch('sys.argv',[str(script),'--joint','5','--out','/unused',mode,
                            '--execute-authorized-test']+scope+effort), \
                     patch('reflex.canbus.ReadOnlyCAN') as bus,patch('os.open') as lock_open,patch('sys.stderr'):
                    with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
                    self.assertEqual(done.exception.code,2)
                    bus.assert_not_called();lock_open.assert_not_called()

    def test_pp_calibration_requires_explicit_capture_and_keeps_reviewed_hold(self):
        result,calls,code,bus=self.run_bench(5,gripper_options=['--native-pp-calibration',
            '--wrist-amplitude','1','--wrist-duration','40','--wrist-peak-hold','20','--wrist-return-hold','8',
            '--announce-peak','--peak-settle-seconds','8'])
        self.assertEqual(code,0)
        self.assertTrue(result['capture_qualification_requested'])
        self.assertTrue(result['passed'])
        self.assertGreater(result['peak_capture_event']['remaining_hold_s'],11.9)
        self.assertLessEqual(result['peak_capture_event']['remaining_hold_s'],12.01)
        self.assertTrue(result['target_deliberately_left_disabled'])

    def test_pp_calibration_does_not_admit_missing_capture_window_before_can(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        for options in (['--joint','6'],['--joint','5','--wrist-amplitude','1','--wrist-duration','40'],
                        ['--joint','6','--native-pp-profile']):
            with patch('sys.argv',[str(script),'--out','/unused','--native-pp-calibration','--execute-authorized-test']+options), \
                 patch('reflex.canbus.ReadOnlyCAN') as bus,patch('sys.stderr'):
                with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
                self.assertEqual(done.exception.code,2)
                bus.assert_not_called()

    def test_j5_torque_comparison_preserves_current_limit_and_restores(self):
        result,calls,code,bus=self.run_bench(5,gripper_options=['--j5-torque-limit','0.6','--j5-current','0.7',
            '--wrist-amplitude','2','--wrist-duration','40','--repetitions','3'])
        self.assertEqual(code,0)
        self.assertEqual(result['torque_limit_Nm'],.6)
        self.assertEqual(result['current_limit_A'],.7)
        self.assertEqual(bus.params[0x700B],14.)
        self.assertEqual(bus.params[0x7018],16.)
        self.assertEqual(len(result['cycle_results']),3)
        self.assertEqual({j for _,j,_ in calls},{5})

    def test_enable_reset_of_speed_limit_aborts_before_moving_reference(self):
        result,calls,code,bus=self.run_bench(6,reset_limits_on_enable=True)
        self.assertEqual(code,1)
        self.assertEqual(result['failure_phase'],'enable_verification')
        self.assertIn('configuration changed',result['error'])
        self.assertEqual(result['samples'],0)
        self.assertEqual(bus.mode[6],0)
        self.assertTrue(result['parameters_restored'])

    def test_loaded_axis_half_gain_is_volatile_and_restored(self):
        for joint in (2,3):
            with self.subTest(joint=joint):
                result,calls,code,bus=self.run_bench(joint,gripper_options=['--loaded-speed-scale','0.5','--loaded-duration','30'])
                self.assertEqual(code,0)
                self.assertEqual(result['temporary_speed_kp'],4.)
                self.assertEqual(result['profile_duration_s'],30.)
                self.assertEqual(bus.params[0x701F],8.)
                self.assertTrue(result['tracking_criteria_passed'])
                self.assertEqual({j for _,j,_ in calls},{joint})

    def test_loaded_gain_options_rejected_for_other_joints_before_can(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        for args in (['--joint','7','--loaded-speed-scale','0.5'],['--joint','5','--loaded-duration','30'],
                     ['--joint','7','--loaded-peak-hold','8'],['--joint','5','--loaded-return-hold','8'],
                     ['--joint','2','--wrist-duration','12','--load-support-verified'],
                     ['--joint','5','--j6-speed-kp','2.5'],['--joint','6','--j5-current','0.7'],
                     ['--joint','6','--j5-torque-limit','0.6']):
            with self.subTest(args=args),patch('sys.argv',[str(script),'--out','/unused','--execute-authorized-test']+args), \
                 patch('reflex.canbus.ReadOnlyCAN') as bus,patch('sys.stderr'):
                with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
                self.assertEqual(done.exception.code,2)
                bus.assert_not_called()

    def test_undershoot_is_not_a_tracking_pass(self):
        result,calls,code,bus=self.run_bench(2,tracking_scale=.3)
        self.assertEqual(code,1)
        self.assertTrue(result['completed_profile'])
        self.assertTrue(result['movement_diagnostic_passed'])
        self.assertFalse(result['tracking_criteria_passed'])
        self.assertFalse(result['passed'])
        self.assertTrue(result['parameters_restored'])

    def test_bounded_dwell_holds_peak_and_returns_without_gain_change(self):
        import math
        result,calls,code,bus=self.run_bench(3,gripper_options=['--loaded-duration','30',
            '--loaded-peak-hold','8','--loaded-return-hold','8'])
        self.assertEqual(code,0)
        self.assertEqual(result['peak_hold_s'],8.)
        self.assertEqual(result['return_hold_s'],8.)
        self.assertEqual(bus.params[0x701F],8.)
        samples=bus.test_samples
        peak=[r for r in samples if 16.1<r['elapsed_s']<23.9]
        end=[r for r in samples if 39.1<r['elapsed_s']<46.9]
        self.assertGreater(len(peak),770)
        self.assertGreater(len(end),770)
        self.assertTrue(all(abs(r['command_rad']-math.radians(.25))<1e-7 for r in peak))
        self.assertTrue(all(abs(r['command_rad'])<1e-7 for r in end))
        self.assertEqual({j for _,j,_ in calls},{3})
        self.assertTrue(result['parameters_restored'])

    def test_slow_wrist_profile_retains_original_excursion_and_limits(self):
        result,calls,code,bus=self.run_bench(6,gripper_options=['--wrist-duration','12'])
        self.assertEqual(code,0)
        self.assertEqual(result['profile_duration_s'],12.)
        self.assertEqual(result['amplitude_motor_deg'],.25)
        self.assertEqual(result['speed_limit_motor_deg_s'],.25)
        self.assertEqual(result['current_limit_A'],.6)
        self.assertEqual(bus.params[0x7018],16.)
        self.assertTrue(result['parameters_restored'])
        self.assertEqual({j for _,j,_ in calls},{6})

    def test_loaded_axis_requires_explicit_support_prerequisite(self):
        script=Path(__file__).resolve().parents[1]/'scripts/wrist_csp_bench.py'
        with patch('sys.argv',[str(script),'--joint','3','--out','/unused','--execute-authorized-test']), \
             patch('reflex.canbus.ReadOnlyCAN') as bus,patch('sys.stderr'):
            with self.assertRaises(SystemExit) as done:runpy.run_path(str(script),run_name='__main__')
            self.assertEqual(done.exception.code,2)
            bus.assert_not_called()

    def test_ten_cycles_keep_one_absolute_anchor_and_all_guards(self):
        result,calls,code,bus=self.run_bench(5,gripper_options=['--repetitions','10'])
        self.assertEqual(code,0)
        self.assertEqual(len(result['cycle_results']),10)
        self.assertTrue(result['fixed_anchor_between_cycles'])
        self.assertEqual({j for _,j,_ in calls},{5})
        self.assertTrue(all(r['passed'] for r in result['cycle_results']))

    def test_repetition_block_stops_at_first_failed_tracking_cycle(self):
        result,calls,code,bus=self.run_bench(7,tracking_scale=.6,
            gripper_options=['--gripper-amplitude','0.5','--gripper-peak-hold','8',
                '--gripper-return-hold','8','--repetitions','10'])
        self.assertEqual(code,1)
        self.assertEqual(len(result['cycle_results']),1)
        self.assertTrue(result['parameters_restored'])
        self.assertTrue(result['initial_enable_state_restored'])

    def test_loaded_speed_limit_comparison_keeps_excursion_current_and_restores(self):
        result,calls,code,bus=self.run_bench(3,gripper_options=['--loaded-speed-limit','0.5'])
        self.assertEqual(code,0)
        self.assertEqual(result['amplitude_motor_deg'],.25)
        self.assertEqual(result['current_limit_A'],12.)
        self.assertEqual(result['speed_limit_motor_deg_s'],.5)
        self.assertEqual(bus.params[0x7017],33.)
        self.assertTrue(result['parameters_restored'])

    def test_wrist_calibration_profiles_keep_current_speed_and_return_guards(self):
        for joint,direction in ((5,1),(6,-1)):
            result,calls,code,bus=self.run_bench(joint,gripper_options=['--wrist-amplitude','1',
                '--wrist-duration','20','--wrist-direction',str(direction),'--wrist-peak-hold','20',
                '--wrist-return-hold','8','--announce-peak'])
            self.assertEqual(code,0)
            self.assertEqual(result['direction'],direction)
            self.assertEqual(result['current_limit_A'],.6)
            self.assertEqual(result['speed_limit_motor_deg_s'],.25)
            self.assertEqual(result['peak_hold_s'],20.)
            self.assertEqual(result['tracking_criteria']['absolute_return_error_max_motor_deg'],.05)
            self.assertEqual({j for _,j,_ in calls},{joint})
            self.assertTrue(result['initial_enable_state_restored'])


if __name__=='__main__':unittest.main(verbosity=2)
