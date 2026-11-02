"""Prevent a partial control experiment from releasing calibration motion."""


def require_experiment_release(profile, action):
    if action not in ('calibration_motion', 'gain_adoption'):
        raise ValueError('Unknown experiment release action')
    if profile.get(action + '_allowed') is not True:
        findings = '; '.join(profile.get('unresolved_findings', []))
        raise RuntimeError(action + ' is blocked: ' + (findings or 'no current qualification release'))

