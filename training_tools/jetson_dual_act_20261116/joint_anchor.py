"""Remove the model's reconstruction offset for same-sample joint actions."""
import numpy as np


def anchor_joint_chunk(actions, observation_state, max_bias_rad=.15):
    """Preserve predicted motion increments and suction; anchor t=0 to observed q.

    This applies only to this dataset's absolute joint actions, where action[0]
    is the measured state at the observation timestamp. The anchor must be that
    observation's state, never a newer feedback sample or the last sent target.
    """
    raw = np.asarray(actions)
    state = np.asarray(observation_state)
    if raw.shape != (50, 16) or state.shape != (16,):
        raise ValueError('Expected a 50x16 chunk and its 16D observation state')
    if not np.isfinite(raw).all() or not np.isfinite(state).all():
        raise ValueError('Nonfinite policy output or observation')
    if not 0 < max_bias_rad <= .15:
        raise ValueError('Invalid reconstruction-bias limit')
    bias = raw[0, :14] - state[:14]
    if np.max(abs(bias)) > max_bias_rad:
        raise ValueError(f'Policy first-step reconstruction error exceeds {max_bias_rad} rad')
    result = raw.copy()
    result[:, :14] = state[None, :14] + (raw[:, :14] - raw[0:1, :14])
    return result
