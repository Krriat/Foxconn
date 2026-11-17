"""Shared wire validation and joint target limits; no ROS or hardware imports."""
import json
import math
from stage_policy import stage_specs

PREFIX = "/robot/policy"
JOINT_NAMES = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(1, 8)]
# Both controllers returned these limits in the read-only readiness probe.
HARD_LOWER = [-3.10668606855, -2.09439510239, -3.10668606855, -1.0471975512,
              -3.10668606855, -1.0471975512, -1.57079632679]
HARD_UPPER = [3.10668606855, 2.09439510239, 3.10668606855, 2.53072741539,
              3.10668606855, 1.0471975512, 1.57079632679]
ACTIVE_PHASES = ("WAITING_ACTION", "STARTING_SERVO", "RUNNING", "PREVIEW")
# Upper bounds needed by recorded near-flap demonstrations (maxima 136.113
# and 140.057 degrees). Keep every other bound at the original 12-degree margin.
# These caps remain inside the native driver's independent 0.08-radian margin.
REVIEWED_UPPER_CAPS = {"left_joint_4": math.radians(136.5),
                       "right_joint_4": math.radians(140.25)}


def joint_bounds(margin, upper_overrides=None):
    lower = [HARD_LOWER[i % 7] + margin for i in range(14)]
    upper = [HARD_UPPER[i % 7] - margin for i in range(14)]
    if upper_overrides is None:
        return lower, upper
    if not isinstance(upper_overrides, dict):
        raise ValueError("joint upper overrides must be a named mapping")
    for name, value in upper_overrides.items():
        if name not in REVIEWED_UPPER_CAPS:
            raise ValueError("unreviewed joint upper override: " + str(name))
        index = JOINT_NAMES.index(name)
        cap = min(REVIEWED_UPPER_CAPS[name], HARD_UPPER[index % 7] - .08)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not lower[index] < value <= cap):
            raise ValueError("joint upper override exceeds reviewed cap: " + name)
        upper[index] = value
    return lower, upper


def validate_config(config):
    stage_specs(config)
    if config["model"] not in ("act", "interact"):
        raise ValueError("unknown policy")
    if config["action_hz"] != 20 or config["servo_hz"] != 100:
        raise ValueError("reviewed time bases are action=20 Hz, Servo J=100 Hz")
    if config.get("prediction_hz") != 20 or config.get("chunk_steps") != 50:
        raise ValueError("timed fusion requires full fifty-step predictions at 20 Hz")
    for name, lower, upper in (("temporal_ensemble_coefficient", 0., 1.),
                               ("trajectory_response_seconds", .05, .5),
                               ("execution_lead_seconds", 0., .05),
                               ("execution_speed_scale", 1., 1.5)):
        value = config.get(name, 1. if name == "execution_speed_scale" else None)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid {name}")
    limit = config.get("max_round_seconds")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or
                              not math.isfinite(limit) or limit <= 0):
        raise ValueError("max_round_seconds must be null (manual stop only) or a positive finite duration")
    bounds = {"command_timeout_seconds": .25,
              "heartbeat_timeout_seconds": .4, "feedback_timeout_seconds": .25,
              "input_max_age_seconds": .3, "input_max_skew_seconds": .1,
              "max_joint_speed_rad_s": .5, "max_joint_acceleration_rad_s2": 2.25, "max_relative_target_rad": .035,
              "max_tracking_error_rad": .15, "home_tolerance_rad": .01,
              "home_joint_speed_scale": .05, "home_speed_mm_s": 50}
    for name, upper in bounds.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"{name} exceeds reviewed bound {upper}")
    if not .08 <= config["joint_limit_margin_rad"] <= .3:
        raise ValueError("invalid joint limit margin")
    if not 0 <= config["suction_off_threshold"] < config["suction_on_threshold"] <= 1:
        raise ValueError("invalid suction hysteresis")
    if type(config["enable_motion"]) is not bool or type(config["suction_enabled"]) is not bool:
        raise ValueError("enable_motion and suction_enabled must be boolean")
    validate_joints(config["home_left"] + config["home_right"], config["joint_limit_margin_rad"],
                    config.get("joint_upper_limits_rad"))
    return config


def round_time_expired(elapsed_seconds, config):
    limit = config["max_round_seconds"]
    return limit is not None and elapsed_seconds >= limit


def finite_vector(values, count, name):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must contain {count} values")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           for x in values):
        raise ValueError(f"{name} contains nonfinite or nonnumeric values")
    return list(map(float, values))


def validate_joints(values, margin, upper_overrides=None):
    values = finite_vector(values, 14, "joints")
    lower, upper = joint_bounds(margin, upper_overrides)
    for i, value in enumerate(values):
        if not lower[i] <= value <= upper[i]:
            raise ValueError(f"{JOINT_NAMES[i]}={value:.6f} exceeds joint margin")
    return values


def validate_action(message, run_id, publisher_id, last_sequence, config):
    if not isinstance(message, dict) or message.get("schema") != "dual_arm_suction_action_v1":
        raise ValueError("unsupported action schema")
    if message.get("run_id") != run_id or not run_id:
        raise ValueError("action belongs to another round")
    identity = message.get("publisher_id")
    if not isinstance(identity, str) or not identity or len(identity) > 128:
        raise ValueError("invalid publisher identity")
    if publisher_id and identity != publisher_id:
        raise ValueError("publisher changed within round")
    sequence = message.get("sequence")
    if type(sequence) is not int or sequence <= last_sequence:
        raise ValueError("replayed or out-of-order action")
    if message.get("model") != config["model"]:
        raise ValueError("unexpected model")
    if message.get("checkpoint_sha256") != config["checkpoint_sha256"][config["model"]]:
        raise ValueError("unexpected model weights")
    if message.get("action_hz") != config["action_hz"]:
        raise ValueError("action time base differs from training")
    age = message.get("max_input_receive_age_ms")
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age):
        raise ValueError("missing input freshness")
    if not 0 <= age <= config["input_max_age_seconds"] * 1000:
        raise ValueError("policy input is stale")
    action = finite_vector(message.get("action"), 16, "action")
    validate_joints(action[:14], config["joint_limit_margin_rad"], config.get("joint_upper_limits_rad"))
    if any(not -0.25 <= x <= 1.25 for x in action[14:]):
        raise ValueError("suction prediction is outside tolerated normalized range")
    return action


def validate_chunk(message, run_id, publisher_id, last_sequence, config, receive_ns):
    if not isinstance(message, dict) or message.get("schema") != "dual_arm_suction_chunk_v2":
        raise ValueError("unsupported chunk schema")
    scale = message.get("execution_speed_scale", 1.)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale)
            or not 1. <= scale <= 1.5 or scale != config.get("execution_speed_scale", 1.)):
        raise ValueError("chunk execution speed scale differs from the current round")
    proxy = dict(message, schema="dual_arm_suction_action_v1", action=message.get("actions", [None])[0])
    validate_action(proxy, run_id, publisher_id, last_sequence, config)
    actions = message.get("actions")
    if not isinstance(actions, list) or len(actions) != 50 or message.get("step_ns") != 50_000_000:
        raise ValueError("expected full fifty-step chunk with training time base")
    for action in actions:
        vector = finite_vector(action, 16, "chunk action")
        validate_joints(vector[:14], config["joint_limit_margin_rad"], config.get("joint_upper_limits_rad"))
        if any(not -.25 <= x <= 1.25 for x in vector[14:]):
            raise ValueError("invalid suction prediction")
    names = ("observation_time_ns", "inference_start_ns", "inference_end_ns", "send_time_ns")
    stamps = [message.get(name) for name in names]
    if any(type(stamp) is not int or stamp <= 0 for stamp in stamps):
        raise ValueError("missing timing metadata")
    observation, start, end, send = stamps
    # Hosts share Jetson NTP. Allow small residual LAN clock asymmetry.
    tolerance = 10_000_000
    if observation > start + tolerance or not start <= end <= send or send > receive_ns + tolerance:
        raise ValueError("timestamps are inconsistent; check host clock synchronization")
    if receive_ns - observation > int(config["input_max_age_seconds"] * 1e9):
        raise ValueError("observation is stale at Jetson reception")
    if receive_ns - send > int(config["command_timeout_seconds"] * 1e9):
        raise ValueError("chunk was delayed in transit")
    return actions


def bounded_target(action, measured, previous, dt, config):
    validate_joints(measured, config["joint_limit_margin_rad"], config.get("joint_upper_limits_rad"))
    validate_joints(previous, config["joint_limit_margin_rad"], config.get("joint_upper_limits_rad"))
    if max(abs(a - b) for a, b in zip(previous, measured)) > config["max_tracking_error_rad"]:
        raise ValueError("measured joints do not track last Servo J command")
    # Never catch up with a large step after a delayed timer callback.
    dt = min(max(dt, 0.0), 1.0 / config["servo_hz"])
    step = config["max_joint_speed_rad_s"] * dt
    relative = config["max_relative_target_rad"]
    target, limited = [], False
    for wanted, actual, last in zip(action[:14], measured, previous):
        near = min(actual + relative, max(actual - relative, wanted))
        value = min(last + step, max(last - step, near))
        target.append(value)
        limited |= abs(value - wanted) > 1e-9
    return target, limited


def suction_target(value, previous, config):
    if value >= config["suction_on_threshold"]:
        return True
    if value <= config["suction_off_threshold"]:
        return False
    return previous


def encode(message):
    return json.dumps(message, separators=(",", ":"), allow_nan=False)
