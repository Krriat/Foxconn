"""Offline schema/trajectory helpers. Importing this module never loads the SDK."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SDK = ROOT.parent.parent
COMMAND_FIELDS = 'trial_id,command_seq,arm,joint_name,host_tx_start_ns,host_tx_done_ns,device_command_applied_ns,q_cmd_rad,qd_cmd_rad_s,tau_ff_cmd_nm,mode,trajectory_t_s,qd_reference_rad_s,segment'.split(',')
SAMPLE_FIELDS = 'trial_id,sample_seq,arm,joint_name,device_sample_ns,host_rx_ns,q_rad,qd_rad_s,torque_nm,torque_kind,current_a,current_kind,motor_temperature_c,drive_temperature_c,bus_voltage_v,limit_flags,fault_flags,mode,identity_kind,duplicate,identical_payload,valid,invalid_reason,segment'.split(',')

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def read_json(path):
    return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    if not isinstance(c, dict) or c.get('schema_version') != 1:
        raise ValueError('Expected measurement config schema_version: 1')
    # Reject NaN even inside optional site fields.
    json.dumps(c, allow_nan=False)
    return c

def new_run(parent):
    path = Path(parent).resolve() / (time.strftime('run_%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    (path / 'raw_vendor').mkdir()
    return path

def envelope():
    return dict(status='unavailable', source=None, joint_output_units='Nm, rad/s',
                continuous_curves=None, peak_curves=None, peak_duration_s=None,
                temperature_and_voltage_conditions=None,
                reason='本地SDK及需求摘录未提供逐轴输出端连续/峰值转速曲线；未进行厂商许可台架试验。')

def metadata(config, backend):
    return dict(schema_version=1, backend=backend, synthetic=backend == 'mock',
                complete=False, invalid_reasons=[], config=config,
                host=dict(platform=platform.platform(), python=platform.python_version()),
                script_sha256={p.name: digest(p) for p in ROOT.iterdir() if p.suffix in ('.py','.cpp')},
                clocks=dict(host='CLOCK_MONOTONIC / steady_clock, integer ns',
                            device='unavailable', synchronization=None, synchronization_error_ns=None),
                units=dict(q='rad', qd='rad/s', torque='Nm; see torque_kind'),
                sample_sequence_source='host_transport_counter; not a device sequence',
                device_sample_identity_available=False,
                feedforward='position-only candidate; qd_cmd and tau_ff are empty because not sent',
                torque_conversion=None,
                raw_log_sha256={})

def finish_metadata(run, meta):
    meta['raw_log_sha256'] = {str(p.relative_to(run)): digest(p)
                              for p in sorted(Path(run).rglob('*')) if p.is_file()
                              and (p.name in ('samples.csv', 'commands.csv', 'events.jsonl')
                                   or 'raw_vendor' in p.parts)}
    write_json(Path(run) / 'metadata.json', meta)

def finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)

# C3 envelope: its first three derivatives vanish at both boundaries.
S = np.polynomial.Polynomial([0, 0, 0, 0, 35, -84, 70, -20])

def polynomial_bound(p):
    roots = p.deriv().roots()
    x = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]
    return max(abs(p(np.array(x)))) * (1 + 1e-12)

def derivative_bounds(frequency, ramp_s):
    w = 2 * math.pi * frequency
    e = [1.] + [polynomial_bound(S.deriv(i)) / ramp_s ** i for i in (1, 2, 3)]
    return [1., w + e[1], w*w + 2*e[1]*w + e[2],
            w**3 + 3*e[1]*w*w + 3*e[2]*w + e[3]]

def trajectory(trial, t):
    """Return analytic delta q, velocity, acceleration, jerk and phase label."""
    t = np.asarray(t, dtype=float)
    x = t - trial['stationary_before_s']
    ramp = trial['ramp_s']
    steady = trial['stable_cycles'] / trial['frequency_hz']
    stop = 2*ramp + steady
    active = (x >= 0) & (x <= stop)
    entering = active & (x < ramp)
    leaving = active & (x > ramp + steady)
    u = np.clip(np.where(leaving, (stop-x)/ramp, x/ramp), 0, 1)
    e = [np.where(active, S(u), 0.)]
    for i in (1, 2, 3):
        e.append(np.where(entering | leaving, S.deriv(i)(u) / ramp**i * np.where(leaving, (-1)**i, 1), 0.))
    w = 2*math.pi*trial['frequency_hz']
    sn, cs = np.sin(w*x), np.cos(w*x)
    a = trial['amplitude_rad']
    q = a*e[0]*sn
    v = a*(e[1]*sn + e[0]*w*cs)
    acc = a*(e[2]*sn + 2*e[1]*w*cs - e[0]*w*w*sn)
    jerk = a*(e[3]*sn + 3*e[2]*w*cs - 3*e[1]*w*w*sn - e[0]*w**3*cs)
    segment = np.where(~active, 'stationary', np.where(entering, 'enter', np.where(leaving, 'exit', 'steady')))
    return q, v, acc, jerk, segment

HARDWARE_BLOCKERS = [
    '运动后端尚未开放：缺少经验证的独立保护/SDK调用超时和控制器watchdog停止链路',
    '公开接口无原生设备采样序号/时间戳；只能区分SDK新收包，不能证明设备采样去重',
    '未取得实际控制权、保护设置、温度/电流限额及停止轨迹的完整核验方案',
]

def site_blockers(c, arm):
    blocked = list(HARDWARE_BLOCKERS)
    for key in ('deployment_interface_confirmed',):
        if c['interface'].get(key) is not True:
            blocked.append('interface.' + key)
    for key in ('lookahead_s', 'gain'):
        if c['interface'].get(key) is None:
            blocked.append('interface.' + key)
    for key in ('j4_sign_verified', 'j6_j7_coupling_verified'):
        if c['mapping'].get(key) is not True:
            blocked.append('mapping.' + key)
    if not c['mapping'].get(arm): blocked.append('mapping.' + arm)
    if c['center_q_rad'].get(arm) is None: blocked.append('center_q_rad.' + arm)
    for key, value in c['limits'].items():
        if value is None: blocked.append('limits.' + key)
    for key, value in c['verification'].items():
        if not value: blocked.append('verification.' + key)
    for key in ('posture_id', 'installed_total_mass_kg', 'includes_connectors_and_sensors',
                'controller_tool_mass_kg', 'controller_tool_cog_m', 'controller_tool_inertia_kg_m2'):
        if c['condition'].get(key) is None: blocked.append('condition.' + key)
    return blocked

def capability_report():
    def entry(value, source, note=None):
        return dict(value=value, source=source, note=note)
    h = 'include/rokae/'
    return dict(schema_version=1, connected=False,
        vendor=entry('ROKAE', 'README.md'),
        model=entry('AR5 intended; runtime type unverified', 'test.md; robot.h: ArRobot'),
        sdk_version=entry('0.7.1.ar_6 directory / project 0.7.1', 'CMakeLists.txt; BaseRobot::sdkVersion'),
        controller_version=entry(None, h+'robot.h: Info.version / robotInfo'),
        firmware_version=entry(None, h+'robot.h: Info.version', 'No separate firmware getter identified'),
        connection=entry(None, h+'robot.h: ArRobot::connectToRobot; cxl/pos.cpp', 'Explicit --connect only; no port guessed'),
        streaming=entry(['JointPosition', 'CartesianPosition', 'Torque', 'joint/cartesian impedance'],
                        h+'motion_control_rt.h: sendCommand, startMove', 'Advertised API; AR5/firmware support needs verification'),
        velocity_and_position_feedforward=entry(None, h+'data_types.h: JointPosition', 'No verified joint velocity or position+velocity+effort command; never emulate using torque mode'),
        servo_period=entry(0.02, 'example/rt/servoj_demo.cpp: planPeriod', '50Hz ServoJ example; generic sendCommand documentation requires 1ms. Verify actual ServoJ firmware semantics'),
        interpolation=entry(None, h+'motion_control_rt.h: setServoJoint(T, Lookahead, Kp)', 'Lookahead and gain configurable; active values/getter not identified'),
        command_completion=entry(None, h+'motion_control_rt.h: sendCommand returns void', 'Return is not proof of execution; queue semantics and applied timestamp unavailable'),
        feedback_intervals_ms=entry([1, 2, 4, 8, 1000], h+'robot.h: startReceiveRobotState', 'Requested packet intervals, not measured native sensor sampling rates'),
        device_timestamp_and_sequence=entry(None, h+'data_types.h: RtSupportedFields', 'Not exposed. updateRobotState > 0 identifies received data only; identical values do not prove duplicate samples'),
        position=entry('q_m [rad]', h+'data_types.h: RtSupportedFields::jointPos_m'),
        velocity=entry('dq_m [rad/s]', h+'data_types.h: RtSupportedFields::jointVel_m'),
        torque=entry('tau_m [Nm], provenance unverified', h+'data_types.h: RtSupportedFields::tau_m', 'jointTorque() separately documents joint force sensor; do not assume streaming tau_m is identical without verification'),
        motor_torque=entry('motor_tau; units and output conversion unknown', h+'data_types.h: RtSupportedFields::motorTau', 'Not collected or converted'),
        current=entry(None, h+'data_types.h: RtSupportedFields', 'Iq/phase/RMS/bus current not exposed'),
        temperature_voltage_derating=entry(None, h+'data_types.h: RtSupportedFields', 'No motor/drive temperature, bus voltage or full derating flag fields found'),
        fault_mode=entry('operationState / operateMode / powerState; updateRobotState exceptions', h+'robot.h', 'Not a complete protection/ownership snapshot; no motion-control-mode getter identified'),
        limits=entry('getSoftLimit: enabled, per-axis lower/upper [rad]', h+'robot.h: Robot_T::getSoftLimit', 'J4 mapping, J6/J7 coupling, dynamic and workspace limits require site evidence'),
        tool=entry('toolset().load: mass, cog, inertia', h+'robot.h: toolset; data_types.h: Load'),
        ownership=entry(None, h+'robot.h: connectToRobot / disconnectFromRobot', 'No verified acquire/release token; disconnect explicitly stops motion, so inspection is only for an idle robot with other clients exited'),
        controlled_stop=entry('RtMotionControl::stopMove; then read operationState and velocity', h+'motion_control_rt.h: stopMove', 'Stop trajectory, duration and failure behavior must be verified; this release contains no hardware motion backend'),
        watchdog=entry(None, h+'robot.h: setRtNetworkTolerance', 'Packet-loss tolerance is not a validated watchdog timeout; do not change it'),
        sdk_timeout=entry('startReceiveRobotState first packet 3s; updateRobotState(timeout)', h+'robot.h; 1.pdf §3.3.2', 'No bounded send/stop/connect timeout guarantee identified; parent timeout cannot establish physical stop'),
        hardware_execution=dict(supported=False, reasons=HARDWARE_BLOCKERS),
        available_torque_envelope=envelope(),
        source_sha256={str(p.relative_to(SDK)): digest(p) for p in
                       [SDK/'include/rokae/robot.h', SDK/'include/rokae/data_types.h', SDK/'include/rokae/motion_control_rt.h', SDK/'example/rt/servoj_demo.cpp']})
