#!/usr/bin/env python3
r"""键盘双臂拖动示教、USR-M050 双吸盘控制与训练数据采集。

先启动双臂示教服务和头部/左腕/右腕三相机，再在一个 ROS 终端运行：
  python3 record_xcore_teach_suction.py --fps 20 --task "双臂吸盘抓取"
或在普通终端一键启动以上服务、三相机和本采集器：
  bash start_xcore_teach_suction.sh --fps 20 --task "双臂吸盘抓取"

不需要头显、手柄、遥操桥接、头部电机或腰部电机节点。
终端获得焦点后直接按键，无需回车；管道输入或 --line-input 使用单字母加回车：
  A 开始双臂示教；Z 请求结束双臂示教。
  B 回到本次实测保存的双臂位置；回位继续录在同一条数据里，最后 S 保存。
  R 开始一条录制；S 停止并保存；D 停止并丢弃当前未保存记录。
  1 左吸盘 DO1 开；2 左吸盘 DO1 关；3 右吸盘 DO2 开；4 右吸盘 DO2 关。
  H 帮助；Q/Ctrl+C 退出并请求结束本程序启动的示教。
每条机械臂仍需按住自己的末端实体按钮才能拖动。示教时限沿用驱动默认值。
R 后持续保持录制，直到 S/D/Q；缺图、跳帧、IO 或示教暂不可用时等待，恢复后自动继续。
缺失/不同步样本不补造，恢复时另起连续片段；示教结束后可按 A 再开启，无需重复 R。
终端每 5 秒显示帧数与录制状态，保存时记录总时长、间断及有效帧数。
相机频率用于诊断告警；短时低频不会阻断完整、同步且新鲜的样本。
按头部相机曝光顺序匹配腕相机和关节，保留真实时间戳、真实缺帧及连续片段。
默认采样 20 Hz；相机保持原 30 Hz 输出。20 Hz 下源时间戳断点阈值为 80 ms，
正常降采样产生的 33.3/66.7 ms 图像间隔不会因此切段，100 ms 间断仍会切段。
各采样频率共用无损 RGB PNG 快速写入；系统有 libdeflate 时减小文件体积，
缺少时沿用快速 PNG 后备路径，无需额外安装依赖。episode.json 保存写入诊断。
图像队列最多缓冲 64 帧，短暂拥堵时保留已对齐样本重试，不将排队本身标记为缺帧。
真实源时戳跳帧、示教/回位切换仍划分片段；S 只保存已排队样本并记录未入队计数。
吸盘使用独立开/关键，避免依赖终端无法可靠提供的松键事件；长按不反复切换。

新数据是 16 维：左 7 关节 + 右 7 关节 + 左吸盘 + 右吸盘。
state 的吸盘值是 M050 逻辑 DO 读回；action 的吸盘值是键盘目标，开=1、关=0。
IO 与按键历史按图像时间选取过去样本，并保存各自时间戳。不是吸附成功检测。
数据不含头腰角度，使用新 schema/目录，不与原头腰 16 维数据混写。
  python3 record_xcore_teach_suction.py --root /path/to/data --validate-only
  python3 record_xcore_teach_suction.py --root /path/to/data \
      --export-to /path/to/new_lerobot --repo-id local/dual_arm_suction
导出需要原 LeRobot 0.4.3 环境；原始采集/校验不需要 LeRobot。

--controls-only 只做键盘示教/吸盘控制；--check 只读检查，不发启停或 DO 写入。
--self-test 不连接任何硬件。退出时保留未确认数据为 .partial。
启动读取吸盘当前位置并采用其现有目标，不切换输出；退出保持最后吸盘输出。
示教物理按钮规则、已有运动参数和其他配置保持原样。
B 的固定目标在同目录 teach_suction_return_pose.json；启动不运动，也不覆盖此目标。
B 等待示教退出，再通过现有初始化/MoveAbsJ 服务以 5% 关节速度回位，保持吸盘输出。
回位使用专用高频实测关节反馈，关节 action 沿用逐帧实测位置；Z 可停止回位。
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import ctypes as C
import ctypes.util
from dataclasses import dataclass
import fcntl
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import select
import signal
import struct
import sys
import tempfile
import termios
import threading
import time
import tty
import zlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xcore_teach_recording.core import Inputs as OriginalInputs, NotReady, AwaitFrame
from xcore_teach_recording.dataset import EpisodeWriter as OriginalWriter, prepare_root
from xcore_teach_recording.quality import episode_quality
from xcore_teach_recording.feedback_trace import FeedbackTrace
from xcore_teach_recording.local_feedback import LocalFeedbackReceiver, DEFAULT_PORT as JOINT_FEEDBACK_PORT

PREFIX = "/robot_02/upperlimb"
SIDES = ("left", "right")
CAMERAS = ("head", "left_wrist", "right_wrist")
STREAMS = SIDES + CAMERAS
STATE_NAMES = tuple(f"{side}_joint_{i}.pos" for side in SIDES for i in range(1, 8)) + (
    "left_suction.on", "right_suction.on")
SCHEMA_ID = "keyboard_dual_arm_suction_v1"
DEFAULT_ROOT = ROOT / "assets/dataset/act_teach_keyboard_suction_30hz"
RETURN_POSE_PATH = Path(__file__).with_name("teach_suction_return_pose.json")
HELP = ("A=开始示教 Z=结束示教/停止回位 B=双臂回到已保存位置\n"
        "R=开始录制 S=停止保存 D=丢弃未保存记录\n"
        "1=左吸盘开 2=左吸盘关 3=右吸盘开 4=右吸盘关 H=帮助 Q=退出\n"
        "R 后持续录制；临时缺数据会等待并自动恢复，S 停止保存")


def notify(message):
    print(message, flush=True)


def load_png_compressor():
    """Optional system libdeflate; portable fallback needs only Python zlib."""
    try:
        name = ctypes.util.find_library("deflate")
        if not name:
            return None
        lib = C.CDLL(name)
        for name, arguments, result in (
            ("libdeflate_alloc_compressor", [C.c_int], C.c_void_p),
            ("libdeflate_zlib_compress_bound", [C.c_void_p, C.c_size_t], C.c_size_t),
            ("libdeflate_zlib_compress", [C.c_void_p, C.c_void_p, C.c_size_t,
                                         C.c_void_p, C.c_size_t], C.c_size_t),
            ("libdeflate_free_compressor", [C.c_void_p], None),
        ):
            function = getattr(lib, name)
            function.argtypes, function.restype = arguments, result
        return lib
    except (OSError, AttributeError):
        return None


PNG_COMPRESSOR = load_png_compressor()


class PngEncoder:
    """One lossless RGB8 encoder per worker; native handles are never shared.

    PNG Sub prediction plus fast native deflate reduces the measured camera
    images to about one third of stored PNG size. Reuse input/output buffers.
    The fallback retains the previous fast, uncompressed PNG implementation.
    """
    def __init__(self):
        self.lib = PNG_COMPRESSOR
        self.handle = self.lib.libdeflate_alloc_compressor(1) if self.lib else None
        self.scanlines = self.output = None
        self.backend = "libdeflate_sub_1" if self.handle else "zlib_store"

    @staticmethod
    def chunk(kind, data):
        checksum = zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    def write(self, path, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("PNG input must be RGB uint8")
        height, width, _ = rgb.shape
        shape = (height, width*3 + 1)
        if self.scanlines is None or self.scanlines.shape != shape:
            self.scanlines = np.empty(shape, dtype=np.uint8)
            if self.handle:
                bound = self.lib.libdeflate_zlib_compress_bound(self.handle, self.scanlines.nbytes)
                self.output = C.create_string_buffer(bound)
        scanlines = self.scanlines
        if self.handle:
            scanlines[:, 0] = 1  # PNG Sub filter, three bytes per RGB pixel.
            pixels = scanlines[:, 1:].reshape(height, width, 3)
            pixels[:, 0] = rgb[:, 0]
            np.subtract(rgb[:, 1:], rgb[:, :-1], out=pixels[:, 1:])
            size = self.lib.libdeflate_zlib_compress(
                self.handle, scanlines.ctypes.data, scanlines.nbytes,
                self.output, len(self.output))
            if not size:
                raise RuntimeError("native PNG compression failed")
            compressed = C.string_at(self.output, size)
        else:
            scanlines[:, 0] = 0
            scanlines[:, 1:] = rgb.reshape(height, width*3)
            compressed = zlib.compress(scanlines, level=0)
        header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        data = (b"\x89PNG\r\n\x1a\n" + self.chunk(b"IHDR", header) +
                self.chunk(b"IDAT", compressed) + self.chunk(b"IEND", b""))
        return Path(path).write_bytes(data)

    def close(self):
        if self.handle:
            self.lib.libdeflate_free_compressor(self.handle)
            self.handle = None


def write_rgb_png(path, rgb):
    encoder = PngEncoder()
    try:
        return encoder.write(path, rgb)
    finally:
        encoder.close()


class M050Config(C.Structure):
    _fields_ = [("host", C.c_char_p), ("port", C.c_uint16), ("unit_id", C.c_uint8),
                ("timeout_ms", C.c_int), ("source_ip", C.c_char_p)]


class M050Channels(C.Structure):
    _fields_ = [("ch1", C.c_uint8), ("ch2", C.c_uint8)]


class M050Error(C.Structure):
    _fields_ = [("code", C.c_int), ("system_errno", C.c_int),
                ("modbus_exception", C.c_uint8), ("write_may_have_applied", C.c_int),
                ("message", C.c_char * 192)]


class M050Device:
    """Use one existing SDK handle, with all IO serialized in one worker."""
    def __init__(self, args):
        self.lib = C.CDLL(str(args.m050_library))
        self.handle = C.c_void_p()
        declarations = {
            "m050_config_default": ([C.POINTER(M050Config)], None),
            "m050_create": ([C.POINTER(M050Config), C.POINTER(C.c_void_p), C.POINTER(M050Error)], C.c_int),
            "m050_destroy": ([C.c_void_p], None),
            "m050_read_do": ([C.c_void_p, C.POINTER(M050Channels), C.POINTER(M050Error)], C.c_int),
            "m050_set_do": ([C.c_void_p, C.c_uint, C.c_int, C.POINTER(M050Channels),
                            C.POINTER(M050Error)], C.c_int),
        }
        for name, (argtypes, restype) in declarations.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = argtypes, restype
        config = M050Config()
        self.lib.m050_config_default(C.byref(config))
        config.host = args.m050_host.encode("ascii")
        config.port, config.unit_id, config.timeout_ms = args.m050_port, args.m050_unit, args.m050_timeout_ms
        config.source_ip = args.m050_source_ip.encode("ascii") if args.m050_source_ip else None
        error = M050Error()
        self.check(self.lib.m050_create(C.byref(config), C.byref(self.handle), C.byref(error)), error)

    @staticmethod
    def check(result, error):
        if result:
            uncertain = "；写入可能已经执行，先读回确认" if error.write_may_have_applied else ""
            raise RuntimeError(f"M050 {result}: {bytes(error.message).decode(errors='replace')}{uncertain}")

    def read(self):
        state, error = M050Channels(), M050Error()
        self.check(self.lib.m050_read_do(self.handle, C.byref(state), C.byref(error)), error)
        return bool(state.ch1), bool(state.ch2)

    def write(self, side, enabled):
        state, error = M050Channels(), M050Error()
        self.check(self.lib.m050_set_do(self.handle, SIDES.index(side) + 1, int(enabled),
                                        C.byref(state), C.byref(error)), error)
        return bool(state.ch1), bool(state.ch2)

    def close(self):
        if self.handle:
            self.lib.m050_destroy(self.handle)
            self.handle = C.c_void_p()


@dataclass(frozen=True)
class SuctionSample:
    actual: tuple
    target: tuple
    feedback_at: float
    command_at: tuple


class SuctionOutputs:
    """Level commands, no startup/shutdown writes and no automatic write retries."""
    def __init__(self, device, poll_seconds=.1, report=notify, read_only=False):
        self.device, self.poll_seconds, self.report = device, poll_seconds, report
        self.read_only = read_only
        self.condition = threading.Condition()
        self.desired = {side: None for side in SIDES}
        self.command_at = {side: None for side in SIDES}
        self.sequence = {side: 0 for side in SIDES}
        self.attempted = dict(self.sequence)
        self.order = dict(self.sequence)
        self.counter = 0
        self.readings, self.commands = deque(maxlen=600), deque(maxlen=600)
        self.error = None
        self.closing = False
        self.started = False
        self.last_report = None
        self.thread = threading.Thread(target=self.run, name="keyboard-m050-io", daemon=True)

    def start(self):
        self.thread.start()
        self.started = True

    def _remember_command(self):
        if all(self.desired[side] is not None for side in SIDES):
            self.commands.append((max(self.command_at.values()),
                                  tuple(self.desired[side] for side in SIDES),
                                  tuple(self.command_at[side] for side in SIDES)))

    def set(self, side, enabled, at=None):
        if side not in SIDES:
            raise ValueError("unknown suction side")
        if self.read_only:
            raise RuntimeError("只读检查不能切换吸盘")
        with self.condition:
            if self.closing:
                return False
            enabled = bool(enabled)
            if self.desired[side] == enabled:
                return False
            self.desired[side] = enabled
            self.command_at[side] = time.monotonic() if at is None else at
            self.counter += 1
            self.order[side] = self.counter
            self.sequence[side] += 1
            self._remember_command()
            self.condition.notify_all()
            return True

    def record_readback(self, values, at):
        with self.condition:
            self.readings.append((at, tuple(bool(value) for value in values)))
            adopted = False
            for index, side in enumerate(SIDES):
                if self.desired[side] is None:
                    self.desired[side] = bool(values[index])
                    self.command_at[side] = at
                    adopted = True  # Adopt existing DO without writing it.
            if adopted:
                self._remember_command()
            mismatch = [side for i, side in enumerate(SIDES)
                        if bool(values[i]) != self.desired[side] and
                        self.attempted[side] == self.sequence[side]]
            self.error = ("吸盘目标尚未由读回确认：" + ",".join(mismatch)) if mismatch else None

    def check_ready(self, now, max_age):
        with self.condition:
            if self.error:
                raise NotReady(self.error)
            if not self.readings or now - self.readings[-1][0] > max_age:
                raise NotReady("等待新鲜的吸盘读回")

    def sample_at(self, anchor, max_age):
        with self.condition:
            if self.error:
                raise NotReady(self.error)
            readings = [row for row in self.readings if row[0] <= anchor]
            commands = [row for row in self.commands if row[0] <= anchor]
            if not readings or not commands:
                raise NotReady("等待图像时刻之前的有效吸盘读回和目标")
            observed_at, actual = readings[-1]
            if anchor - observed_at > max_age:
                raise NotReady("吸盘读回过期")
            _, target, command_at = commands[-1]
            return SuctionSample(actual, target, observed_at, command_at)

    def _report(self, message):
        if message != self.last_report:
            self.report(message)
            self.last_report = message

    def run(self):
        next_poll = 0.
        while True:
            with self.condition:
                pending = [side for side in SIDES if self.sequence[side] != self.attempted[side]]
                side = min(pending, key=self.order.get) if pending else None
                if self.closing and side is None:
                    return
                if side is None and time.monotonic() < next_poll:
                    self.condition.wait(max(0., next_poll - time.monotonic()))
                    continue
                if side is not None:
                    self.attempted[side] = self.sequence[side]
                    target = self.desired[side]
            try:
                values = self.device.write(side, target) if side is not None else self.device.read()
                self.record_readback(values, time.monotonic())
                with self.condition:
                    error = self.error
                self._report(error or f"[吸盘读回] 左DO1={int(values[0])} 右DO2={int(values[1])}")
            except Exception as exc:
                with self.condition:
                    self.error = str(exc)
                self._report(f"[吸盘] {exc}；状态不明时暂停数采")
            next_poll = time.monotonic() + self.poll_seconds

    def close(self, timeout=10.):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        if self.started:
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise RuntimeError("M050 IO 线程尚未退出，未销毁正在使用的 SDK 句柄")
        self.device.close()


class DualTeaching:
    """Drive only existing physical-button drag services; start futures end at cleanup."""
    def __init__(self, starts, stops, request_factory, report=notify):
        self.starts, self.stops, self.request_factory, self.report = starts, stops, request_factory, report
        self.status = {side: None for side in SIDES}
        self.phase = "idle"
        self.owned = False
        self.blocked = False
        self.start_futures, self.stop_futures = {}, {}
        self.handled = set()
        self.started_at = 0.
        self.last_stop = {side: -math.inf for side in SIDES}

    def on_status(self, side, value):
        self.status[side] = value
        if value.startswith("error:"):
            self.blocked = True
        self.report(f"[示教] {side}: {value}")

    def start(self, now):
        if self.blocked:
            self.report("示教清理未确认，禁止再次开始；请检查驱动状态")
            return False
        if self.phase != "idle" or any(self.status[s] in ("starting", "teaching") for s in SIDES):
            self.report("示教已经开始或仍在清理，未重复发送开始请求")
            return False
        if not all(self.status[s] in ("idle", "stopped") for s in SIDES):
            self.report("等待左右臂示教状态就绪")
            return False
        if not all(client.service_is_ready() for client in (*self.starts.values(), *self.stops.values())):
            self.report("左右臂示教启停服务未齐备，请启用 start_cartesian_drag_safe_service:=true")
            return False
        self.phase, self.owned, self.started_at = "starting", True, now
        self.start_futures, self.stop_futures, self.handled = {}, {}, set()
        try:
            for side in SIDES:
                self.start_futures[side] = self.starts[side].call_async(self.request_factory())
        except Exception as exc:
            self.blocked = True
            self.stop(f"开始请求异常：{exc}")
        return True

    def stop(self, reason="键盘请求结束示教"):
        if not self.owned and not any(self.status[s] in ("starting", "teaching") for s in SIDES):
            self.report("当前没有运行中的示教")
            return
        if self.phase != "stopping":
            self.report(reason + "；等待双臂 disableDrag 清理确认")
        self.phase, self.owned = "stopping", True

    def tick(self, now):
        ended = False
        for side, future in self.start_futures.items():
            if future.done() and side not in self.handled:
                self.handled.add(side)
                ended = True
                try:
                    result = future.result()
                    message = str(result.message)
                    if message.startswith("cleanup unverified:"):
                        self.blocked = True
                    self.report(f"[示教结束] {side}: {message}")
                except Exception as exc:
                    self.blocked = True
                    self.report(f"[示教] {side} 开始请求结果不明：{exc}")
        if self.phase in ("starting", "teaching") and ended:
            self.stop("一侧示教结束或失败，同时请求另一侧结束")
        if self.phase == "starting":
            if all(self.status[s] == "teaching" for s in SIDES):
                self.phase = "teaching"
                self.report("双臂示教就绪；按住各自末端实体按钮拖动，R 开始录制")
            elif now - self.started_at > 10.:
                self.stop("等待双臂进入示教超时")
        elif self.phase == "teaching" and not all(self.status[s] == "teaching" for s in SIDES):
            self.stop("双臂示教状态不再一致")
        if self.phase != "stopping":
            return
        for side in SIDES:
            pending_start = side in self.start_futures and not self.start_futures[side].done()
            needs_stop = pending_start or self.status[side] in ("starting", "teaching")
            previous = self.stop_futures.get(side)
            if (needs_stop and (previous is None or previous.done()) and
                    now - self.last_stop[side] >= .25 and self.stops[side].service_is_ready()):
                try:
                    self.stop_futures[side] = self.stops[side].call_async(self.request_factory())
                    self.last_stop[side] = now
                except Exception as exc:
                    self.blocked = True
                    self.report(f"[示教停止] {side}: {exc}")
        clean = (all(f.done() for f in self.start_futures.values()) and
                 all(f.done() for f in self.stop_futures.values()) and
                 all(self.status[s] in ("idle", "stopped") for s in SIDES))
        if clean:
            self.phase = "blocked" if self.blocked else "idle"
            self.owned = False
            self.report("双臂示教已结束" if not self.blocked else "示教响应存在不确定性，禁止重新开始")


def load_return_pose(path=RETURN_POSE_PATH):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("units") != "rad":
        raise ValueError("回位点必须使用 rad 单位")
    result = {}
    for side in SIDES:
        joints = np.asarray(value.get(side), dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all() or np.any(np.abs(joints) > 2 * math.pi):
            raise ValueError(f"{side} 回位点必须是 7 个有效关节角")
        result[side] = joints.tolist()
    return result


class DualArmReturn:
    """B owns stop-teach -> initialize -> MoveAbsJ; recording remains armed."""
    def __init__(self, session, initializer, moves, stops, trigger_factory, move_factory,
                 inputs=None, targets=None, report=notify):
        self.session, self.initializer, self.moves, self.stops = session, initializer, moves, stops
        self.trigger_factory, self.move_factory = trigger_factory, move_factory
        self.inputs, self.targets, self.report = inputs, targets, report
        self.phase, self.blocked = "idle", False
        self.initialize_future = None
        self.move_futures, self.stop_futures = {}, {}
        self.deadline = 0.
        self.final_stop_sent = False
        self.last_stop = dict.fromkeys(SIDES, -math.inf)

    @property
    def active(self):
        return self.phase != "idle"

    def start(self, now):
        if self.blocked or self.session.blocked:
            self.report("回位/示教状态未确认，请先检查驱动状态")
            return False
        if self.active:
            self.report("正在回位或停止回位，未重复发送 B")
            return False
        if self.targets is None:
            self.report("没有有效回位点，未发运动命令")
            return False
        clients = (self.initializer, *self.moves.values(), *self.stops.values(),
                   *self.session.stops.values())
        if not all(client.service_is_ready() for client in clients):
            self.report("回位需要双臂示教停止、初始化、MoveAbsJ 和运动停止服务；请重启完整双臂驱动")
            return False
        if not all(self.session.status[side] in ("idle", "stopped", "starting", "teaching") for side in SIDES):
            self.report("等待双臂示教状态就绪，未发回位命令")
            return False
        self.initialize_future = None
        self.move_futures, self.stop_futures = {}, {}
        self.final_stop_sent = False
        self.last_stop = dict.fromkeys(SIDES, -math.inf)
        self.phase, self.deadline = "waiting_teach", now + 15.
        if self.inputs is not None:
            self.inputs.set_return_recording(True)
        if self.session.owned or any(self.session.status[side] in ("starting", "teaching") for side in SIDES):
            self.session.stop("B 回位：先结束双臂示教")
        self.report("[回位] 已请求；R 录制保持开启，回位和到位后的画面继续记录，最后按 S 保存")
        return True

    def stop(self, now, reason="键盘请求停止回位"):
        if not self.active:
            return
        self.report("[回位] " + reason)
        if self.phase == "waiting_teach":
            self.phase = "idle"  # No initialization/motion request was sent.
            return
        if self.phase != "stopping":
            self.phase, self.deadline = "stopping", now + 10.
            self.stop_futures, self.final_stop_sent = {}, False

    @staticmethod
    def result(future):
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("服务返回空结果" if result is None else result.message)
        return result

    def tick(self, now, ros_now=None, recording=False):
        if not self.active:
            return
        try:
            if self.phase == "waiting_teach":
                if self.session.blocked:
                    raise RuntimeError("示教清理未确认")
                if (self.session.phase != "idle" or self.session.owned or
                        not all(self.session.status[side] in ("idle", "stopped") for side in SIDES)):
                    if now >= self.deadline:
                        raise RuntimeError("等待双臂退出示教超时；未发回位命令")
                    return
                self.initialize_future = self.initializer.call_async(self.trigger_factory())
                self.phase, self.deadline = "initializing", now + 15.
            elif self.phase == "initializing":
                if not self.initialize_future.done():
                    if now >= self.deadline:
                        raise RuntimeError("双臂初始化结果超时")
                    return
                self.result(self.initialize_future)
                self.phase, self.deadline = "waiting_feedback", now + 15.
            elif self.phase == "waiting_feedback":
                # Wait for fresh measured joints AND cameras/IO before starting
                # a recorded return. No motion is issued into a known data gap.
                if recording and self.inputs is not None:
                    try:
                        self.inputs.snapshot(ros_now, now)
                    except NotReady as exc:
                        if now >= self.deadline:
                            raise RuntimeError(f"回位录制反馈未就绪，未发运动命令：{exc}") from exc
                        return
                if not all(client.service_is_ready() for client in self.moves.values()):
                    raise RuntimeError("MoveAbsJ 服务已离线")
                if not all(self.session.status[side] in ("idle", "stopped") for side in SIDES):
                    raise RuntimeError("双臂再次进入示教，取消回位")
                self.phase, self.deadline = "moving", now + 65.
                for side in SIDES:
                    request = self.move_factory()
                    request.joint_positions = self.targets[side]
                    request.speed_mm_s, request.joint_speed_scale = 50., .05
                    request.zone_mm, request.timeout_s = 0., 60.
                    request.joint_tolerance_rad, request.stable_samples = .01, 3
                    request.poll_interval_ms = 50
                    self.move_futures[side] = self.moves[side].call_async(request)
                self.report("[回位] 双臂正在回到保存的关节位置；Z 停止，S 停止保存录制")
            elif self.phase == "moving":
                for future in self.move_futures.values():
                    if future.done():
                        self.result(future)
                if all(future.done() for future in self.move_futures.values()):
                    self.phase = "idle"
                    self.report("[回位] 双臂均已到位；录制继续，按 S 保存；下次拖动按 A")
                elif now >= self.deadline:
                    raise RuntimeError("等待回位完成超时")
            elif self.phase == "stopping":
                self.tick_stopping(now)
        except Exception as exc:
            self.report(f"[回位失败] {exc}")
            if self.phase == "waiting_teach":
                self.phase = "idle"
            elif self.phase == "stopping":
                self.blocked, self.phase = True, "idle"
                self.report("[回位] 未确认双臂停止，禁止再次回位；请检查驱动状态")
            else:
                self.stop(now, "一侧失败或结果未知，同时请求双臂停止")

    def tick_stopping(self, now):
        pending = ([self.initialize_future] if self.initialize_future is not None else [])
        pending += list(self.move_futures.values())
        all_done = all(future.done() for future in pending)
        if all_done and not self.final_stop_sent:
            # A stop sent before a queued MoveAbsJ begins is insufficient. Send
            # a final pair after all original requests have returned.
            self.stop_futures, self.final_stop_sent = {}, True
        for side in SIDES:
            previous = self.stop_futures.get(side)
            if previous is None or (previous.done() and not self.final_stop_sent and
                                    now - self.last_stop[side] >= .25):
                if not self.stops[side].service_is_ready():
                    continue
                self.stop_futures[side] = self.stops[side].call_async(self.trigger_factory())
                self.last_stop[side] = now
        if (self.final_stop_sent and len(self.stop_futures) == 2 and
                all(future.done() for future in self.stop_futures.values())):
            for future in self.stop_futures.values():
                self.result(future)
            self.phase = "idle"
            self.report("[回位] 双臂停止已确认；录制保持开启，S 保存")
        elif now >= self.deadline:
            raise RuntimeError("双臂停止确认超时")


class KeyDecoder:
    """Ignore escape sequences, so arrow keys cannot accidentally invoke A/Z."""
    def __init__(self):
        self.escape = ""
        self.last_key, self.last_at = None, -math.inf

    def feed(self, text, at):
        result = []
        for char in text:
            if self.escape:
                if self.escape == "esc" and char in "[O":
                    self.escape = "sequence"
                elif self.escape == "esc" or ("@" <= char <= "~"):
                    self.escape = ""
                continue
            if char == "\x1b":
                self.escape = "esc"
                continue
            key = char.lower()
            if key not in "abzrsd1234hq":
                continue
            if key == self.last_key and at - self.last_at < .15:
                self.last_at = at
                continue
            self.last_key, self.last_at = key, at
            result.append(key)
        return result


class Keyboard:
    def __init__(self, events, line_input=False):
        self.events = events
        self.fd = sys.stdin.fileno()
        self.line = line_input or not os.isatty(self.fd)
        self.previous = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run_safely, name="teach-keyboard", daemon=True)

    def __enter__(self):
        if not self.line:
            self.previous = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        try:
            self.thread.start()
        except BaseException:
            if self.previous is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)
            raise
        return self

    def run_safely(self):
        try:
            self.run()
        except Exception as exc:
            self.events.put(("keyboard_error", exc))

    def run(self):
        decoder, buffered = KeyDecoder(), ""
        while not self.stopping.is_set():
            readable, _, _ = select.select([self.fd], [], [], .1)
            if not readable:
                continue
            data = os.read(self.fd, 128)
            if not data:
                self.events.put(("key", "q"))
                return
            text = data.decode("utf-8", errors="ignore")
            if self.line:
                buffered += text
                while "\n" in buffered:
                    line, buffered = buffered.split("\n", 1)
                    line = line.strip().lower()
                    if len(line) == 1 and line in "abzrsd1234hq":
                        self.events.put(("key", line))
            else:
                for key in decoder.feed(text, time.monotonic()):
                    self.events.put(("key", key))

    def __exit__(self, *unused):
        self.stopping.set()
        self.thread.join(1.)
        if self.previous is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)


@dataclass(frozen=True)
class Bundle:
    stamp: float
    state: np.ndarray
    action: np.ndarray
    images: dict
    source_stamps: np.ndarray
    suction_feedback_stamp: float
    suction_command_stamps: np.ndarray


class KeyboardInputs(OriginalInputs):
    """Align only 14 measured joints and three cameras; no controller or head/waist input."""
    def __init__(self, outputs, args):
        super().__init__(args.max_skew, args.max_age, args.state_max_age, required_fps=args.fps)
        self.outputs, self.suction_max_age = outputs, args.suction_max_age
        self.return_recording = False
        self.local_feedback = None
        self.sampling_origin = None
        # Network images retain exposure time. Wait for transport before pairing
        # them with the head image and joint history; arrival time is not a label.
        self.camera_lookahead = (min(self.max_skew, 1 / args.fps)
                                 if args.camera_lookahead is None else args.camera_lookahead)
        # Latest joint feedback must still be fresh under state_max_age. The
        # historical sample paired with a delayed image may legitimately be older.
        self.joint_history_age = min(self.max_age, self.state_max_age + max(0.,
            self.camera_lookahead - min(self.max_skew, 1 / args.fps)))
        self.receive_metrics = {key: deque(maxlen=600) for key in STREAMS}
        self.camera_driver_rates = {}
        for key in ("head_pha", "waist_pitch"):
            self.buffers.pop(key)
            self.rate_history.pop(key)

    def observe(self, key, stamp, received, ros_now, decode_ms=0.):
        with self.lock:
            self.receive_metrics[key].append((stamp, received, ros_now - stamp, decode_ms))

    def diagnostics(self, mono_now):
        result = {}
        with self.lock:
            for key, history in self.receive_metrics.items():
                rows = np.asarray([v for v in history if mono_now - v[1] <= 5.])
                if len(rows) < 2:
                    continue
                source_span, receive_span = rows[-1, :2] - rows[0, :2]
                result[key] = dict(messages=len(rows),
                    source_hz=(len(rows)-1)/source_span if source_span > 0 else None,
                    receive_hz=(len(rows)-1)/receive_span if receive_span > 0 else None,
                    latency_ms_p95=float(np.percentile(rows[:, 2], 95)*1000),
                    decode_ms_p95=float(np.percentile(rows[:, 3], 95)),
                    source_max_interval_ms=float(np.diff(rows[:, 0]).max()*1000))
                driver = self.camera_driver_rates.get(key)
                if driver is not None and mono_now - driver[1] <= 5.:
                    result[key]['camera_driver_hz'] = driver[0]
        return result

    def set_return_recording(self, enabled):
        with self.lock:
            if self.return_recording == enabled:
                return
            self.return_recording = enabled
            # Never mix a previous drag session's timestamps/rates with the
            # independent stream used during powered return and idle holding.
            for side in SIDES:
                self.buffers[side].clear()
                self.rate_history[side].clear()

    def push_local_joint_packet(self, packet, received, ros_now):
        with self.lock:
            if self.return_recording != (packet.source == "recording"):
                return False
            self.push(packet.side,packet.stamp_ns/1e9,packet.mono_ns/1e9,
                      np.asarray(packet.joints,dtype=np.float64))
            self.observe(packet.side,packet.stamp_ns/1e9,received,ros_now)
            return True

    def snapshot(self, ros_now, mono_now, previous_images=None, previous_sources=None):
        with self.lock:
            local_feedback = getattr(self, "local_feedback", None)
            if local_feedback is not None and local_feedback.error:
                raise NotReady("本机关节通道故障：" + local_feedback.error)
            self.outputs.check_ready(mono_now, self.suction_max_age)
            if self.clock_reset:
                self.clock_reset = False
                self.sampling_origin = None
                raise NotReady("ROS 时间回退，等待新数据")
            if not self.return_recording and not all(self.teaching.values()):
                raise NotReady("左右臂都必须处于 teaching 状态")
            for key in STREAMS:
                if not self.buffers[key]:
                    raise NotReady(f"等待 {key} 数据")
                age = self.max_age if key in CAMERAS else self.state_max_age
                if mono_now - self.buffers[key][-1].received > age:
                    raise NotReady(f"{key} 反馈停止")
            # Drive sampling from unused head-camera exposures, not a polling
            # timer or a two-second all-stream frequency veto. A short lookahead
            # lets delayed wrist frames arrive before choosing their nearest
            # exposure. Every saved image and joint timestamp must still advance.
            previous_sources, previous_images = previous_sources or {}, previous_images or {}
            unused = {key: [sample for sample in self.buffers[key]
                           if sample.stamp > max(previous_sources.get(key, -math.inf),
                                                 previous_images.get(key, -math.inf))]
                      for key in STREAMS}
            for key in STREAMS:
                if not unused[key]:
                    raise AwaitFrame(f"等待新的 {key} 样本")
            if self.sampling_origin is None:
                self.sampling_origin = unused["head"][0].stamp
            def bucket(stamp):
                return math.floor((stamp - self.sampling_origin) * self.required_fps + .05)
            previous_head = previous_sources.get("head", -math.inf)
            previous_bucket = (bucket(previous_head) if previous_head >= self.sampling_origin
                               else -math.inf)
            continuity_limit = 1.5 / self.required_fps + min(.02, .1 / self.required_fps)
            samples = None
            for reference in unused["head"]:
                if bucket(reference.stamp) <= previous_bucket:
                    continue
                if mono_now - reference.received < self.camera_lookahead:
                    continue
                if ros_now - reference.stamp > self.max_age:
                    continue
                wrists = {key: sorted((v for v in unused[key]
                           if abs(v.stamp - reference.stamp) <= self.max_skew),
                           key=lambda v: abs(v.stamp - reference.stamp))
                          for key in ("left_wrist", "right_wrist")}
                best = None
                for left in wrists["left_wrist"]:
                    for right in wrists["right_wrist"]:
                        low = min(reference.stamp, left.stamp, right.stamp)
                        high = max(reference.stamp, left.stamp, right.stamp)
                        if high - low > self.max_skew:
                            continue
                        candidate = dict(head=reference, left_wrist=left, right_wrist=right)
                        for side in SIDES:
                            values = [v for v in unused[side]
                                      if high - self.max_skew <= v.stamp <= low + self.max_skew and
                                      mono_now - v.received <= self.joint_history_age]
                            if not values:
                                break
                            candidate[side] = min(values, key=lambda v: (
                                v.stamp - previous_sources.get(side, v.stamp) > continuity_limit,
                                abs(v.stamp-reference.stamp)))
                        if len(candidate) == len(STREAMS):
                            # Camera phases drift across the nearest-exposure
                            # midpoint. Prefer a valid continuous combination
                            # before a tiny improvement in alignment creates a
                            # skipped source period. Keep the same skew limits
                            # and still expose genuine loss to Capture.tick.
                            breaks = sum(v.stamp - previous_sources.get(key, v.stamp) > continuity_limit
                                         for key, v in candidate.items())
                            score = (breaks, sum(abs(v.stamp-reference.stamp)
                                                 for v in candidate.values()))
                            if best is None or score < best[0]:
                                best = score, candidate
                if best is not None:
                    samples = best[1]
                    break
            if samples is None:
                raise AwaitFrame("等待可同步的新图像/关节组合")
            anchor = min(samples[key].stamp for key in CAMERAS)
            stamps = np.asarray([samples[key].stamp for key in STREAMS], dtype=np.float64)
            if np.ptp(stamps) > self.max_skew:
                raise NotReady("图像/关节时间戳超过同步容差")
            if ros_now - stamps.min() > self.max_age or stamps.max() - ros_now > .05:
                raise NotReady("源时间戳过期或超前")
            joints = [np.asarray(samples[side].value).reshape(-1) for side in SIDES]
            if any(len(value) != 7 for value in joints):
                raise NotReady("双臂关节维数必须是 7+7")
            joints = np.concatenate(joints).astype(np.float32)
            if not np.isfinite(joints).all() or np.any(np.abs(joints) > 2 * math.pi):
                raise NotReady("关节数据无效")
            anchor_mono = mono_now - (ros_now - anchor)
            suction = self.outputs.sample_at(anchor_mono, self.suction_max_age)
            state = np.concatenate((joints, suction.actual)).astype(np.float32)
            action = np.concatenate((joints, suction.target)).astype(np.float32)
            return Bundle(anchor, state, action, {key: samples[key].value for key in CAMERAS},
                          stamps, anchor + suction.feedback_at - anchor_mono,
                          anchor + np.asarray(suction.command_at) - anchor_mono)


def dataset_schema(args):
    return dict(version=2, schema_id=SCHEMA_ID, fps=float(args.fps), width=args.width,
                height=args.height, cameras=list(CAMERAS), state_names=list(STATE_NAMES),
                action_names=list(STATE_NAMES), units=["radian"] * 14 + ["binary"] * 2,
                source_order=list(STREAMS), suction_channels=dict(left=1, right=2),
                source_max_skew_seconds=args.max_skew,
                joint_action_source="same_sample_measured_joint_positions",
                suction_observation_source="verified_logical_DO_readback_not_vacuum_detection",
                suction_action_source="keyboard_requested_binary_target",
                suction_max_age_seconds=args.suction_max_age,
                suction_time_alignment="latest_IO_and_command_at_or_before_image_time")


class WriteQueueFull(RuntimeError):
    """Temporary backpressure; the aligned sample can be retried intact."""


class EpisodeWriter(OriginalWriter):
    def __init__(self, root, task, settings, *, image_queue_size=64):
        self.metrics_lock = threading.Lock()
        self.write_times, self.queue_waits = deque(maxlen=1800), deque(maxlen=1800)
        self.written_frames = self.written_bytes = self.queue_full_count = self.peak_queue = 0
        # 64 raw RGB VGA bundles occupy about 177 MB before encoding, bounded
        # independently of episode length. Three workers also serve 15/20 Hz.
        super().__init__(root, task, settings, image_workers=3, image_queue_size=image_queue_size)

    def _write_images(self):
        encoder = PngEncoder()
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    encoder.close()
                    return
                if self.error:
                    continue
                index, images, queued_at = job
                begin = time.monotonic()
                size = 0
                for key, rgb in images.items():
                    if rgb.shape != (self.settings["height"], self.settings["width"], 3) or rgb.dtype != np.uint8:
                        raise ValueError(f"invalid RGB shape/dtype for {key}")
                    path = self.path / "images" / key / f"{index:06d}.png"
                    size += encoder.write(path, rgb)
                with self.metrics_lock:
                    self.write_times.append(time.monotonic() - begin)
                    self.queue_waits.append(begin - queued_at)
                    self.written_frames += 1
                    self.written_bytes += size
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.jobs.task_done()

    def write_metrics(self):
        with self.metrics_lock:
            return dict(written_frames=self.written_frames, written_bytes=self.written_bytes,
                queue_capacity_frames=self.jobs.maxsize, image_workers=len(self.workers),
                png_encoder="libdeflate_sub_1" if PNG_COMPRESSOR else "zlib_store",
                queue_full_count=self.queue_full_count, peak_queue=self.peak_queue,
                write_ms_p50_p95_max=(np.percentile(self.write_times, [50,95,100])*1000).tolist()
                    if self.write_times else [],
                queue_wait_ms_p95=float(np.percentile(self.queue_waits,95)*1000)
                    if self.queue_waits else 0.)

    def check_capacity(self):
        self.check_error()
        if self.jobs.full():
            self.queue_full_count += 1
            raise WriteQueueFull("图像写盘队列已满")

    def add(self, bundle):
        self.check_error()
        try:
            self.jobs.put_nowait((len(self.rows), bundle.images, time.monotonic()))
            self.peak_queue = max(self.peak_queue, self.jobs.qsize())
        except queue.Full:
            self.queue_full_count += 1
            raise WriteQueueFull("图像写盘队列已满")
        self.rows.append((bundle.stamp, bundle.state.copy(), bundle.action.copy(),
                          bundle.source_stamps.copy(), bundle.suction_feedback_stamp,
                          bundle.suction_command_stamps.copy()))

    def flush(self, complete=False):
        self.jobs.join()
        self.check_error()
        if self.rows:
            stamp, state, action, source, feedback, command = zip(*self.rows)
            np.savez_compressed(self.path / "trajectory.npz",
                                state=np.stack(state).astype(np.float32),
                                action=np.stack(action).astype(np.float32),
                                ros_timestamp=np.asarray(stamp, dtype=np.float64),
                                source_timestamp=np.stack(source),
                                suction_feedback_timestamp=np.asarray(feedback, dtype=np.float64),
                                suction_command_timestamp=np.stack(command),
                                timestamp=np.arange(len(self.rows), dtype=np.float64) / self.settings["fps"])
        metadata = dict(task=self.task, frames=len(self.rows), segments=self.segments,
                        complete=complete, **self.settings)
        if hasattr(self, "recording_summary"):
            metadata["recording_summary"] = self.recording_summary
            metadata["capture_gaps"] = self.capture_gaps
        if hasattr(self, "control_events"):
            metadata["control_events"] = self.control_events
        if hasattr(self, "capture_diagnostics"):
            metadata["capture_diagnostics"] = self.capture_diagnostics
        metadata["writer_diagnostics"] = self.write_metrics()
        (self.path / "episode.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        if complete:
            # Verify temporal/label invariants before the staging directory can
            # become a completed episode. Image workers already joined above.
            checked, arrays = validate_episode(self.path, self.settings, check_images=False)
            self.training_quality = metadata["training_quality"] = episode_quality(checked, arrays)
            (self.path / "episode.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


class Capture:
    def __init__(self, inputs, writer_factory, fps, report=notify):
        self.inputs, self.writer_factory, self.fps, self.report = inputs, writer_factory, fps, report
        self.writer = None
        self.recording = False
        self.previous_sources = self.previous_images = None
        self.last_tick = self.next_tick = None
        self.started_at = self.last_report_at = self.last_ros_now = None
        self.segment_open = False
        self.waiting = None
        self.last_poll_at = None
        self.max_poll_interval = 0.
        self.scheduler_delay_count = 0
        self.wait_reason_counts = Counter()
        self.pending_bundle = None
        self.backpressure_since = None
        self.backpressure_seconds = 0.
        self.pending_discard_count = 0

    def start(self, ros_now, mono_now):
        if self.recording:
            return
        if self.writer is not None:
            self.report("先按 S 保存或 D 丢弃上一条记录")
            return
        self.writer = self.writer_factory()
        self.writer.capture_gaps = []
        self.writer.recording_summary = {}
        self.writer.control_events = []
        self.writer.begin_segment()
        self.segment_open = True
        self.waiting = None
        # Start at R, rather than exporting buffered images from before R.
        self.previous_sources = {key: ros_now - 1e-6 for key in STREAMS}
        self.previous_images = None
        self.last_tick, self.next_tick = mono_now - 1 / self.fps, mono_now
        self.started_at = self.last_report_at = mono_now
        self.last_ros_now = ros_now
        self.last_poll_at = mono_now
        self.max_poll_interval = 0.
        self.scheduler_delay_count = 0
        self.wait_reason_counts.clear()
        self.pending_bundle = None
        self.backpressure_since = None
        self.backpressure_seconds = 0.
        self.pending_discard_count = 0
        self.recording = True
        local_feedback = self.inputs.local_feedback if isinstance(self.inputs,KeyboardInputs) else None
        self.local_feedback_baseline = local_feedback.diagnostics() if local_feedback is not None else None
        self.update_summary(mono_now)
        self.report("录制开始；持续录制直到 S 停止保存，D 丢弃；临时缺数据会自动等待恢复")

    def update_summary(self, mono_now, stopped_reason=None):
        summary = dict(elapsed_seconds=max(0., mono_now - self.started_at),
                       frames=len(self.writer.rows), target_fps=self.fps,
                       saved_fps=len(self.writer.rows)/max(1e-6, mono_now-self.started_at),
                       nominal_data_seconds=len(self.writer.rows) / self.fps,
                       gap_count=len(self.writer.capture_gaps))
        if stopped_reason is not None:
            summary["stopped_reason"] = stopped_reason
        self.writer.recording_summary = summary
        self.writer.capture_diagnostics = dict(
            receive=self.inputs.diagnostics(mono_now) if isinstance(self.inputs, KeyboardInputs) else {},
            wait_reason_counts=dict(self.wait_reason_counts),
            scheduler_delay_count=self.scheduler_delay_count,
            max_poll_interval_ms=self.max_poll_interval*1000,
            write_backpressure_seconds=self.backpressure_seconds + (
                max(0., mono_now-self.backpressure_since) if self.backpressure_since is not None else 0.),
            pending_samples_discarded_on_control_change=self.pending_discard_count)
        local_feedback = self.inputs.local_feedback if isinstance(self.inputs,KeyboardInputs) else None
        if local_feedback is not None:
            self.writer.capture_diagnostics["local_joint_feedback"] = dict(
                at_record_start=getattr(self,"local_feedback_baseline",None),
                current=local_feedback.diagnostics())

    def clear_pending(self, mono_now):
        if self.pending_bundle is not None:
            self.pending_discard_count += 1
            self.pending_bundle = None
        if self.backpressure_since is not None:
            self.backpressure_seconds += max(0., mono_now-self.backpressure_since)
            self.backpressure_since = None

    def mark_gap(self, reason, mono_now=None):
        """Keep R armed, closing only the continuous segment with missing data."""
        if not self.recording:
            return
        self.wait_reason_counts[reason] += 1
        now = time.monotonic() if mono_now is None else mono_now
        self.clear_pending(now)
        if self.segment_open:
            self.writer.end_segment(reason)
            self.segment_open = False
        if self.waiting is None:
            start = min(now, max(self.started_at, self.last_tick + 1 / self.fps))
            self.waiting = dict(after_frame=len(self.writer.rows), reasons=[reason],
                                start_elapsed_seconds=start - self.started_at,
                                end_elapsed_seconds=None)
            self.writer.capture_gaps.append(self.waiting)
            self.report(f"录制仍开启，等待有效数据：{reason}；恢复后自动继续，S 停止保存")
        elif reason not in self.waiting["reasons"]:
            self.waiting["reasons"].append(reason)

    def close_gap(self, mono_now):
        if self.waiting is not None:
            self.waiting["end_elapsed_seconds"] = max(
                self.waiting["start_elapsed_seconds"], mono_now - self.started_at)
            self.waiting = None

    def progress(self, mono_now):
        if mono_now - self.last_report_at >= 5.:
            self.update_summary(mono_now)
            status = "等待数据：" + self.waiting["reasons"][-1] if self.waiting is not None else "正在采样"
            if self.backpressure_since is not None:
                status = "写盘排队；已对齐样本保留重试"
            self.report(f"[录制中] {len(self.writer.rows)} 帧，已持续 {mono_now - self.started_at:.1f} 秒，"
                        f"平均保存 {self.writer.recording_summary['saved_fps']:.2f} 帧/秒；{status}")
            if isinstance(self.inputs, KeyboardInputs):
                rates = self.inputs.diagnostics(mono_now)
                cameras = " ".join(f"{key}={rates[key]['receive_hz']:.1f}Hz"
                                   for key in CAMERAS if rates.get(key, {}).get('receive_hz') is not None)
                slow = [key for key in CAMERAS if rates.get(key, {}).get('receive_hz') is not None and
                        rates[key]['receive_hz'] < self.fps*.95]
                if slow:
                    cameras += '；低于目标：' + ','.join(slow) + '（继续保存有效帧）'
                writer = self.writer.write_metrics()
                write_p95 = writer['write_ms_p50_p95_max'][1] if writer['write_ms_p50_p95_max'] else 0.
                self.report(f"[流水线] 收图 {cameras}；写盘已完成 {writer['written_frames']} 帧，"
                            f"队列 {self.writer.jobs.qsize()}/{self.writer.jobs.maxsize}，"
                            f"满队列 {writer['queue_full_count']} 次，三图写入p95={write_p95:.1f}ms；"
                            f"实际调度延迟 {self.scheduler_delay_count} 次")
            self.last_report_at = mono_now

    def pause(self, reason, mono_now=None):
        if self.recording:
            now = time.monotonic() if mono_now is None else mono_now
            self.clear_pending(now)
            self.recording = False
            if self.segment_open:
                self.writer.end_segment(reason)
                self.segment_open = False
            self.close_gap(now)
            self.update_summary(now, stopped_reason=reason)
            self.report(f"录制停止：{reason}；共 {len(self.writer.rows)} 帧")

    def finish(self, save, mono_now=None):
        self.pause("keyboard save" if save else "keyboard discard", mono_now)
        if self.writer is None:
            self.report("没有待处理的记录")
            return
        if save:
            if self.writer.rows:
                path = self.writer.save()
                summary = self.writer.recording_summary
                self.report(f"Saved: {path}（{len(self.writer.rows)} 帧，录制 {summary['elapsed_seconds']:.1f} 秒，"
                            f"平均保存 {summary['saved_fps']:.2f} 帧/秒，"
                            f"有效数据约 {summary['nominal_data_seconds']:.2f} 秒）")
                quality = getattr(self.writer, "training_quality", {})
                for warning in quality.get("warnings", []):
                    self.report("[训练数据需复核] " + warning)
            else:
                path = self.writer.preserve()
                self.report(f"本条没有有效帧，未生成训练 episode；诊断记录保留为：{path}")
        else:
            self.writer.discard()
            self.report("当前未保存记录已丢弃")
        self.writer = None

    def tick(self, ros_now, mono_now):
        if not self.recording:
            return
        poll_interval = max(0., mono_now - self.last_poll_at)
        self.last_poll_at = mono_now
        self.max_poll_interval = max(self.max_poll_interval, poll_interval)
        if poll_interval > 2 / self.fps + .02:
            self.scheduler_delay_count += 1
        self.progress(mono_now)
        if ros_now < self.last_ros_now:
            self.mark_gap("ROS 时间回退，等待新数据", mono_now)
            self.previous_sources = self.previous_images = None
        self.last_ros_now = ros_now
        # Snapshot selects unused source exposures. Polling more often than the
        # camera period does not duplicate frames and avoids phase-lock skips.
        try:
            self.writer.check_error()
            # Freeze a valid measured sample before enqueueing. A disk stall
            # must not resample its joints/IO or repeatedly split its segment.
            if self.pending_bundle is None:
                self.pending_bundle = self.inputs.snapshot(
                    ros_now, mono_now, self.previous_images, self.previous_sources)
            self.writer.check_capacity()
            bundle = self.pending_bundle
            self.pending_bundle = None
            if self.previous_sources is not None:
                gap = 1.5 / self.fps + min(.02, .1 / self.fps)
                jumps = [f"{key}={(bundle.source_stamps[i] - self.previous_sources[key]) * 1000:.1f}ms"
                         for i, key in enumerate(STREAMS)
                         if bundle.source_stamps[i] - self.previous_sources[key] > gap]
                if jumps:
                    self.mark_gap("图像/关节跳过了采集周期：" + ", ".join(jumps), mono_now)
            if not self.segment_open:
                self.writer.begin_segment()
                self.segment_open = True
            self.writer.add(bundle)
        except WriteQueueFull:
            # Capacity is checked before any segment edits. There is only one
            # producer, so workers can only free space between check and add.
            if self.pending_bundle is None:
                self.pending_bundle = bundle
            if self.backpressure_since is None:
                self.backpressure_since = mono_now
            return
        except AwaitFrame:
            return
        except (NotReady, RuntimeError, OSError) as exc:
            self.mark_gap(str(exc), mono_now)
            return
        if self.backpressure_since is not None:
            self.backpressure_seconds += max(0., mono_now-self.backpressure_since)
            self.backpressure_since = None
        if self.waiting is not None:
            self.close_gap(mono_now)
            self.report(f"有效数据已恢复，继续同一条录制；已记录 {len(self.writer.rows)} 帧")
        self.last_tick = mono_now
        self.next_tick = max(self.next_tick + 1 / self.fps, mono_now)
        self.previous_sources = dict(zip(STREAMS, bundle.source_stamps))
        self.previous_images = {key: self.previous_sources[key] for key in CAMERAS}


def validate_schema(settings):
    if (settings.get("schema_id") != SCHEMA_ID or settings.get("version") != 2 or
            settings.get("state_names") != list(STATE_NAMES) or settings.get("action_names") != list(STATE_NAMES) or
            settings.get("source_order") != list(STREAMS) or settings.get("cameras") != list(CAMERAS) or
            settings.get("units") != ["radian"] * 14 + ["binary"] * 2 or
            settings.get("suction_channels") != dict(left=1, right=2) or
            settings.get("suction_action_source") != "keyboard_requested_binary_target"):
        raise ValueError("数据不是键盘双臂+双吸盘 schema，不能与原头腰16维混用")
    if not 0 < settings["fps"] <= 30 or min(settings["width"], settings["height"]) <= 0:
        raise ValueError("无效的采样率或图像尺寸")
    for key in ("suction_max_age_seconds", "source_max_skew_seconds"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("无效的时间同步/读回时效设置")


def validate_episode(path, settings, check_images=True):
    validate_schema(settings)
    path = Path(path)
    meta = json.loads((path / "episode.json").read_text())
    if not meta.get("complete") or any(meta.get(key) != value for key, value in settings.items()):
        raise ValueError("未确认的记录或 schema 不一致")
    if not isinstance(meta.get("task"), str) or not meta["task"].strip():
        raise ValueError("任务描述缺失")
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    n = meta["frames"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("空 episode")
    for key, shape in (("state", (n, 16)), ("action", (n, 16)), ("source_timestamp", (n, 5)),
                       ("ros_timestamp", (n,)), ("timestamp", (n,)),
                       ("suction_feedback_timestamp", (n,)), ("suction_command_timestamp", (n, 2))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"无效的 {key} 数据")
    if not np.array_equal(arrays["state"][:, :14], arrays["action"][:, :14]):
        raise ValueError("14维关节动作必须与同采样实测位置一致")
    if np.any(np.abs(arrays["state"][:, :14]) > 2 * math.pi):
        raise ValueError("关节位置超过合理范围")
    if np.any(np.ptp(arrays["source_timestamp"], axis=1) > settings["source_max_skew_seconds"] + 1e-5):
        raise ValueError("源时间戳超过同步容差")
    if np.any(np.abs(arrays["source_timestamp"]-arrays["ros_timestamp"][:, None]) >
              settings["source_max_skew_seconds"]+1e-5):
        raise ValueError("采样时间与图像/关节源时间戳不匹配")
    if not np.allclose(arrays["timestamp"], np.arange(n) / settings["fps"], atol=1e-6):
        raise ValueError("数据时间轴与采样率不一致")
    for key in ("state", "action"):
        if not np.isin(arrays[key][:, 14:], (0., 1.)).all():
            raise ValueError("吸盘状态/动作必须是0或1")
    stamp = arrays["ros_timestamp"]
    age = stamp - arrays["suction_feedback_timestamp"]
    if (np.any(age < -1e-5) or np.any(age > settings["suction_max_age_seconds"] + 1e-5) or
            np.any(arrays["suction_command_timestamp"] > stamp[:, None] + 1e-5)):
        raise ValueError("吸盘时间戳过期或使用了未来数据")
    cursor = 0
    for segment in meta["segments"]:
        start, end = segment["start"], segment["end"]
        if start != cursor or not start < end <= n:
            raise ValueError("连续片段边界不合法")
        if np.any(np.diff(stamp[start:end]) <= 0):
            raise ValueError("片段内时间戳未递增")
        if np.any(np.diff(arrays["source_timestamp"][start:end], axis=0) <= 0):
            raise ValueError("片段内使用了重复或倒退的图像/关节样本")
        gap_limit = 1.5 / settings["fps"] + min(.02, .1 / settings["fps"])
        if np.any(np.diff(arrays["source_timestamp"][start:end], axis=0) > gap_limit + 1e-5):
            raise ValueError("片段内存在未分段的真实缺帧，不能跨间断训练动作块")
        cursor = end
    if cursor != n:
        raise ValueError("有帧未归属于连续片段")
    if check_images:
        for key in CAMERAS:
            for index in range(n):
                image = cv2.imread(str(path / "images" / key / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
                if (image is None or image.shape != (settings["height"], settings["width"], 3)
                        or image.dtype != np.uint8):
                    raise ValueError(f"图像缺失/尺寸错误：{key}/{index}")
    return meta, arrays


def export_dataset(source, destination, repo_id, factory=None):
    source, destination = Path(source), Path(destination)
    settings = json.loads((source / "schema.json").read_text())
    validate_schema(settings)
    if not float(settings["fps"]).is_integer():
        raise ValueError("LeRobot 导出要求整数采样率，不能截断 fps")
    episodes = sorted(source.glob("episode_*"))
    if not episodes:
        raise ValueError("没有完成的 episode，.partial 不参与导出")
    if destination.exists():
        raise FileExistsError("导出目录已经存在，请使用新目录")
    for path in episodes:
        validate_episode(path, settings)
    if factory is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError("导出需要 LeRobot 0.4.3 环境；原始采集不需要") from exc
        factory = LeRobotDataset.create
    features = {key: dict(dtype="float32", shape=(16,), names=list(STATE_NAMES))
                for key in ("observation.state", "action")}
    features.update({f"observation.images.{key}": dict(
        dtype="video", shape=(settings["height"], settings["width"], 3),
        names=["height", "width", "channels"]) for key in CAMERAS})
    dataset = factory(repo_id=repo_id, root=destination, fps=int(settings["fps"]), features=features,
                      robot_type="xcore_dual_arm_suction_keyboard", use_videos=True, video_backend="pyav",
                      vcodec="h264", image_writer_processes=0, image_writer_threads=3)
    mapping = []
    try:
        for path in episodes:
            meta, arrays = validate_episode(path, settings, check_images=False)
            for segment in meta["segments"]:
                for index in range(segment["start"], segment["end"]):
                    frame = {"observation.state": arrays["state"][index].copy(),
                             "action": arrays["action"][index].copy(), "task": meta["task"]}
                    for key in CAMERAS:
                        bgr = cv2.imread(str(path / "images" / key / f"{index:06d}.png"))
                        frame[f"observation.images.{key}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    dataset.add_frame(frame)
                if dataset.image_writer is not None:
                    dataset.image_writer.wait_until_done()
                dataset.save_episode()
                mapping.append(dict(episode_index=len(mapping), raw_episode=path.name, **segment))
        finalize = getattr(dataset, "finalize", None)
        if finalize:
            finalize()
        (destination / "meta" / "keyboard_teach_source.json").write_text(
            json.dumps(dict(schema=settings, episodes=mapping), indent=2) + "\n")
    finally:
        stop = getattr(dataset, "stop_image_writer", None)
        if stop:
            stop()
    return len(mapping)


def make_node(args, inputs, events):
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from rokae_interfaces.srv import MoveAbsJ
    from diagnostic_msgs.msg import DiagnosticArray
    from xcore_teach_recording.ros_node import decode_rgb, joint_values
    from xcore_teach_recording.udp_preview import UdpPreview

    class KeyboardTeachNode(Node):
        def __init__(self):
            super().__init__("keyboard_teach_suction", namespace=PREFIX)
            self.starts, self.stops, self.subscriptions_kept = {}, {}, []
            self.return_moves, self.motion_stops = {}, {}
            self.initializer = self.create_client(Trigger, f"{PREFIX}/initialize/dual_arm")
            self.move_request_factory = MoveAbsJ.Request
            self.preview = None
            self.local_feedback = None
            self.last_warning = -math.inf
            self.feedback_trace = FeedbackTrace(os.environ.get("AIDE_FEEDBACK_TRACE_DIR"))
            if self.feedback_trace.enabled:
                notify(f"AIDE_FEEDBACK_RECEIVE_TRACE={self.feedback_trace.path}")
                self.trace_timer = self.create_timer(.05, lambda: self.feedback_trace.emit("executor.heartbeat"))
            elif self.feedback_trace.error:
                self.get_logger().warning("Feedback trace unavailable: " + self.feedback_trace.error)
            sensor = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
            status = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
            for side in SIDES:
                self.starts[side] = self.create_client(Trigger, f"{PREFIX}/cartesian_drag_safe/start/{side}_arm")
                self.stops[side] = self.create_client(Trigger, f"{PREFIX}/cartesian_drag_safe/stop/{side}_arm")
                self.return_moves[side] = self.create_client(MoveAbsJ, f"{PREFIX}/move_absj/{side}_arm")
                self.motion_stops[side] = self.create_client(Trigger, f"{PREFIX}/stop/{side}_arm")
                self.subscriptions_kept.append(self.create_subscription(
                    String, f"{PREFIX}/cartesian_drag_safe/status/{side}_arm",
                    lambda msg, key=side: self.on_status(key, msg.data), status))
                if inputs is not None and args.joint_feedback == "ros":
                    self.subscriptions_kept.append(self.create_subscription(
                        JointState, f"{PREFIX}/cartesian_drag_safe/joint_states/{side}_arm",
                        lambda msg, key=side: self.on_joints(key, msg), sensor))
                    self.subscriptions_kept.append(self.create_subscription(
                        JointState, f"{PREFIX}/recording_joint_states/{side}_arm",
                        lambda msg, key=side: self.on_joints(key, msg, for_return=True), sensor))
            if inputs is not None:
                self.subscriptions_kept.append(self.create_subscription(
                    DiagnosticArray, '/robot/sensors/cameras/diagnostics', self.on_camera_diagnostics, 20))
                for key in CAMERAS:
                    self.subscriptions_kept.append(self.create_subscription(
                        Image, f"/robot/sensors/cameras/{key}/color/image_raw",
                        lambda msg, camera=key: self.on_image(camera, msg),
                        QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)))
                if args.udp_video:
                    self.preview = UdpPreview(args.udp_video_ip, args.udp_video_jpeg_quality,
                                              on_error=self.get_logger().warning)
            if inputs is not None and args.joint_feedback == "local":
                try:
                    self.local_feedback = LocalFeedbackReceiver(self.on_local_joints,
                        args.joint_feedback_port, max_age=args.state_max_age)
                except BaseException:
                    self.feedback_trace.close()
                    raise
                inputs.local_feedback = self.local_feedback
                notify(f"关节数采使用独立本机通道 127.0.0.1:{args.joint_feedback_port}；ROS图像配对不变")

        def on_status(self, side, value):
            self.feedback_trace.emit(side + ".status", state=value)
            if inputs is not None:
                with inputs.lock:
                    inputs.teaching[side] = value == "teaching"
            events.put(("status", (side, value)))

        def warn(self, message):
            now = time.monotonic()
            if now - self.last_warning > 2.:
                self.get_logger().warning(str(message))
                self.last_warning = now

        def on_camera_diagnostics(self, message):
            for status in message.status:
                for key in CAMERAS:
                    if status.name.endswith(key + ': color'):
                        for value in status.values:
                            if value.key == 'Actual frequency (Hz)':
                                try:
                                    rate = float(value.value)
                                    if math.isfinite(rate) and rate >= 0:
                                        with inputs.lock:
                                            inputs.camera_driver_rates[key] = rate, time.monotonic()
                                except ValueError:
                                    pass

        def on_joints(self, side, message, for_return=False):
            callback_ns = time.monotonic_ns() if self.feedback_trace.enabled else None
            accepted = False
            try:
                received = time.monotonic()
                names = [f"{side}_joint_{i}" for i in range(1, 8)]
                values = joint_values(message, names)
                stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
                with inputs.lock:
                    if inputs.return_recording == for_return:
                        inputs.push(side, stamp, received, values)
                        inputs.observe(side, stamp, received, self.ros_now())
                        accepted = True
            except (ValueError, KeyError) as exc:
                self.warn(exc)
            finally:
                self.feedback_trace.emit(side + (".recording" if for_return else ".drag"),
                    message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec,
                    callback_ns=callback_ns, accepted=accepted)

        def on_local_joints(self, packet, received):
            callback_ns = time.monotonic_ns() if self.feedback_trace.enabled else None
            # Arrival time must not make a stale SDK sample fresh.
            accepted = inputs.push_local_joint_packet(packet,received,self.ros_now())
            self.feedback_trace.emit(packet.side+".local_"+packet.source,packet.stamp_ns,
                callback_ns=callback_ns,accepted=accepted,sequence=packet.sequence,
                sample_mono_ns=packet.mono_ns,ros_queue_overwrites=packet.ros_overwrites)

        def on_image(self, key, message):
            callback_ns = time.monotonic_ns() if self.feedback_trace.enabled else None
            accepted = False
            try:
                received = time.monotonic()
                begin = time.perf_counter()
                image = decode_rgb(message, args.width, args.height)
                decode_ms = (time.perf_counter() - begin)*1000
                stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
                inputs.push(key, stamp, received, image)
                inputs.observe(key, stamp, received, self.ros_now(), decode_ms)
                accepted = True
                if self.preview is not None:
                    self.preview.offer(key, image)
            except ValueError as exc:
                self.warn(exc)
            finally:
                self.feedback_trace.emit(key + ".camera",
                    message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec,
                    callback_ns=callback_ns, accepted=accepted)

        def ros_now(self):
            return self.get_clock().now().nanoseconds / 1e9

        def destroy_node(self):
            if self.local_feedback is not None:
                self.local_feedback.close()
                self.local_feedback = None
            self.feedback_trace.close()
            if self.feedback_trace.error:
                notify("Feedback trace error: " + self.feedback_trace.error)
            if self.preview is not None:
                self.preview.close()
            return super().destroy_node()

    return KeyboardTeachNode(), Trigger.Request


def handle_key(key, session, capture, outputs, ros_now, mono_now, return_control=None):
    if key == "q":
        return True
    if key == "a":
        if return_control is not None and (return_control.active or return_control.blocked):
            notify("正在回位/停止或停止未确认，请先等待回位结束")
        elif session.start(mono_now) and return_control is not None and return_control.inputs is not None:
            return_control.inputs.set_return_recording(False)
            if capture is not None:
                capture.mark_gap("切换回拖动示教关节反馈", mono_now)
    elif key == "b" and return_control is not None:
        if return_control.start(mono_now) and capture is not None and capture.recording:
            capture.mark_gap("B 回位：等待示教退出及回位反馈就绪，随后继续同一条录制", mono_now)
            capture.writer.control_events.append(dict(
                key="b", ros_timestamp=ros_now, after_frame=len(capture.writer.rows),
                action="return_to_saved_joints", joint_targets=return_control.targets))
    elif key == "z":
        if return_control is not None and return_control.active:
            return_control.stop(mono_now)
            return False
        if capture is not None:
            capture.mark_gap("键盘结束示教，等待下次 A 开启示教", mono_now)
        session.stop()
    elif key in "1234":
        outputs.set("left" if key in "12" else "right", key in "13", at=mono_now)
    elif key == "r" and capture is not None:
        capture.start(ros_now, mono_now)
    elif key in "sd" and capture is not None:
        capture.finish(key == "s", mono_now)
    elif key == "h":
        notify(HELP)
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--controls-only", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--export-to", type=Path)
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", default="双臂吸盘拖动示教")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--joint-feedback", choices=("local","ros"), default="local",
                        help="local=独立本机关节通道；ros=兼容旧ROS反馈（可能受DDS堵塞影响）")
    parser.add_argument("--joint-feedback-port",type=int,default=JOINT_FEEDBACK_PORT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-skew", type=float, default=None)
    parser.add_argument("--max-age", type=float, default=.5)
    parser.add_argument("--camera-lookahead", type=float, default=None,
                        help="配对图像前等待秒数；网络腕相机一键启动默认0.20，保留采集时间戳")
    parser.add_argument("--state-max-age", type=float, default=.25)
    parser.add_argument("--suction-max-age", type=float, default=.5)
    parser.add_argument("--check-seconds", type=float, default=15.)
    parser.add_argument("--shutdown-timeout", type=float, default=10.)
    parser.add_argument("--line-input", action="store_true")
    parser.add_argument("--udp-video", action="store_true", default=False)
    parser.add_argument("--no-udp-video", action="store_false", dest="udp_video")
    parser.add_argument("--udp-video-ip", default="172.16.13.184")
    parser.add_argument("--udp-video-jpeg-quality", type=int, default=50)
    parser.add_argument("--repo-id", default="local/xcore_keyboard_dual_suction")
    parser.add_argument("--m050-host", default="172.16.13.167")
    parser.add_argument("--m050-port", type=int, default=20108)
    parser.add_argument("--m050-unit", type=int, default=1)
    parser.add_argument("--m050-timeout-ms", type=int, default=3000)
    parser.add_argument("--m050-source-ip", default=None)
    parser.add_argument("--m050-poll-seconds", type=float, default=.1)
    parser.add_argument("--m050-library", type=Path, default=ROOT / "hardware/USR-M050/build/libm050.so")
    args = parser.parse_args(argv)
    if not args.task.strip() or not 1 <= args.fps <= 30 or min(args.width, args.height) <= 0:
        parser.error("任务、采样率(1..30)、图像尺寸无效")
    if not 1 <= args.joint_feedback_port <= 65535:
        parser.error("joint-feedback-port 必须在1..65535内")
    if args.max_skew is None:
        args.max_skew = min(.15, 1 / args.fps)
    if args.camera_lookahead is not None and (not math.isfinite(args.camera_lookahead) or
            not 0 <= args.camera_lookahead < min(args.max_age, .3)):
        parser.error("camera-lookahead 必须在 [0, min(max-age, 0.3)) 内")
    for name in ("max_skew", "max_age", "state_max_age", "suction_max_age", "check_seconds",
                 "shutdown_timeout", "m050_poll_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " 必须是有限正数")
    if not 1 <= args.m050_port <= 65535 or not 1 <= args.m050_unit <= 247:
        parser.error("M050 port=1..65535，unit=1..247")
    if not 1 <= args.m050_timeout_ms <= 2147483647:
        parser.error("M050 timeout 必须是正整数毫秒")
    if not 5 <= args.udp_video_jpeg_quality <= 100:
        parser.error("JPEG quality=5..100")
    try:
        for address in (args.m050_host, args.udp_video_ip, args.m050_source_ip):
            if address:
                ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError as exc:
        parser.error(str(exc))
    args.root = args.root.expanduser().resolve()
    return args


def run(args):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    node = executor = thread = outputs = capture = lock_file = session = return_control = None
    settings = dataset_schema(args)
    events = queue.SimpleQueue()
    code = 0
    previous_signals = {}
    try:
        if not args.check and not args.controls_only:
            root = args.root
            root.mkdir(parents=True, exist_ok=True)
            lock_file = (root / ".keyboard_recorder.lock").open("a")
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare_root(root, settings)
        outputs = SuctionOutputs(M050Device(args), args.m050_poll_seconds, read_only=args.check)
        outputs.start()
        inputs = None if args.controls_only else KeyboardInputs(outputs, args)
        rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
        node, request_factory = make_node(args, inputs, events)
        session = DualTeaching(node.starts, node.stops, request_factory)
        if not args.check:
            try:
                targets = load_return_pose()
            except (ValueError, OSError) as exc:
                targets = None
                notify(f"B 回位点不可用：{exc}；其他采集功能仍可使用")
            return_control = DualArmReturn(
                session, node.initializer, node.return_moves, node.motion_stops,
                request_factory, node.move_request_factory, inputs, targets)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except Exception as exc:
                events.put(("executor_error", exc))

        thread = threading.Thread(target=spin, name="keyboard-teach-ros", daemon=True)
        thread.start()
        if node.local_feedback is not None:
            deadline = time.monotonic()+5.
            while not node.local_feedback.ready() and time.monotonic()<deadline:
                if node.local_feedback.error:
                    break
                time.sleep(.01)
            if not node.local_feedback.ready():
                raise RuntimeError("本机关节反馈未就绪。请退出旧双臂驱动后使用一键入口，"
                    f"或以 AIDE_JOINT_FEEDBACK_PORT={args.joint_feedback_port} 启动新版驱动；"
                    "旧ROS通道可显式指定 --joint-feedback ros")
        if not args.check and not args.controls_only:
            capture = Capture(inputs, lambda: EpisodeWriter(args.root, args.task, settings), args.fps)
        if args.check:
            deadline, last_wait = time.monotonic() + args.check_seconds, None
            while rclpy.ok():
                while True:
                    try:
                        kind, value = events.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "executor_error":
                        raise RuntimeError(f"ROS 接收线程失败：{value}")
                try:
                    inputs.snapshot(node.ros_now(), time.monotonic())
                except NotReady as exc:
                    if str(exc) != last_wait:
                        notify("等待：" + str(exc))
                        last_wait = str(exc)
                else:
                    notify(f"Ready: 目标 {args.fps} Hz，14关节+2吸盘；频率为诊断值，不再阻断有效帧；" +
                           str(inputs.diagnostics(time.monotonic())))
                    break
                if time.monotonic() >= deadline:
                    notify("就绪检查超时")
                    code = 2
                    break
                time.sleep(.01)
        else:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_signals[signum] = signal.signal(signum, lambda *_: events.put(("key", "q")))
            notify(HELP)
            notify("仅使用键盘；按住机械臂实体按钮拖动。头腰电机与遥操程序无需启动。")
            notify(f"B 固定回位点：{RETURN_POSE_PATH}；启动不运动，B 回位继续录制，最后 S 保存")
            with Keyboard(events, args.line_input):
                quitting = False
                while rclpy.ok() and not quitting:
                    for _ in range(100):
                        try:
                            kind, value = events.get_nowait()
                        except queue.Empty:
                            break
                        if kind in ("executor_error", "keyboard_error"):
                            raise RuntimeError(f"输入/接收线程失败：{value}")
                        if kind == "status":
                            side, status = value
                            session.on_status(side, status)
                            if status != "teaching" and capture is not None:
                                capture.mark_gap(f"{side} teaching: {status}", time.monotonic())
                        elif kind == "key":
                            try:
                                quitting = handle_key(value, session, capture, outputs,
                                                      node.ros_now(), time.monotonic(), return_control)
                            except (RuntimeError, OSError) as exc:
                                notify(str(exc))
                            if quitting:
                                break
                    session.tick(time.monotonic())
                    if return_control is not None and not quitting:
                        return_control.tick(time.monotonic(), node.ros_now(),
                                            capture is not None and capture.recording)
                    if capture is not None and not quitting:
                        capture.tick(node.ros_now(), time.monotonic())
                    time.sleep(.005)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        notify(f"程序失败：{exc}")
        code = 1
    finally:
        if capture is not None:
            capture.pause("recorder exiting")
        if return_control is not None and return_control.active:
            return_control.stop(time.monotonic(), "程序退出，请求停止双臂回位")
        if session is not None and session.owned:
            session.stop("程序退出")
        if return_control is not None and return_control.active:
            deadline = time.monotonic() + args.shutdown_timeout
            while return_control.active and time.monotonic() < deadline:
                return_control.tick(time.monotonic())
                time.sleep(.01)
            if return_control.active or return_control.blocked:
                notify("未获得双臂回位停止确认，请检查驱动和机器人状态")
                code = 1
        if capture is not None and capture.writer is not None:
            try:
                notify(f"未确认数据保留为：{capture.writer.preserve()}")
            except Exception as exc:
                notify(f"保留未完成记录失败：{exc}")
                code = 1
        if session is not None and session.owned:
            deadline = time.monotonic() + args.shutdown_timeout
            while session.owned and time.monotonic() < deadline:
                while True:
                    try:
                        kind, value = events.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "status":
                        session.on_status(*value)
                session.tick(time.monotonic())
                time.sleep(.01)
            if session.owned:
                notify("未获得双臂示教结束确认，请检查驱动和机器人状态")
                code = 1
        if executor is not None:
            executor.shutdown(timeout_sec=2.)
        if thread is not None:
            thread.join(3.)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if outputs is not None:
            try:
                outputs.close(timeout=3 * args.m050_timeout_ms / 1000. + 1.)
            except Exception as exc:
                notify(str(exc))
                code = 1
        if lock_file is not None:
            lock_file.close()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    return code


def self_test():
    from concurrent.futures import Future

    class Client:
        def __init__(self):
            self.futures = []
        def service_is_ready(self):
            return True
        def call_async(self, request):
            future = Future()
            self.futures.append(future)
            return future

    starts, stops = ({side: Client() for side in SIDES} for _ in range(2))
    session = DualTeaching(starts, stops, object, report=lambda _: None)
    for side in SIDES:
        session.on_status(side, "idle")
    assert session.start(0.)
    assert not session.start(.1)  # Keyboard repeat cannot restart/toggle a session.
    for side in SIDES:
        session.on_status(side, "teaching")
    session.tick(1.)
    assert session.phase == "teaching"
    session.stop()
    session.tick(2.)
    for side in SIDES:
        starts[side].futures[0].set_result(type("Result", (), {"success": True, "message": "stopped"})())
        stops[side].futures[0].set_result(object())
        session.on_status(side, "stopped")
    session.tick(3.)
    assert session.phase == "idle" and not session.owned
    decoder = KeyDecoder()
    assert decoder.feed("\x1b[A", 1.) == []
    assert decoder.feed("1", 2.) == ["1"]
    assert decoder.feed("1", 2.01) == []
    fake = SuctionOutputs(None, report=lambda _: None)
    fake.record_readback((False, True), 1.)
    fake.set("left", True, at=2.)
    assert fake.sample_at(1.5, 1.).actual == (False, True)
    assert fake.sample_at(2.1, 2.).target == (True, True)
    assert fake.sample_at(1.5, 1.).target == (False, True)  # Never use a future keypress.
    args = parse_args(["--fps", "5", "--width", "4", "--height", "3"])
    settings = dataset_schema(args)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepare_root(root, settings)
        writer = EpisodeWriter(root, "keyboard test", settings)
        writer.begin_segment()
        state = np.zeros(16, dtype=np.float32)
        action = state.copy()
        action[14] = 1.
        bundle = Bundle(10., state, action,
                        {key: np.zeros((3, 4, 3), dtype=np.uint8) for key in CAMERAS},
                        np.full(5, 10.), 9.9, np.asarray([9.95, 9.9]))
        writer.add(bundle)
        writer.end_segment("test")
        path = writer.save()
        _, arrays = validate_episode(path, settings)
        assert arrays["state"][0, 14] == 0 and arrays["action"][0, 14] == 1
    notify("键盘示教、双吸盘因果时间对齐、16维数据保存/校验自检通过；未连接硬件")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.validate_only or args.export_to is not None:
        try:
            if args.export_to is not None:
                count = export_dataset(args.root, args.export_to, args.repo_id)
                notify(f"导出 {count} 条连续 LeRobot episodes")
            else:
                settings = json.loads((args.root / "schema.json").read_text())
                validate_schema(settings)
                paths = sorted(args.root.glob("episode_*"))
                if not paths:
                    raise ValueError("没有已完成的 episode")
                for path in paths:
                    meta, arrays = validate_episode(path, settings)
                    notify(f"{path.name}: {meta['frames']} 帧，格式/时序校验通过")
                    for warning in episode_quality(meta, arrays)["warnings"]:
                        notify("[训练数据需复核] " + warning)
            return 0
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            notify(f"数据错误：{exc}")
            return 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
