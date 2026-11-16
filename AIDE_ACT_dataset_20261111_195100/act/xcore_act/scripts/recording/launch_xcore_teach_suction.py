#!/usr/bin/env python3
"""双臂示教服务、三路 30 Hz 相机与键盘吸盘数采的统一启动管理器。

运行同目录 start_xcore_teach_suction.sh --fps 20 --task "双臂吸盘抓取"。
原采集器参数原样传递；额外支持 --startup-timeout 秒（默认 60）、
--bringup-log-dir 目录、--restart-cameras（结束本仓库旧相机组，再启动三相机）。
网络腕相机：--wrist-camera-host 172.16.12.45 [--wrist-camera-port 9800]；
Jetson 仅启动头部相机，左右腕由电脑发送，默认增加0.20秒图像配对等待。
--cameras-only 仅启动和检查相机，不启动机械臂、吸盘或采集器。
--help、自检、数据校验与导出不会启动硬件。
服务/图像就绪后进入键盘界面；A 开示教，R 开录制，不自动调用开始服务。
Q/Ctrl+C 先等采集器完成示教退出和数据处理，再结束本次启动的后台进程。
已有完整服务/三相机会复用并保留；已有部分相机时报告冲突，不重复打开 USB。
新启动的相机使用 16 MiB Fast DDS 共享内存段，减少大图像传输丢包；
只作用于相机子进程，保留 UDP 网络通信和用户已有的 DDS 配置。
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import record_xcore_teach_suction as recorder

ROOT = recorder.ROOT
RECORDING = Path(__file__).resolve().parent
ARM_NODE = recorder.PREFIX + "/driver"
CAMERA_TOPICS = {key: f"/robot/sensors/cameras/{key}/color/image_raw"
                 for key in recorder.CAMERAS}
CAMERA_STABLE_SECONDS = 1.
CAMERA_DUPLICATE_GRACE_SECONDS = 3.
CAMERA_DDS_PROFILE = RECORDING / "camera_capture_fastdds.xml"
NETWORK_CAMERAS = RECORDING / "network_cameras"


def notify(message):
    print(message, flush=True)


def camera_environment():
    """Scope large-image SHM to cameras; never alter the arm driver's env."""
    env = os.environ.copy()
    if env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp") not in (
            "rmw_fastrtps_cpp", "rmw_fastrtps_dynamic_cpp"):
        return env
    if any(env.get(key) for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE")):
        notify("相机沿用已有 Fast DDS 配置")
        return env
    if not CAMERA_DDS_PROFILE.is_file():
        raise FileNotFoundError(str(CAMERA_DDS_PROFILE))
    env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(CAMERA_DDS_PROFILE)
    notify("相机使用 16 MiB 共享内存传输配置（仅相机进程）")
    return env


def camera_launches(root=ROOT, proc=Path("/proc")):
    """Recognize this workspace's camera launch parents, never arm processes."""
    configs = (Path(root) / "hardware/robot_vision_ws/src/robot_vision_bringup/config").resolve()
    network_dir = (Path(root) / "software/vla/act/xcore_act/scripts/recording/network_cameras").resolve()
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            raw = (entry / "cmdline").read_bytes()
            argv = [os.fsdecode(word) for word in raw.split(b"\0") if word]
            if (len(argv) >= 2 and Path(argv[0]).name.startswith("python") and
                    Path(argv[1]).resolve() == network_dir / "jetson_camera_bridge.py"):
                started = (entry / "stat").read_text().rsplit(")", 1)[1].split()[19]
                host = argv[argv.index("--host") + 1] if "--host" in argv else "172.16.12.45"
                port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 9800
                found.append(dict(pid=int(entry.name), config=str(network_dir / "jetson_camera_bridge.py"),
                                  started=started, kind="network", host=host, port=port))
                continue
            offset = 1 if argv and Path(argv[0]).name == "ros2" else 2
            if (offset == 2 and (len(argv) < 2 or not Path(argv[0]).name.startswith("python") or
                                 Path(argv[1]).name != "ros2")):
                continue
            if argv[offset:offset + 3] != ["launch", "robot_vision_bringup", "multi_realsense.launch.py"]:
                continue
            config_args = [word.split(":=", 1)[1] for word in argv
                           if word.startswith("camera_registry_file:=")]
            if not config_args:
                continue
            config = Path(config_args[-1]).resolve()
            if config.parent != configs and config != network_dir / "cameras_head_network.yaml":
                continue
            # Keep a start-time token to reject a reused PID before signalling.
            started = (entry / "stat").read_text().rsplit(")", 1)[1].split()[19]
            found.append(dict(pid=int(entry.name), config=str(config), started=started))
        except (OSError, ValueError, IndexError):
            continue
    return found


def camera_lock_busy():
    """Probe the same flock as both existing camera scripts; never unlink it."""
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    with (runtime / f"aide-head-camera-{os.getuid()}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


class Readiness:
    """Only discover services and receive status/images; never call a service."""
    def __init__(self, args, cameras_only=False):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.rclpy, self.args = rclpy, args
        self.context = Context()
        self.node = self.executor = None
        self.local_feedback = None
        try:
            if not cameras_only and getattr(args,"joint_feedback","ros") == "local":
                self.local_feedback = recorder.LocalFeedbackReceiver(lambda packet,received:None,
                    args.joint_feedback_port,max_age=args.state_max_age)
            rclpy.init(args=[], context=self.context, signal_handler_options=SignalHandlerOptions.NO)
            self.node = rclpy.create_node(f"keyboard_teach_startup_{os.getpid()}", context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.clients = [self.node.create_client(Trigger,
                            f"{recorder.PREFIX}/cartesian_drag_safe/{operation}/{side}_arm")
                            for operation in ("start", "stop") for side in recorder.SIDES] if not cameras_only else []
            self.status = dict.fromkeys(recorder.SIDES)
            self.frames = {key: deque(maxlen=100) for key in recorder.CAMERAS}
            self.shapes = {}
            self.last_camera_stamp = {}
            self.camera_duplicate_since = self.camera_ready_since = None
            self.subscriptions_kept = []
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            for side in (() if cameras_only else recorder.SIDES):
                self.subscriptions_kept.append(self.node.create_subscription(
                    String, f"{recorder.PREFIX}/cartesian_drag_safe/status/{side}_arm",
                    lambda msg, side=side: self.status.update({side: msg.data}), qos))
            for key, topic in CAMERA_TOPICS.items():
                self.subscriptions_kept.append(self.node.create_subscription(
                    Image, topic, lambda msg, key=key: self.image(key, msg),
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)))
        except BaseException:
            self.close()
            raise

    def image(self, key, message):
        if getattr(self.args, "network_camera_alignment", False):
            stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
            previous = self.last_camera_stamp.get(key, -math.inf)
            self.last_camera_stamp[key] = stamp
            now = self.node.get_clock().now().nanoseconds / 1e9
            if (not math.isfinite(stamp) or stamp <= previous or
                    not -.05 <= now-stamp <= self.args.max_age):
                self.frames[key].clear()
                self.camera_ready_since = None
                return
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(message.encoding, 0)
        if (channels and message.width > 0 and message.height > 0 and
                message.step >= message.width * channels and
                len(message.data) >= message.height * message.step):
            self.frames[key].append(time.monotonic())
            self.shapes[key] = (message.width, message.height, message.encoding)

    def spin(self, seconds=.1):
        self.executor.spin_once(timeout_sec=seconds)

    def arm_present(self):
        if any(client.service_is_ready() for client in self.clients):
            return True
        return any(namespace.rstrip("/") + "/" + name == ARM_NODE
                   for name, namespace in self.node.get_node_names_and_namespaces())

    def arms_ready(self):
        drivers = sum(namespace.rstrip("/") + "/" + name == ARM_NODE
                      for name, namespace in self.node.get_node_names_and_namespaces())
        if drivers > 1:
            raise RuntimeError("检测到多个双臂驱动，请保留一套驱动")
        if any(value and value.startswith("error:") for value in self.status.values()):
            raise RuntimeError(f"双臂示教清理未确认：{self.status}，请检查驱动状态")
        local_feedback = getattr(self,"local_feedback",None)
        if local_feedback is not None and local_feedback.error:
            raise RuntimeError("本机关节通道故障："+local_feedback.error)
        return (all(client.service_is_ready() for client in self.clients) and all(self.status.values())
                and (local_feedback is None or local_feedback.ready()))

    def camera_counts(self):
        # DDS discovery and RealSense stream reconfiguration can briefly expose
        # zero or overlapping endpoints. Report the raw graph for diagnostics
        # and camera shutdown; readiness below requires it to settle.
        return {key: self.node.count_publishers(topic) for key, topic in CAMERA_TOPICS.items()}

    def cameras_ready(self):
        now = time.monotonic()
        counts = self.camera_counts()
        if any(count > 1 for count in counts.values()):
            self.camera_ready_since = None
            if self.camera_duplicate_since is None:
                self.camera_duplicate_since = now
            if now - self.camera_duplicate_since >= CAMERA_DUPLICATE_GRACE_SECONDS:
                publishers = {key: [info.node_namespace.rstrip("/") + "/" + info.node_name
                                    for info in self.node.get_publishers_info_by_topic(CAMERA_TOPICS[key])]
                              for key, count in counts.items() if count > 1}
                raise RuntimeError(f"相机持续存在重复发布者（超过{CAMERA_DUPLICATE_GRACE_SECONDS:g}秒）："
                                   f"{counts}，节点={publishers}")
            return False
        self.camera_duplicate_since = None
        if not all(counts[key] == 1 and len(self.frames[key]) >= 2 and
                   now - self.frames[key][-1] < 2. and
                   self.shapes[key][:2] == (self.args.width, self.args.height) and
                   self.shapes[key][2] in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8")
                   for key in recorder.CAMERAS):
            self.camera_ready_since = None
            return False
        if getattr(self.args, "network_camera_alignment", False):
            for frames in self.frames.values():
                recent = [at for at in frames if now-at <= 2.]
                span = recent[-1]-recent[0] if len(recent)>1 else 0.
                if span < 1. or (len(recent)-1)/span < self.args.fps*.9:
                    self.camera_ready_since = None
                    return False
        if self.camera_ready_since is None:
            self.camera_ready_since = now
        return now - self.camera_ready_since >= CAMERA_STABLE_SECONDS

    def describe(self):
        return (f"双臂服务={[client.service_is_ready() for client in self.clients]}，"
                f"示教状态={self.status}，相机发布者={self.camera_counts()}，尺寸={self.shapes}，"
                f"本机关节={self.local_feedback.diagnostics() if self.local_feedback else 'ROS兼容模式'}")

    def close(self):
        if getattr(self,"local_feedback",None) is not None:
            self.local_feedback.close()
            self.local_feedback = None
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1.)
            self.executor = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.context.ok():
            self.context.shutdown()


class Supervisor:
    def __init__(self, args, log_dir, startup_timeout=60., root=ROOT, restart_cameras=False,
                 wrist_camera_host=None, wrist_camera_port=9800, cameras_only=False):
        self.args, self.log_dir = args, Path(log_dir)
        self.startup_timeout, self.root = startup_timeout, Path(root)
        self.children = []
        self.recording = None
        self.signal_requested = None
        self.restart_cameras = restart_cameras
        self.wrist_camera_host, self.wrist_camera_port = wrist_camera_host, wrist_camera_port
        self.cameras_only = cameras_only

    def interrupt(self, signum, _frame):
        self.signal_requested = signum

    def check(self):
        if self.signal_requested is not None:
            raise InterruptedError("收到退出信号")
        for name, child, path in self.children:
            code = child.poll()
            if code is not None:
                details = ""
                try:
                    with Path(path).open("rb") as log:
                        log.seek(0, os.SEEK_END)
                        log.seek(max(0, log.tell() - 4096))
                        details = log.read().decode(errors="replace").strip()
                except OSError:
                    pass
                raise RuntimeError(f"{name}进程退出（{code}），日志：{path}\n{details}")

    def start(self, name, command):
        path = self.log_dir / (name + ".log")
        environment = camera_environment() if name in ("cameras","wrist_cameras") else None
        if name == "arms" and getattr(self.args,"joint_feedback","ros") == "local":
            environment = dict(os.environ,AIDE_JOINT_FEEDBACK_PORT=str(self.args.joint_feedback_port))
        with path.open("wb") as output:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=environment)
        self.children.append((name, child, path))
        notify(f"启动{name}，日志：{path}")

    def wait_for(self, monitor, predicate, label):
        deadline = time.monotonic() + self.startup_timeout
        while True:
            self.check()
            if predicate():
                notify(label + "就绪")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待{label}超时。{monitor.describe()}。日志：{self.log_dir}")
            monitor.spin()

    def prepare_cameras(self, monitor):
        existing = camera_launches(self.root)
        counts = monitor.camera_counts()
        busy = camera_lock_busy()
        if self.restart_cameras and (existing or busy or any(counts.values())):
            if not existing:
                raise RuntimeError("相机被其他入口占用，未找到本仓库可重启的相机组；请先退出原相机程序")
            for process in existing:
                self.check()
                # Recheck identity immediately before sending SIGINT only to
                # the camera launch parent. ROS launch shuts down its children.
                if process not in camera_launches(self.root):
                    continue
                notify(f"结束旧相机组 PID={process['pid']}，配置={process['config']}")
                try:
                    os.kill(process["pid"], signal.SIGINT)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + min(20., self.startup_timeout)
            while camera_lock_busy() or camera_launches(self.root) or any(monitor.camera_counts().values()):
                self.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("旧相机组尚未完全退出或被后台程序重新启动，未打开第二套相机")
                monitor.spin()
            # Do not use frames received before the camera switch as readiness.
            for frames in monitor.frames.values():
                frames.clear()
            monitor.shapes.clear()
            return False
        if all(count == 1 for count in counts.values()):
            if self.wrist_camera_host and not any(p.get("kind") == "network" and
                    p.get("host") == self.wrist_camera_host and
                    p.get("port", 9800) == self.wrist_camera_port for p in existing):
                raise RuntimeError("现有腕相机不是指定电脑的网络入口；请增加 --restart-cameras 切换")
            if not self.wrist_camera_host and any(p.get("kind") == "network" for p in existing):
                raise RuntimeError("现有腕相机来自网络；请指定 --wrist-camera-host，或 --restart-cameras 切回USB模式")
            notify("复用现有三路相机，退出时保留")
            return True
        if busy or existing or any(counts.values()):
            owners = ", ".join(f"PID={p['pid']} ({Path(p['config']).name})" for p in existing) or "未知入口"
            raise RuntimeError(
                f"相机已被现有入口占用：{owners}，所需三路发布者={counts}。"
                "可在本命令增加 --restart-cameras，将旧相机组切换为三路采集相机")
        return False

    def prepare(self, monitor):
        # Discovery is allowed time before choosing whether to reuse a launch.
        deadline = time.monotonic() + 2.
        while time.monotonic() < deadline:
            self.check()
            monitor.spin()
        arm_script = self.root / "hardware/arm/xCoreROS/tools/start_upperlimb.sh"
        camera_script = self.root / "hardware/robot_vision_ws/scripts/start_cameras_30hz.sh"
        if self.wrist_camera_host:
            camera_script = self.root / "hardware/robot_vision_ws/scripts/start_cameras.sh"
        for path in ((camera_script,) if self.cameras_only else (arm_script, camera_script)):
            if not path.is_file():
                raise FileNotFoundError(str(path))
        # Detect/resolve camera conflicts before opening any new SDK connection.
        reuse_cameras = self.prepare_cameras(monitor)
        if self.cameras_only:
            notify("仅相机模式：不启动机械臂、吸盘和采集器")
        elif monitor.arm_present():
            notify("检测到现有双臂驱动/示教服务，等待就绪并复用；退出时保留")
        else:
            # Explicitly disable its single head-camera entry: the three-camera
            # entry owns that device and uses the recorder's canonical topics.
            self.start("arms", ["bash", str(arm_script), "start_head_camera:=false"])
        if not reuse_cameras:
            if self.wrist_camera_host:
                self.start("cameras", ["bash", str(camera_script),
                    "camera_registry_file:=" + str(NETWORK_CAMERAS / "cameras_head_network.yaml")])
                self.start("wrist_cameras", [sys.executable, "-u", str(NETWORK_CAMERAS / "jetson_camera_bridge.py"),
                    "--host", self.wrist_camera_host, "--port", str(self.wrist_camera_port),
                    "--width", str(self.args.width), "--height", str(self.args.height),
                    "--metrics-file", str(self.log_dir / "network_cameras.jsonl")])
            else:
                self.start("cameras", ["bash", str(camera_script)])
        if not self.cameras_only:
            self.wait_for(monitor, monitor.arms_ready, "双臂示教服务")
        self.wait_for(monitor, monitor.cameras_ready, "三路相机图像")

    def record(self, argv):
        self.check()
        # Inherit the foreground terminal process group so raw keyboard input
        # remains available. Background hardware groups never read the terminal.
        self.recording = subprocess.Popen(
            [sys.executable, "-u", str(RECORDING / "record_xcore_teach_suction.py"), *argv])
        notify("全部就绪：A 开启示教，R 开始录制；Q/Ctrl+C 退出")
        while self.recording.poll() is None:
            self.check()
            time.sleep(.1)
        return self.recording.returncode

    def finish_recorder(self):
        if self.recording is None or self.recording.poll() is not None:
            return
        notify("等待采集器结束示教并保留未确认数据……")
        # SIGTERM uses the recorder's same graceful Q/cleanup path. Leave the
        # driver alive until this path has finished (including the M050 timeout).
        self.recording.terminate()
        timeout = self.args.shutdown_timeout + 3 * self.args.m050_timeout_ms / 1000. + 15.
        try:
            self.recording.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            notify("采集器未在期限内退出；将终止进程，请检查驱动的示教结束状态")
            self.recording.kill()
            self.recording.wait(timeout=3.)

    def close(self):
        self.finish_recorder()
        # Stop arms before cameras, always after recorder cleanup. Only children
        # launched here are signalled; never stop a reused external service.
        for name, child, path in self.children:
            if child.poll() is not None:
                continue
            notify(f"结束本次启动的{name}")
            for signum, timeout in ((signal.SIGINT, 20.), (signal.SIGTERM, 5.), (signal.SIGKILL, 3.)):
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    break
                try:
                    child.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    notify(f"{name}仍在退出，日志：{path}")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--startup-timeout", type=float, default=60.)
    parser.add_argument("--bringup-log-dir", type=Path)
    parser.add_argument("--restart-cameras", action="store_true")
    parser.add_argument("--wrist-camera-host")
    parser.add_argument("--wrist-camera-port", type=int, default=9800)
    parser.add_argument("--cameras-only", "--camera-only", action="store_true")
    options, forwarded = parser.parse_known_args(argv)
    if not math.isfinite(options.startup_timeout) or options.startup_timeout <= 0:
        parser.error("--startup-timeout 必须是有限正数")
    if not 0 < options.wrist_camera_port < 65536:
        parser.error("--wrist-camera-port 必须在1..65535内")
    if options.wrist_camera_host and not any(arg.split("=", 1)[0] == "--camera-lookahead" for arg in forwarded):
        forwarded += ["--camera-lookahead", "0.20"]
    if "--help" in forwarded or "-h" in forwarded:
        notify(__doc__)
    args = recorder.parse_args(forwarded)
    args.network_camera_alignment = bool(options.wrist_camera_host)
    if args.self_test or args.validate_only or args.export_to is not None:
        return recorder.main(forwarded)
    if not options.cameras_only and not args.m050_library.is_file():
        parser.error(f"缺少吸盘 SDK：{args.m050_library}，请先按 USR-M050/README.md 构建")
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    with (runtime / f"aide-keyboard-teach-{os.getuid()}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            notify("键盘示教一键启动程序已运行，请勿重复启动")
            return 1
        log_dir = options.bringup_log_dir
        if log_dir is None:
            log_dir = Path(tempfile.mkdtemp(prefix="aide-teach-suction-"))
        log_dir.mkdir(parents=True, exist_ok=True)
        supervisor = Supervisor(args, log_dir, options.startup_timeout,
                                restart_cameras=options.restart_cameras,
                                wrist_camera_host=options.wrist_camera_host,
                                wrist_camera_port=options.wrist_camera_port, cameras_only=options.cameras_only)
        monitor = None
        handlers = {sig: signal.signal(sig, supervisor.interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            monitor = Readiness(args, cameras_only=True) if options.cameras_only else Readiness(args)
            supervisor.prepare(monitor)
            if options.cameras_only:
                notify("三路相机检查运行中；Ctrl+C 退出；未启动机械臂和吸盘")
                while True:
                    supervisor.check()
                    monitor.spin()
            monitor.close()
            monitor = None
            return supervisor.record(forwarded)
        except InterruptedError:
            return 130
        except Exception as exc:
            notify(f"统一启动失败：{exc}")
            return 1
        finally:
            if monitor is not None:
                monitor.close()
            try:
                supervisor.close()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
