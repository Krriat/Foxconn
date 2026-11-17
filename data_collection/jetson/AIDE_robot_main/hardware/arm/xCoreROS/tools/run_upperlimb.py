"""Supervise the arm launch and head RGB-D publisher; never issue motion requests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CAMERA_ROOT = '/robot_02/sensors/cameras/head/'
IMAGE_TOPICS = ('color/image_raw', 'depth/image_rect_raw', 'aligned_depth_to_color/image_raw')


def head_camera_processes(script):
    spec = importlib.util.spec_from_file_location('head_camera_guard', script.parent / 'head_camera_guard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.running_head_cameras()


def wait_for_camera(check_cancelled, process=None, timeout_s=20.):
    # The head publisher uses Fast DDS even if an arm caller selected another RMW.
    previous = os.environ.get('RMW_IMPLEMENTATION')
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    context = node = executor = None
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image
        context = Context()
        rclpy.init(args=[], context=context, signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node('upperlimb_head_camera_check_' + str(os.getpid()), context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        counts = dict.fromkeys(IMAGE_TOPICS, 0)
        shapes = {}
        subscriptions = []
        for topic in IMAGE_TOPICS:
            def receive(message, topic=topic):
                if message.width > 0 and message.height > 0 and message.data:
                    counts[topic] += 1
                    shapes[topic] = (message.width, message.height, message.encoding)
            subscriptions.append(node.create_subscription(Image, CAMERA_ROOT + topic, receive, qos_profile_sensor_data))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            check_cancelled()
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'头部相机启动进程已退出，code={process.returncode}')
            executor.spin_once(timeout_sec=.1)
            publishers = [len(node.get_publishers_info_by_topic(CAMERA_ROOT + topic)) for topic in IMAGE_TOPICS]
            if any(count > 1 for count in publishers):
                raise RuntimeError('头部相机图像话题存在多个发布者，请保留一个相机实例')
            if min(counts.values()) >= 2 and publishers == [1, 1, 1]:
                if shapes[IMAGE_TOPICS[2]][:2] != shapes[IMAGE_TOPICS[0]][:2]:
                    raise RuntimeError('对齐深度尺寸与彩色图不一致')
                if any(shapes[topic][2] != '16UC1' for topic in IMAGE_TOPICS[1:]):
                    raise RuntimeError('头部深度图编码应为 16UC1')
                print('[HEAD_CAMERA_READY] ' + json.dumps(shapes), flush=True)
                return
        missing = [CAMERA_ROOT + topic for topic, count in counts.items() if count < 2]
        raise TimeoutError('未收到头部彩色/深度/对齐深度有效帧: ' + ', '.join(missing))
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=1.)
        if node is not None:
            node.destroy_node()
        if context is not None and context.ok():
            context.shutdown()
        if previous is None:
            os.environ.pop('RMW_IMPLEMENTATION', None)
        else:
            os.environ['RMW_IMPLEMENTATION'] = previous


def stop_process_group(process):
    if process.poll() is not None:
        return
    for signum, timeout_s in ((signal.SIGINT, 8.), (signal.SIGTERM, 3.), (signal.SIGKILL, 3.)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass


def run(arm_command, camera_script=None):
    stopped = False
    children = []
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    def check_cancelled():
        if stopped:
            raise InterruptedError('收到停止信号')
    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if camera_script is not None:
            camera_script = Path(camera_script).resolve()
            if not camera_script.is_file() or not (camera_script.parent.parent / 'install/local_setup.bash').is_file():
                raise RuntimeError('头部相机工作区未构建或启动脚本不存在: ' + str(camera_script))
            existing = head_camera_processes(camera_script)
            if len(existing) > 1:
                raise RuntimeError(f'头部相机已有多个实例 (PID: {existing})，请保留一个')
            if existing:
                print(f'[HEAD_CAMERA_REUSE] PID: {existing}', flush=True)
                wait_for_camera(check_cancelled)
            else:
                camera = subprocess.Popen(['bash', str(camera_script)], start_new_session=True)
                children.append(camera)
                wait_for_camera(check_cancelled, camera)
        check_cancelled()
        arm = subprocess.Popen(arm_command, start_new_session=True)
        children.append(arm)
        while True:
            check_cancelled()
            for process in children:
                code = process.poll()
                if code is not None:
                    # Losing an owned camera terminates the unified launch too.
                    return code if process is arm and code >= 0 else (code if code > 0 else 1)
            time.sleep(.1)
    except InterruptedError:
        return 130
    except Exception as error:
        print('统一启动失败: ' + str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        for process in reversed(children):
            stop_process_group(process)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head-camera-script', type=Path)
    parser.add_argument('arm_command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.arm_command[1:] if args.arm_command[:1] == ['--'] else args.arm_command
    if not command:
        parser.error('缺少机械臂 launch 命令')
    return run(command, args.head_camera_script)


if __name__ == '__main__':
    sys.exit(main())
