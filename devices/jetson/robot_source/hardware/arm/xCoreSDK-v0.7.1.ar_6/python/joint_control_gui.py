# -*- coding: utf-8 -*-
"""PyQt interface for controlling seven robot joints with MoveAbsJ."""

from __future__ import annotations

import math
import os
import platform
import sys
import traceback

import setup_path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if platform.system() == "Windows":
    from Release.windows import xCoreSDK_python
elif platform.system() == "Linux":
    arch = platform.machine()
    if arch == "aarch64":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        arm_lib_path = os.path.join(root_dir, "Release", "linux", "arm")
        sys.path.insert(0, arm_lib_path)
        import xCoreSDK_python
    elif arch == "x86_64":
        from Release.linux import xCoreSDK_python
    else:
        raise ImportError(f"不支持的Linux架构: {arch}")
else:
    raise ImportError("Unsupported operating system")


def ec_message(name: str, ec: dict) -> str:
    if ec.get("ec", 0) != 0:
        return f"{name}: {ec.get('message', ec)}"
    return f"{name}: {ec.get('message', '操作成功完成')}"


def ec_failed(ec: dict) -> bool:
    return ec.get("ec", 0) != 0


SLIDER_SCALE = 10000
JOINT_MIN_RAD = -10.0
JOINT_MAX_RAD = 10.0


class JointControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("7 Axis Joint Control - MoveAbsJ")
        self.robot = None
        self.current_joints = [0.0] * 7
        self.has_valid_joints = False
        self.target_spins: list[QDoubleSpinBox] = []
        self.target_sliders: list[QSlider] = []
        self.current_labels: list[QLabel] = []
        self.deg_labels: list[QLabel] = []

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(800)
        self.status_timer.timeout.connect(self.refresh_operation_state)

        self.build_ui()
        self.set_connected(False)

    def build_ui(self):
        root = QWidget()
        main = QVBoxLayout(root)

        connection_group = QGroupBox("Connection")
        connection_form = QFormLayout(connection_group)
        self.remote_ip_edit = QLineEdit("192.168.2.160")
        self.local_ip_edit = QLineEdit("192.168.2.10")
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(lambda: self.connect_robot(auto_read=True))
        self.read_button = QPushButton("Read Current / Auto Connect")
        self.read_button.clicked.connect(self.read_current_joints)
        self.set_target_button = QPushButton("Target = Current")
        self.set_target_button.clicked.connect(self.copy_current_to_target)

        connection_form.addRow("Robot IP", self.remote_ip_edit)
        connection_form.addRow("Local IP", self.local_ip_edit)
        connection_buttons = QHBoxLayout()
        connection_buttons.addWidget(self.connect_button)
        connection_buttons.addWidget(self.read_button)
        connection_buttons.addWidget(self.set_target_button)
        connection_form.addRow(connection_buttons)
        main.addWidget(connection_group)

        motion_group = QGroupBox("MoveAbsJ")
        motion_layout = QVBoxLayout(motion_group)
        motion_form = QFormLayout()
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 4000.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setValue(3000.0)
        self.zone_spin = QDoubleSpinBox()
        self.zone_spin.setRange(0.0, 100.0)
        self.zone_spin.setDecimals(1)
        self.zone_spin.setValue(10.0)
        self.prepare_checkbox = QCheckBox("Auto power on and set NRT mode before move")
        self.prepare_checkbox.setChecked(True)
        motion_form.addRow("Speed", self.speed_spin)
        motion_form.addRow("Zone", self.zone_spin)
        motion_form.addRow(self.prepare_checkbox)
        motion_layout.addLayout(motion_form)

        grid = QGridLayout()
        headers = ["Joint", "Current rad", "Target rad", "Target deg", "Slider"]
        for col, text in enumerate(headers):
            grid.addWidget(QLabel(text), 0, col)

        for idx in range(7):
            row = idx + 1
            grid.addWidget(QLabel(f"J{idx + 1}"), row, 0)
            cur_label = QLabel("--")
            self.current_labels.append(cur_label)
            grid.addWidget(cur_label, row, 1)

            spin = QDoubleSpinBox()
            spin.setRange(JOINT_MIN_RAD, JOINT_MAX_RAD)
            spin.setDecimals(6)
            spin.setSingleStep(0.01)
            spin.valueChanged.connect(lambda value, i=idx: self.on_spin_changed(i, value))
            self.target_spins.append(spin)
            grid.addWidget(spin, row, 2)

            deg_label = QLabel("0.000")
            self.deg_labels.append(deg_label)
            grid.addWidget(deg_label, row, 3)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(
                self.rad_to_slider(JOINT_MIN_RAD),
                self.rad_to_slider(JOINT_MAX_RAD),
            )
            slider.setSingleStep(self.rad_to_slider(0.001))
            slider.setPageStep(self.rad_to_slider(0.05))
            slider.valueChanged.connect(lambda value, i=idx: self.on_slider_changed(i, value))
            self.target_sliders.append(slider)
            grid.addWidget(slider, row, 4)

        motion_layout.addLayout(grid)

        action_buttons = QHBoxLayout()
        self.send_button = QPushButton("确定移动")
        self.send_button.clicked.connect(self.send_target)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_robot)
        self.power_off_button = QPushButton("Power Off")
        self.power_off_button.clicked.connect(self.power_off_robot)
        action_buttons.addWidget(self.send_button)
        action_buttons.addWidget(self.stop_button)
        action_buttons.addWidget(self.power_off_button)
        motion_layout.addLayout(action_buttons)
        main.addWidget(motion_group)

        self.operation_label = QLabel("Operation state: -")
        main.addWidget(self.operation_label)
        self.pose_label = QLabel("Pose: -")
        main.addWidget(self.pose_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        main.addWidget(self.log)

        self.setCentralWidget(root)

    def set_connected(self, connected: bool):
        self.read_button.setEnabled(True)
        self.set_target_button.setEnabled(connected)
        self.send_button.setEnabled(connected and self.has_valid_joints)
        self.stop_button.setEnabled(connected)
        self.power_off_button.setEnabled(connected)
        for spin in self.target_spins:
            spin.setEnabled(connected)
        for slider in self.target_sliders:
            slider.setEnabled(connected)

    def append_log(self, message: str):
        self.log.append(message)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def show_error(self, title: str, message: str):
        self.append_log(f"{title}: {message}")
        QMessageBox.warning(self, title, message)

    def mark_joint_read_invalid(self):
        self.has_valid_joints = False
        for label in self.current_labels:
            label.setText("ERR")
        self.send_button.setEnabled(False)

    def connect_robot(self, auto_read=True):
        try:
            remote_ip = self.remote_ip_edit.text().strip()
            local_ip = self.local_ip_edit.text().strip()
            self.append_log(f"Connecting: remote={remote_ip}, local={local_ip}")
            QApplication.processEvents()
            self.robot = xCoreSDK_python.ArRobot(remote_ip, local_ip)
            self.set_connected(True)
            self.status_timer.start()
            self.append_log(f"Robot object created: remote={remote_ip}, local={local_ip}")
            if auto_read:
                self.read_current_joints()
        except Exception as exc:
            self.robot = None
            self.set_connected(False)
            self.mark_joint_read_invalid()
            self.show_error("Connect failed", str(exc))
            self.append_log(traceback.format_exc())

    def require_robot(self):
        if self.robot is None:
            raise RuntimeError("Robot is not connected")
        return self.robot

    def read_current_joints(self):
        try:
            if self.robot is None:
                self.append_log("Robot is not connected; connecting before read...")
                self.connect_robot(auto_read=False)
                if self.robot is None:
                    return
            robot = self.require_robot()
            self.append_log("Reading current joint positions...")
            QApplication.processEvents()
            ec = {}
            raw_joints = list(robot.jointPos(ec))
            self.append_log(ec_message("jointPos", ec))
            if ec_failed(ec):
                self.mark_joint_read_invalid()
                self.show_error("Read failed", ec.get("message", str(ec)))
                return
            self.append_log("Raw jointPos: " + ", ".join(f"{v:.9f}" for v in raw_joints))
            if len(raw_joints) < 7:
                self.mark_joint_read_invalid()
                self.show_error("Read failed", f"jointPos returned {len(raw_joints)} values, expected 7")
                return
            joints = raw_joints[:7]
            self.current_joints = joints
            self.has_valid_joints = True
            for idx, value in enumerate(joints):
                self.current_labels[idx].setText(f"{value:.6f}")
            self.copy_current_to_target()
            self.send_button.setEnabled(True)
            self.append_log("Current joints loaded into target fields")
            self.read_cart_pose(robot)
        except Exception as exc:
            self.append_log(f"Read failed: {exc}")
            self.append_log(traceback.format_exc())

    @staticmethod
    def format_values(values, decimals=9) -> str:
        return ", ".join(f"{float(value):.{decimals}f}" for value in values)

    def read_cart_pose(self, robot):
        self.append_log("Reading cartesian pose...")

        posture_ec = {}
        pos = robot.posture(xCoreSDK_python.CoordinateType.endInRef, posture_ec)
        self.append_log(ec_message("posture", posture_ec))
        if ec_failed(posture_ec):
            self.append_log(f"Read posture failed: {posture_ec}")
            return

        pos_values = list(pos)
        trans = pos_values[:3]
        rpy = pos_values[3:]
        self.append_log("pos trans_m: " + self.format_values(trans))
        self.append_log("pos rpy_rad: " + self.format_values(rpy))
        self.pose_label.setText(
            "Pose trans_m: "
            + self.format_values(trans, 6)
            + "    rpy_rad: "
            + self.format_values(rpy, 6)
        )

        cart_ec = {}
        cart_posture = robot.cartPosture(xCoreSDK_python.CoordinateType.endInRef, cart_ec)
        self.append_log(ec_message("cartPosture", cart_ec))
        if ec_failed(cart_ec):
            self.append_log(f"Read cartPosture failed: {cart_ec}")
            return

        self.append_log(f"elbow_rad: {cart_posture.elbow}")
        self.append_log(f"hasElbow: {cart_posture.hasElbow}")
        self.append_log("confData: " + ",".join(map(str, cart_posture.confData)))
        self.append_log("external: " + self.format_values(cart_posture.external))
        self.append_log("cartPosture trans_m: " + self.format_values(cart_posture.trans))
        self.append_log("cartPosture rpy_rad: " + self.format_values(cart_posture.rpy))

    def copy_current_to_target(self):
        if not self.has_valid_joints:
            self.show_error("No valid joint data", "请先成功读取当前7个关节角。")
            return
        for idx, value in enumerate(self.current_joints):
            self.set_target_value(idx, value)

    @staticmethod
    def rad_to_slider(rad: float) -> int:
        return int(round(rad * SLIDER_SCALE))

    @staticmethod
    def slider_to_rad(value: int) -> float:
        return value / SLIDER_SCALE

    def set_target_value(self, idx: int, rad: float):
        rad = max(JOINT_MIN_RAD, min(JOINT_MAX_RAD, rad))
        spin = self.target_spins[idx]
        slider = self.target_sliders[idx]
        spin.blockSignals(True)
        slider.blockSignals(True)
        spin.setValue(rad)
        slider.setValue(self.rad_to_slider(rad))
        spin.blockSignals(False)
        slider.blockSignals(False)
        self.update_deg_label(idx)

    def on_spin_changed(self, idx: int, value: float):
        slider = self.target_sliders[idx]
        slider.blockSignals(True)
        slider.setValue(self.rad_to_slider(value))
        slider.blockSignals(False)
        self.update_deg_label(idx)

    def on_slider_changed(self, idx: int, value: int):
        rad = self.slider_to_rad(value)
        spin = self.target_spins[idx]
        spin.blockSignals(True)
        spin.setValue(rad)
        spin.blockSignals(False)
        self.update_deg_label(idx)

    def update_deg_label(self, idx: int):
        rad = self.target_spins[idx].value()
        self.deg_labels[idx].setText(f"{math.degrees(rad):.3f}")

    def target_joints(self) -> list[float]:
        return [spin.value() for spin in self.target_spins]

    def prepare_robot(self) -> bool:
        if not self.prepare_checkbox.isChecked():
            return True
        robot = self.require_robot()
        for name, call in [
            ("setOperateMode", lambda ec: robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)),
            ("setPowerState", lambda ec: robot.setPowerState(True, ec)),
            ("setMotionControlMode", lambda ec: robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)),
        ]:
            ec = {}
            call(ec)
            self.append_log(ec_message(name, ec))
            if ec_failed(ec):
                return False
        return True

    def send_target(self):
        try:
            robot = self.require_robot()
            if not self.prepare_robot():
                return
            target = self.target_joints()
            speed = self.speed_spin.value()
            zone = self.zone_spin.value()

            reset_ec = {}
            robot.moveReset(reset_ec)
            self.append_log(ec_message("moveReset", reset_ec))
            if ec_failed(reset_ec):
                return

            cmd = xCoreSDK_python.MoveAbsJCommand(target, speed, zone)
            cmd_id = xCoreSDK_python.PyString()
            append_ec = {}
            robot.moveAppend([cmd], cmd_id, append_ec)
            self.append_log(f"Command ID: {cmd_id.content()}")
            self.append_log(ec_message("moveAppend", append_ec))
            if ec_failed(append_ec):
                return

            start_ec = {}
            robot.moveStart(start_ec)
            self.append_log(ec_message("moveStart", start_ec))
            self.append_log("Target rad: " + ", ".join(f"{v:.6f}" for v in target))
        except Exception as exc:
            self.append_log(f"Move failed: {exc}")
            self.append_log(traceback.format_exc())

    def stop_robot(self):
        try:
            robot = self.require_robot()
            ec = {}
            robot.stop(ec)
            self.append_log(ec_message("stop", ec))
        except Exception as exc:
            self.append_log(f"Stop failed: {exc}")
            self.append_log(traceback.format_exc())

    def power_off_robot(self):
        try:
            robot = self.require_robot()
            ec = {}
            robot.setPowerState(False, ec)
            self.append_log(ec_message("setPowerState(False)", ec))
        except Exception as exc:
            self.append_log(f"Power off failed: {exc}")
            self.append_log(traceback.format_exc())

    def refresh_operation_state(self):
        if self.robot is None:
            return
        try:
            ec = {}
            state = self.robot.operationState(ec)
            if ec_failed(ec):
                self.operation_label.setText(f"Operation state: error {ec}")
            else:
                self.operation_label.setText(f"Operation state: {state}")
        except Exception:
            self.operation_label.setText("Operation state: unavailable")


def main():
    app = QApplication(sys.argv)
    window = JointControlWindow()
    window.resize(900, 650)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
