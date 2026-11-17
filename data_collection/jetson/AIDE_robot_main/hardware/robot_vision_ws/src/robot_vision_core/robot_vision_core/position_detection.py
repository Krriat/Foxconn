import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# 官方标准检测结果消息
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesis, BoundingBox3D
from vision_msgs.msg import ObjectHypothesisWithPose
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


class YoloDepthDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detector', namespace='/robot/vision')

        # ================= ROS2 参数 =================
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 10),
                ('enable_gui', False),
                # 彩色图像话题
                ('image_topic', '/robot/sensors/cameras/head/color/image_raw'),
                # 深度图像话题
                ('depth_topic', '/robot/sensors/cameras/head/aligned_depth_to_color/image_raw'),
                # 深度相机内参话题
                ('depth_camera_info_topic', '/robot/sensors/cameras/head/aligned_depth_to_color/camera_info'),
                # 检测结果发布话题（留空则按 image_topic 自动推导为 /robot/vision/<cam_name>/detection）
                ('detection_topic', ''),
                # 检测结果输出坐标系的 frame_id（已用 T_cam2base 把目标点变换到此系）
                ('output_frame_id', 'base_link'),
                # 模型路径
                ('model_path', str(Path(get_package_share_directory('robot_vision_core')) / 'models/best_b.pt')),
                # 置信度阈值
                ('confidence_threshold', 0.5),
                # 相机到机器人基坐标系的外参（4x4齐次矩阵，行优先）
                # 需要根据你的实际手眼标定结果修改
                ('T_cam2base', [
                    -0.094428, -0.741979,  0.66374,   0.072383,
                    -0.995499,  0.075759, -0.056937,  0.054335,
                    -0.008039, -0.666129, -0.745793,  0.130817,
                    0.0,0.0,0.0,1.0
                ]),
            ]
        )

        # 读取参数
        self.publish_rate = self.get_parameter('publish_rate').value
        self.enable_gui = self.get_parameter('enable_gui').value
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_camera_info_topic = self.get_parameter('depth_camera_info_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.output_frame_id = self.get_parameter('output_frame_id').value

        # 推导相机名（哪个相机图像做检测就用哪个），并据此确定检测结果话题
        self.camera_name = self._derive_camera_name(self.image_topic)
        detection_topic = self.get_parameter('detection_topic').value
        self.detection_topic = detection_topic if detection_topic else f"/robot/vision/{self.camera_name}/detection"

        # 解析外参矩阵
        T_list = self.get_parameter('T_cam2base').value
        self.T_cam2base = np.array(T_list, dtype=np.float64).reshape(4, 4)

        # ================= YOLOv8 模型加载 =================
        self.get_logger().info(f"正在加载 YOLOv8 模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info("YOLOv8 模型加载成功！")

        # ================= CV Bridge & 数据缓存 =================
        self.bridge = CvBridge()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.latest_image_header = None  # 彩色图像 header，用作检测输出时间戳
        self.camera_matrix = None  # 相机内参矩阵 K
        self.dist_coeffs = None    # 畸变系数 D

        # ================= 订阅话题 =================
        # 订阅彩色图像
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        # 订阅深度图像
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, 10
        )
        # 订阅相机内参
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.depth_camera_info_topic, self.camera_info_callback, 10
        )

        # ================= 发布话题 =================
        # 官方标准格式：vision_msgs/Detection3DArray
        # 话题名随相机变：/robot/vision/<cam_name>/detection
        self.detection_pub = self.create_publisher(
            Detection3DArray, self.detection_topic, 10
        )
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info(
            f"YOLOv8 深度检测节点已启动！相机: {self.camera_name} | "
            f"检测输出话题: {self.detection_topic}"
        )

    @staticmethod
    def _derive_camera_name(image_topic: str) -> str:
        """
        从 image_topic 推导相机名。
        约定路径形如 /robot/sensors/cameras/<cam_name>/color/image_raw，
        取 'cameras' 后第一个段作为相机名；无法解析时返回 'camera'。
        """
        parts = [p for p in image_topic.split('/') if p]
        if 'cameras' in parts:
            idx = parts.index('cameras')
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return 'camera'

    def image_callback(self, msg: Image):
        """缓存最新彩色图像"""
        try:
            self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"彩色图像转换失败: {e}")

    def depth_callback(self, msg: Image):
        """缓存最新深度图像（单位：毫米）"""
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"深度图像转换失败: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """获取相机内参"""
        # 从 CameraInfo 中提取 3x3 内参矩阵
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def pixel_to_camera_3d(self, u, v, depth_mm):
        """
        将像素坐标 (u, v) + 深度值 转换为相机坐标系下的三维坐标
        """
        if self.camera_matrix is None:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # 深度图单位通常是毫米，转换为米
        Z = depth_mm / 1000.0

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return np.array([X, Y, Z])

    def camera_to_robot(self, point_cam):
        """
        将相机坐标系下的点变换到机器人基坐标系
        P_robot = T_cam2base * P_camera
        """
        point_cam_homo = np.append(point_cam, 1.0)
        point_robot_homo = self.T_cam2base @ point_cam_homo
        return point_robot_homo[:3]

    def get_mask_center(self, mask):
        """计算掩码的几何中心（像素坐标）"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        moments = cv2.moments(mask_uint8)

        if moments['m00'] == 0:
            return None

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy

    def get_median_depth(self, mask):
        """取掩码区域内深度值的中位数"""
        mask_bool = mask.astype(bool)
        depths = self.latest_depth_img[mask_bool]

        valid_depths = depths[(depths > 0) & (depths < 10000)]
        if len(valid_depths) == 0:
            return None

        return float(np.median(valid_depths))

    def timer_callback(self):
        """主循环：检测 + 坐标计算 + 发布 vision_msgs/Detection3DArray"""
        if self.latest_color_img is None or self.latest_depth_img is None:
            return
        if self.camera_matrix is None:
            self.get_logger().warn("等待相机内参...")
            return

        try:
            color_img = self.latest_color_img.copy()
            display_img = color_img.copy()

            # --- YOLOv8 分割推理 ---
            results = self.model(color_img, verbose=False)
            result = results[0]

            if result.masks is None:
                return

            masks = result.masks.data.cpu().numpy()       # (N, H, W)
            boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
            clss = result.boxes.cls.cpu().numpy()         # (N,)
            confs = result.boxes.conf.cpu().numpy()       # (N,)

            # 组装 Detection3DArray
            det_array_msg = Detection3DArray()
            # 时间戳对齐到相机采集时刻
            if self.latest_image_header is not None:
                det_array_msg.header = self.latest_image_header
            else:
                det_array_msg.header.stamp = self.get_clock().now().to_msg()
            # 目标坐标已变换到此系
            det_array_msg.header.frame_id = self.output_frame_id

            for i in range(len(masks)):
                if confs[i] < self.conf_threshold:
                    continue

                mask = masks[i]

                # 1. 掩码中心点（像素坐标）
                center = self.get_mask_center(mask)
                if center is None:
                    continue
                u, v = center

                # 2. 掩码区域深度（中位数，毫米）
                depth_mm = self.get_median_depth(mask)
                if depth_mm is None:
                    continue

                # 3. 像素 + 深度 → 相机坐标系 3D
                point_cam = self.pixel_to_camera_3d(u, v, depth_mm)
                if point_cam is None:
                    continue

                # 4. 相机坐标系 → 机器人基坐标系
                point_robot = self.camera_to_robot(point_cam)

                # 5. 组装 Detection3D
                det_msg = Detection3D()
                det_msg.header = det_array_msg.header

                # 类别 ID + 置信度
                # hyp = ObjectHypothesis()
                # hyp.class_id = str(clss[i])
                # hyp.score = float(confs[i])
                # det_msg.results.append(hyp)

                ohwp = ObjectHypothesisWithPose()
                ohwp.hypothesis.class_id = str(clss[i])
                ohwp.hypothesis.score = float(confs[i])
                # ohwp.pose 可以保持默认，因为你的检测没有提供目标的姿态信息
                det_msg.results.append(ohwp)

                # 用 BoundingBox3D.center 承载目标在 output_frame_id 下的 3D 位置
                # YOLOv8-seg 这里只输出掩码中心点 3D 坐标，无真实尺寸/姿态
                bbox = BoundingBox3D()
                bbox.center.position.x = float(point_robot[0])
                bbox.center.position.y = float(point_robot[1])
                bbox.center.position.z = float(point_robot[2])
                bbox.center.orientation.w = 1.0  # 单位四元数，无姿态信息
                bbox.size.x = 0.0
                bbox.size.y = 0.0
                bbox.size.z = 0.0
                det_msg.bbox = bbox

                det_array_msg.detections.append(det_msg)

                # 日志
                self.get_logger().info(
                    f"检测目标: {result.names[int(clss[i])]} | "
                    f"置信度: {confs[i]:.2f} | "
                    f"像素: ({u}, {v}) | 深度: {depth_mm:.0f}mm | "
                    f"相机坐标: [{point_cam[0]:.3f}, {point_cam[1]:.3f}, {point_cam[2]:.3f}] | "
                    f"机器人坐标: [{point_robot[0]:.3f}, {point_robot[1]:.3f}, {point_robot[2]:.3f}]"
                )

                # --- 可视化 ---
                color = tuple(np.random.randint(0, 255, 3).tolist())

                mask_resized = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_img, contours, -1, color, 2)

                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                cv2.circle(display_img, (u, v), 5, (0, 0, 255), -1)

                label = f"{result.names[int(clss[i])]} {confs[i]:.2f}"
                coord_label = f"R: [{point_robot[0]:.3f}, {point_robot[1]:.3f}, {point_robot[2]:.3f}]"
                cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display_img, coord_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 有检测结果时发布（避免空消息刷屏）
            if len(det_array_msg.detections) > 0:
                self.detection_pub.publish(det_array_msg)

            # 显示图像
            if self.enable_gui:
                cv2.imshow("YOLOv8 Depth Detection", display_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"处理帧时出错: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
