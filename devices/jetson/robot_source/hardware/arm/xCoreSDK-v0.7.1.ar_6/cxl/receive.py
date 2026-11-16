import socket
import struct
import math


PACKET_FORMAT = "<B9fB"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

def normalize_angle(angle):
    """将角度归一化到0-360°范围"""
    angle = angle % 360  # 取模运算处理超过360的情况
    if angle < 0:
        angle += 360  # 处理负数情况
    return angle

def quaternion_to_euler(x, y, z, w):
    """将四元数转换为欧拉角（调整轴对应和符号以匹配实际需求）"""
    # 计算绕X轴旋转角度 (roll)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # 计算绕Y轴旋转角度 (pitch)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # 处理边界情况
    else:
        pitch = math.asin(sinp)
    
    # 计算绕Z轴旋转角度 (yaw)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # 转换为度并处理轴对应关系和符号
    roll_deg = normalize_angle(math.degrees(roll))  # 原X轴
    pitch_deg = normalize_angle(math.degrees(pitch))  # 原Y轴
    yaw_deg = normalize_angle(-math.degrees(yaw))  # 原Z轴取负（反转方向）
    
    # 交换X和Y轴的输出（根据你的描述：原X实际是Y，原Y实际是X）
    return pitch_deg, roll_deg - 45, yaw_deg - 90  # 输出顺序改为 (Y, X, -Z)


def unpack_packet(data):
    """按 Unity 发送格式解包: hand(1B) + 9个float + buttons(1B)"""
    if len(data) != PACKET_SIZE:
        raise ValueError(f"数据包长度错误: 期望 {PACKET_SIZE} 字节, 实际 {len(data)} 字节")

    (
        hand_id,
        px,
        py,
        pz,
        qx,
        qy,
        qz,
        qw,
        jx,
        jy,
        buttons,
    ) = struct.unpack(PACKET_FORMAT, data)

    hand_name = "左手" if hand_id == 0 else "右手" if hand_id == 1 else f"未知手({hand_id})"

    return {
        "hand": hand_name,
        "position": (px, py, pz),
        "quaternion": (qx, qy, qz, qw),
        "joystick": (jx, jy),
        "trigger": bool(buttons & (1 << 0)),
        "grip": bool(buttons & (1 << 1)),
        "down": bool(buttons & (1 << 2)),
        "up": bool(buttons & (1 << 3)),
    }


def format_packet(packet):
    """格式化二进制包为可读文本"""
    px, py, pz = packet["position"]
    qx, qy, qz, qw = packet["quaternion"]
    jx, jy = packet["joystick"]

    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

    return (
        f"{packet['hand']}数据:\n"
        f"  位置: {px:.2f}, {py:.2f}, {pz:.2f}\n"
        f"  旋转角度(X,Y,Z): {roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°\n"
        f"  扳机键: {'按下' if packet['trigger'] else '释放'}\n"
        f"  Grip键: {'按下' if packet['grip'] else '释放'}\n"
        f"  Down键: {'按下' if packet['down'] else '释放'}\n"
        f"  Up键: {'按下' if packet['up'] else '释放'}\n"
        f"  摇杆: X={jx:.3f}, Y={jy:.3f}"
    )


def receive_udp_data(ip, port):
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # 绑定到指定的IP和端口
        sock.bind((ip, port))
        print(f"正在监听 UDP 数据 on {ip}:{port}...")
        
        while True:
            # 接收单个UDP数据包
            data, addr = sock.recvfrom(1024)

            # 按二进制协议解包，异常包仅警告并继续
            try:
                packet = unpack_packet(data)
            except ValueError as packet_error:
                print(f"来自 {addr} 的数据包无效: {packet_error}")
                continue

            formatted_message = format_packet(packet)

            # 打印接收到的数据
            print(f"收到来自 {addr} 的数据:")
            print(formatted_message)
            print("-" * 50)
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭套接字
        sock.close()

if __name__ == "__main__":
    # 根据Unity代码中的设置，这里应该是接收端的IP和端口
    # 注意：如果Python程序和Unity运行在同一台机器上，可以使用localhost (127.0.0.1)
    # 如果在不同机器上，需要设置为Python程序所在机器的IP地址
    UDP_IP = "0.0.0.0"  # 监听所有可用网络接口
    UDP_PORT = 5050     # 与Unity代码中设置的端口一致

    # 直接在主线程接收，按 Ctrl+C 退出
    try:
        receive_udp_data(UDP_IP, UDP_PORT)
    except KeyboardInterrupt:
        print("程序已退出")
