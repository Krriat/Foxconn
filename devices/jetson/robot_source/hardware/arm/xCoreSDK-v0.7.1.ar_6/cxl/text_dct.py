import gpiod

# 对应 /dev/gpiochip0 的 line 6，用户确认实际接线为 DO2。
CHIP = "/dev/gpiochip0"
LINE = 6

# 暂按 1=吸合、0=断开；需要通过实际测试确认。
# 如果实际相反，交换这两个值。
ON = 1
OFF = 0

chip = gpiod.Chip(CHIP)
line = chip.get_line(LINE)
requested = False

try:
    # 只申请一次，初始设为断开
    line.request(
        consumer="relay-control",
        type=gpiod.LINE_REQ_DIR_OUT,
        default_vals=[OFF],
    )
    requested = True

    print("输入 on 吸合，off 断开，q 退出")

    while True:
        command = input("> ").strip().lower()

        if command == "on":
            line.set_value(ON)
            print("已发送吸合指令")
        elif command == "off":
            line.set_value(OFF)
            print("已发送断开指令")
        elif command == "q":
            break
        else:
            print("请输入 on、off 或 q")

except (KeyboardInterrupt, EOFError):
    pass

finally:
    try:
        if requested:
            # 正常退出或 Ctrl+C 时，先发断开指令
            try:
                line.set_value(OFF)
            finally:
                line.release()
    finally:
        chip.close()
