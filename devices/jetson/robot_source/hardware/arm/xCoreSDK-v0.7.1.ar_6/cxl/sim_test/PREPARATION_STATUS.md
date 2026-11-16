# 本次准备结果（2026-09-09）

- aarch64 只读SDK采集器已编译：`build/inspect_ar5_sdk`。
- 10项离线自动化测试通过，其中端到端故障测试覆盖11种故障注入。测试不证明实机watchdog或制动性能。
- 本地能力调查：[capabilities.json](capabilities.json)。实际控制器型号/版本、反馈、限位/工具参数待现场采集。
- 左右臂各7份预览已生成：`plans/left_J1`～`left_J7`、`plans/right_J1`～`right_J7`。均为相对关节轨迹预览，未通过实机执行检查。
- 完整mock采集、回放、多频辨识、整段留出验证、绘图和打包已跑通。索引：[output/validation_manifest.json](output/validation_manifest.json)。所有数据均为合成数据，不能用作机器人实测结论。
- 未连接机器人；没有上电、切模式、运动、修改保护设置或运行现有控制程序。

操作步骤见 [README.md](README.md)。现在可分别运行左右臂静止只读采集；退出其他控制程序后再连接。

**未完成的实机部分：** ServoJ运动执行适配器、独立保护/超时与控制器watchdog链路、停止轨迹及TCP/双臂碰撞核验、现场关节映射/限额和载荷确认。SDK运动入口在连接前拒绝执行，不会因为配置被标成已确认就自动开放。真实伺服响应、工作区力矩需求和厂商能力包络尚未测得。证据与缺口见 [SDK_FINDINGS.md](SDK_FINDINGS.md)。
