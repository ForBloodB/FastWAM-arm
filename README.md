# FastWAM-arm

面向 Seeed B601-RS 的端侧视觉—语言—动作（VLA）机械臂工程。仓库整合 395 推理主机、AMD Kria KR260、双相机、MuJoCo 仿真、FPGA 参考设计以及 Star Arm 102 主从示教代码。

## 当前验证边界

- 已完成真实双相机与真实关节状态下的 FastWAM 推理、限幅空走、手眼标定和通信测试。
- 已完成 KR260 上 LeRobot 主从连接、从臂校准、动作数据录制及轨迹回放流程验证。
- 轨迹录制／回放成功不等于视觉引导抓放或精密插装成功；上述物理任务仍需按场景验收。
- FPGA 内容包含独立上板和历史参考设计；不能据此宣称 PL 已串入真实电机安全控制链路。
- `research/evoarm/` 是独立的仿真研究模块，其结果不能当作 FastWAM 实机结果。

## 目录

| 路径 | 内容 |
|---|---|
| `host/` | FastWAM 推理适配、真实状态输入、双相机、标定与视觉定位 |
| `simulation/b601/` | 官方 B601-RS MuJoCo 场景、双视角和抓放／插装实验 |
| `stack/` | 395—KR260 通信、安全约束与协议代码 |
| `integration/` | 公共模块、Protobuf 和板端接口 |
| `kr260/` | ROS 2／KRS 机器人源码及 reflex-arm 控制、FPGA 和测试代码 |
| `lerobot/SHOW/shell/` | 主从动作录制和一句话关键词回放脚本 |
| `hardware/` | PLA 演示工装的 Blender 源文件、生成脚本和 STL |
| `research/evoarm/` | 独立仿真研究源码与 FPGA 参考资料 |
| `third_party/` | FastWAM、Seeed LeRobot 及机器人支持包；许可证随源码保留 |
| `docs/` | 技术报告与项目文档 |

## 主从示教与回放

在 KR260 的 LeRobot 环境中，将 `lerobot/SHOW/shell/` 放到 `~/rebot_lerobot/SHOW/shell/`。

```bash
# 录制一段动作，最长 180 秒
EPISODE_TIME_S=180 bash ~/rebot_lerobot/SHOW/shell/record.sh \
  pick_block "Pick up the left block" 1

# 清空工作区后，回放第 0 段轨迹
bash ~/rebot_lerobot/SHOW/shell/replay.sh "抓起左边方块" 0
```

回放会使从臂主动运动。运行前停止其他 CAN／串口控制程序，清空工作区，确认夹爪空载并可正常归零，并准备切断动力。

## 复现说明

各模块依赖不同，不能共用一个未经检查的 Python 环境：

- FastWAM：参考 `third_party/FastWAM/README.md` 与 `host/fastwam/requirements-brain-rocm72.txt`。
- B601 仿真：使用 MuJoCo／Pinocchio 环境，入口见 `simulation/b601/`。
- LeRobot：参考 `third_party/lerobot/README.md` 和 Seeed B601-RS 官方说明。
- EVOArm 仿真研究：见 `research/evoarm/README.md`。

模型权重、数据集、虚拟环境、编译产物、日志和个人凭据不在公开源码包中。实机标定参数具有设备和安装位置依赖性，换机或移动相机后必须重新验证。
