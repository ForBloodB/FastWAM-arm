# KR260 PL 诊断工程

已在 hxt@172.18.20.66 的 `/opt/Xilinx/2025.1/Vivado/bin/vivado` 构建，并通过 xmutil 加载至 KR260。该工程仅提供内部计数、状态机和插值器的板上测量，不连接真实电机或外部安全引脚。

## 实际构建

```bash
cd /home/hxt/reflex-arm
/opt/Xilinx/2025.1/Vivado/bin/vivado -mode batch -source fpga/build_diagnostic.tcl
```

目标器件 `xck26-sfvc784-2LV-c`，板卡 preset `xilinx.com:kr260_som:part0:1.1`。诊断主时钟为 PS PL0 的标称 100 MHz，板端已核对 fclk0 指向 clock ID 71，加载前 Linux 报告 99,999,999 Hz。

布局布线输出在 `runs/diagnostic_build`：bit、适用于 FPGA manager 的 bin、XSA、资源报告、时序报告和 DRC 报告。最终资源为 1909 CLB LUT 与 3780 寄存器；WNS 6.480 ns、WHS 0.011 ns。DRC 的 RTSTAT-10 是 SmartConnect 生成的未使用复位流水线网络警告，保留原始报告，没有通过 waiver 隐藏。

## 已核实的寄存器映射

每个 AXI GPIO 占 64 KiB，寄存器均为 32 位小端。双通道 GPIO 第二通道的数据偏移为 0x08。

|基址|UIO 名称|0x00|0x08|
|---|---|---|---|
|0xA0000000|reflex-control|测试控制字|无|
|0xA0010000|reflex-stats|签名与状态|64 位运行计数低 32 位|
|0xA0020000|reflex-clock|运行计数高 32 位|PL tick 累计次数|
|0xA0030000|reflex-period|最近一次 tick 间隔时钟数|最近心跳的时钟计数低 32 位|
|0xA0040000|reflex-fault|进入故障态时的时钟计数低 32 位|内部插值器位置|

测试控制字：bit0 心跳，bit1 arm，bit2 手动 reset，bit3 急停注入，bit4 限位注入，bit5 传感器故障注入，bit6 非法命令注入。心跳按上升沿检测，需要在两次心跳之间清 bit0。

状态：高 16 位固定 `0x5246`；bit0 为内部使能；bits2:1 为 DISARMED=0、ARMED=1、FAULT=2；bits7:4 为故障码，1 急停、2 限位、3 传感器、4 命令、5 看门狗。`test_pl_board.py` 按物理基址和 UIO 名称双重校验，再检查签名，才执行测试寄存器写入；不会访问原有 axi-pmon。

## 加载与恢复

在 KR260 已部署的工程目录中：

```bash
sudo bash scripts/load_pl_diagnostics.sh
sudo python3 scripts/test_pl_board.py
```

加载脚本会编译 overlay，将诊断固件安装到 `/lib/firmware/xilinx/reflex-diagnostic`，通过 xmutil 切换应用。原 `/lib/firmware/xilinx/k26-starter-kits` 保留。恢复命令：

```bash
sudo bash scripts/restore_pl_default.sh
```

实际板上测试记录位于 `runs/pl_board`。测试结束为 DISARMED。1000 次读取的最近 tick 间隔均为 100000 时钟；这些读取不保证覆盖全部连续 tick。看门狗记录为心跳后 10000001 时钟；该值对应 PL 状态记录，不是物理执行器停机时间。

## 真实机械臂集成边界

当前 `drive_enable` 只进入内部插值器。软件寄存器注入故障可验证 PL 状态机，但没有证明外部急停接线、碰撞保护或电机失能路径。PCAN-USB 仍由 PS 的 SocketCAN/USB 栈访问；后续需确定可由 PL 定周期控制的执行接口和物理安全信号，才能完成同执行器同轨迹的比较以及真正独立的保护验证。

参考 AMD 的 FPGA manager 和 Kria 应用流程：

- https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841847
- https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/target.html
