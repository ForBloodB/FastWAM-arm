"""Generate an evidence-linked HTML engineering report; no invented measurements."""
from html import escape
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
def read(path): return json.loads((root/path).read_text(encoding='utf-8'))
abc=read('runs/abc_v1/summary.json')
timing=read('runs/timing_1ms/summary.json')
probe=read('runs/board/can/post_test_probe.json')
motion=read('runs/board/j1_test_v1/summary.json')
monitor=read('runs/board/j1_monitor/summary.json')
kinematics=read('runs/kinematics/summary.json')
faults=read('runs/network_faults/summary.json')
fpga=read('runs/diagnostic_build/summary.json')
pl=read('runs/board/pl_board/summary.json')
test_count=sum(int(s.get('tests',0)) for s in ET.parse(root/'runs/tests/junit.xml').iter('testsuite'))

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def link(path,label): return f'<a href="{escape(path)}">{escape(label)}</a>'
rows=[[r['seed'],f"{r['A']['success_rate']:.2%}",f"{r['B']['success_rate']:.2%}",f"{r['C']['success_rate']:.2%}",r['online_interactions'],r['promotions']] for r in abc['seeds']]
mean=abc['mean_success_rate'];std=abc['std_between_seeds']
rows.append(['均值与种子间标准差',*[f'{mean[g]:.2%} ± {std[g]:.2%}' for g in 'ABC'],'每种子 8192','共 '+str(sum(r['promotions'] for r in abc['seeds']))])
curve=[]
curves=[read(f'runs/abc_v1/seed_{r["seed"]}/learning_curve.json') for r in abc['seeds']]
for i in range(len(curves[0])):
    rates=[c[i]['C']['success_rate'] for c in curves]
    curve.append([curves[0][i]['interactions'],*[f'{v:.2%}' for v in rates],f'{statistics.mean(rates):.2%}'])
adapt=[]
for name in abc['seeds'][0]['adaptation']:
    before=statistics.mean(s['adaptation'][name]['before']['success_rate'] for s in abc['seeds'])
    after=statistics.mean(s['adaptation'][name]['after']['success_rate'] for s in abc['seeds'])
    adapt.append([name,f'{before:.2%}',f'{after:.2%}',f'{(after-before)*100:+.2f} 个百分点'])
udp=timing['udp'];host=timing['host']
motorrows=[[m['id'],round(m['position_rad']*180/3.141592653589793,4),round(m['voltage_V'],3),m['run_mode'],m['timeout_raw']] for m in probe['motors']]
acceptance=[
 ['1 基础抓放','部分完成','模型逆解、轨迹、视觉几何、实物 CAN 反馈','完整识别、抓取、抬升、分类放置尚未执行'],
 ['2 在线增益','代理模型完成','3 种子 A/B/C、独立 gate、模型和逐次记录','不是实物在线学习证据'],
 ['3 条件适应','代理模型完成','位置、摩擦、形变条件前后独立评估','真实材料实验尚未执行'],
 ['4 实时控制','部分完成','主机 1 ms 调度、真实网络 RTT、K26 布局布线、PL 周期计数','PL 诊断未连接电机，缺同执行器同轨迹比较'],
 ['5 可靠性','部分完成','六项 PL 状态机诊断、协议故障测试、J1 限制触发','缺物理急停/限位到电机输出的完整保护链实测']]

html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ReFlex Arm 实施与验证记录</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1100px;margin:42px auto;padding:0 24px;color:#1e293b;line-height:1.7}h1,h2{color:#0f172a}h1{font-size:30px}h2{font-size:22px;margin-top:36px}p{max-width:980px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #cbd5e1;text-align:left;padding:9px 12px}th{background:#e2e8f0}tr:nth-child(even){background:#f8fafc}a{color:#0369a1}code,pre{background:#f1f5f9}pre{padding:16px;white-space:pre-wrap}small{color:#475569}@media print{body{margin:0;max-width:none}h2{break-after:avoid}tr{break-inside:avoid}}</style><body>'''
html+='<h1>ReFlex Arm 实施与验证记录</h1><p>记录日期 2026 年 10 月 19 日。AI PC 172.21.16.96，KR260 172.21.16.34，机械臂 SEEED B601-RS；V80 服务器 172.18.20.66 提供 Vivado 2025.1。<strong>软件与 PL 诊断工程已实际部署运行，原文档要求的整套实物抓取和现场在线学习验收尚未完成。</strong>本报告分清代理模型、RTL 仿真、实物 PS/CAN 测量、真实 PL 诊断及未验证事项。</p>'
html+='<p><strong>2026-10-19 后续修复：</strong>J1 速度尖峰处理及超时单位已修正，J1 和夹爪受限微动复测通过，最后 7 个电机均失能且故障位为零。Gemini 305 已取得 RGB-D 和厂家内参；外参与完整抓取仍未完成。详见 '+link('REPAIR_REPORT.html','最新修复与标定报告')+'、'+link('calibration/START_HERE.html','标定板打印步骤')+'。下方首次 CAN 测试保留为修复前历史证据。</p>'
html+='<h2>验收状态</h2>'+table(['实验','状态','已有证据','缺口'],acceptance)
html+='<h2>首次硬件连通与低速测试（修复前历史记录）</h2><p>以下段落描述首次联调时的状态，后续结果以上方最新修复报告为准。PCAN-USB 已在 KR260 识别为 can0，速率 1 Mbps。先进行仅监听捕获，再发送类型 17 的只读参数请求，全部 7 个电机响应。没有执行归零、改变电机 ID、保存闪存或运行学习策略控制。电机电压约 48 V。当时仅取得 USB RGB 画面，尚未建立相机到机械臂标定。</p>'
html+=table(['电机','实测位置 °','电压 V','run_mode','原有 timeout 参数'],motorrows)
html+='<p>用户授权低速空载后，执行 J1 单关节测试。目标在 6 秒内最多偏转 0.5°并返回，命令峰值速度约 0.262°/秒，临时转矩上限 0.5 N·m，其余电机未使能。脚本在反馈速度超过 3°/秒时中止；<strong>该测试未通过</strong>。J1 失能和原参数恢复记录均已保存。电机看门狗请求为 100 ms，但实际物理触发时限未测量；不得由参数回读推断已完成安全验证。</p>'
html+=table(['项目','结果'],[['运行结果',motion.get('error')],['J1 失能',motion.get('disabled_after_test')],['恢复转矩上限 N·m',motion.get('restored_torque_limit_Nm')],['恢复 timeout 原值',motion.get('restored_timeout_raw')],['静止位置变化范围 °',f"{monitor['position_range_deg']:.6f}"],['静止速度参数峰值 °/秒',f"{monitor['max_abs_reported_velocity_deg_s']:.4f}"],['静止速度超阈值样本',str(monitor['reported_velocity_gt_3_deg_s_samples'])+' / 500']])
html+='<p>现场用户反馈未见明显转动或抖动；这与静止期间速度参数出现尖峰的记录一致，但不能据此确定故障根因。首次失败点的具体数值未进入 CSV；原始异常和之前的 142 个合格采样保留。脚本现已增加失败点快照，尚未重新执行运动。MotorBridge 的 get_state 返回空状态，不能解释为电机无故障。继续动作前应完成速度反馈质量、状态帧兼容性和单位核验。</p>'
html+='<p>另已增加对固件 type-24 主动状态帧的被动解析。全部 7 个电机温度约 28–30°C；7 号夹爪存在原始状态掩码 0x10，其含义尚未针对当前固件核对，未自动清除。原始状态位保留在 '+link('runs/board/can/status_type24.json','被动状态解析记录')+'。</p>'
html+='<p>'+link('runs/board/j1_test_v1/summary.json','J1 失败与恢复记录')+' · '+link('runs/board/j1_test_v1/samples.csv','动作采样 CSV')+' · '+link('runs/board/j1_monitor/samples.csv','静止监测 CSV')+' · '+link('runs/board/can/passive.jsonl','被动 CAN 原始帧')+' · '+link('runs/board/can/post_test_probe.json','电机参数回读')+'</p>'
html+='<h2>代理模型中的在线学习对照</h2><p><strong>以下数值只属于合成的单步任务空间模型，不能视为真实抓取率。</strong>接触、摩擦、滑落与形变由 env.py 中的公式定义，没有物理接触仿真。A 是固定零残差而非成熟传统抓取规划器。B 为预训练冻结；C 从同一 B 权重开始异步更新。每种子预训练 8000 步、在线交互 8192 次，最终每组使用同一批 512 个独立场景。材料类别为模拟输入，未训练真实未知物体分类器。</p>'
html+=table(['种子','A 固定基线','B 冻结 RL','C 在线 RL','在线交互','策略晋升'],rows)
html+='<p>训练、gate 和测试种子空间独立。Learner 只读取 Actor 回放，gate 和测试不进入训练；晋升依据 gate 成功率不退化及奖励改善。初始文件 SHA256 用于确认 B 未被 C 修改。标准差为 3 个训练种子之间的离散程度，不是实物置信区间。异步最后一批交互保留在日志，最终候选只使用其快照时已有的数据。</p>'
html+=table(['累计在线交互','种子 11 C','种子 22 C','种子 33 C','C 均值'],curve)
html+='<p>条件适应单独测试，每种子每条件 512 个场景；这些场景也不进入回放。</p>'+table(['条件','冻结策略均值','在线策略均值','变化'],adapt)
html+='<p>'+link('runs/abc_v1/summary.json','完整学习结果')+' · '+link('runs/abc_v1/config.json','固定实验配置')+' · '+link('runs/abc_v1/seed_11/online_replay.jsonl','示例 Actor 原始交互')+' · '+link('runs/abc_v1/seed_11/policy_switches.json','示例晋升与拒绝记录')+'</p>'
html+='<h2>真实以太网和主机调度测量</h2><p>AI PC 发往 KR260 PS 的 5000 次 UDP 诊断请求使用主机单调时钟测 RTT；板端 Python 用户态服务只回诊断信息。另采集主机目标 1 ms 周期的 5000 次唤醒。这不是同执行器同轨迹的电机控制对照。</p>'
html+=table(['指标','实测值'],[['RTT 均值 ms',f"{udp['rtt']['mean']/1000:.6f}"],['RTT P99 ms',f"{udp['rtt']['p99']/1000:.6f}"],['超时率',f"{udp['timeout_rate']:.2%}"],['主机目标周期 ms',f"{host['target_period_us']/1000:.3f}"],['主机实际周期 P99 ms',f"{host['actual_period_us']['p99']/1000:.6f}"],['主机绝对周期抖动 P99 ms',f"{host['absolute_period_jitter_us']['p99']/1000:.6f}"],['主机截止期超限率',f"{host['deadline_miss_rate']:.2%}"],['电机轨迹跟踪 RMSE 与 PL 对照','尚未完成']])
html+='<p>截止期超限定义为实际唤醒比计划唤醒迟到至少一个完整目标周期。该短时单机结果没有负载扫描，不能推断最坏情况实时性。KR260 系统日历不可靠，不能使用两台设备的日历或单调时间戳相减求单程延迟。防火墙仅新增允许 172.21.16.96 访问 172.21.16.34 UDP 47000 的规则。先前 1000 次/5 ms 测量也保留在 runs/timing_v2。</p><p>'+link('runs/timing_1ms/rtt.csv','逐次 RTT')+' · '+link('runs/timing_1ms/host_timing.csv','主机原始时间戳')+' · '+link('runs/timing_1ms/summary.json','时序指标定义与结果')+'</p>'
html+='<h2>验证和 FPGA 状态</h2><p>软件测试 '+str(test_count)+' 项通过。真实网络解析测试 '+str(sum(r['passed'] for r in faults['cases']))+' 项通过，涵盖有效包、重复包、倒序、CRC 损坏、短包、不支持命令、数值溢出和过深嵌套。B601 原始 URDF 的 20 个近邻位置 IK 样例全部收敛，未验证姿态约束、碰撞和动力学。Icarus 的 Safety FSM、插值器与诊断集成 testbench 均通过。</p><p>使用 V80 上的 Vivado 2025.1 为 K26 完成综合、布局布线和 bitstream 生成。诊断工程含 PS AXI、五组 AXI GPIO 内部寄存器、计数器、Safety FSM 与内部插值器。已通过 xmutil 加载至 KR260，UIO4–UIO8 为本工程寄存器；不含实际电机或外部安全信号输出。原有 starter-kit 固件保留，提供 restore_pl_default.sh 恢复。</p>'
html+=table(['实现或板上指标','结果'],[['CLB LUTs',fpga['clb_luts']],['CLB Registers',fpga['clb_registers']],['WNS ns',fpga['wns_ns']],['WHS ns',fpga['whs_ns']],['布局布线时序约束',fpga['timing_constraints_met']],['板上最近周期计数的 1000 次读值',str(pl['unique_measured_period_cycles'])],['该次读数期间 PL tick 增量',pl['tick_count_delta']],['硬件看门狗记录距心跳的周期',next(t['cycles_since_heartbeat'] for t in pl['tests'] if t['case']=='hardware_watchdog')],['板上状态机检查通过数',sum(t['passed'] for t in pl['tests'])],['结束状态','DISARMED']])
html+='<p>标称 100 MHz 时钟下，100000 周期对应 1 ms，10000001 周期对应约 100.00001 ms。主机取样的 1000 次最近周期值并非保证覆盖每一个连续周期；只能报告已读出的计数，不能据此声称电机级 deadline miss 为零。板上检查涵盖过期使能拒绝、硬件心跳看门狗、通过测试寄存器注入急停/限位/传感器/命令异常及手动复位。它们证明该 PL 逻辑在板上工作；物理急停、限位和电机失能接口仍未接入，不能宣称已实现完整独立硬件保护。DRC 无错误，保留 1 项 RTSTAT-10 无可布线负载警告，涉及 SmartConnect 生成的未使用复位流水线网络。</p><p>'+link('runs/diagnostic_build/timing.rpt','完整布局布线时序报告')+' · '+link('runs/diagnostic_build/utilization.rpt','资源报告')+' · '+link('runs/diagnostic_build/drc.rpt','DRC 报告')+' · '+link('runs/board/pl_board/summary.json','实际 PL 测试结果')+' · '+link('runs/board/pl_board/period_samples.jsonl','板上原始计数读值')+'</p>'
html+='<p>'+link('runs/tests/junit.xml','软件测试证据')+' · '+link('runs/network_faults/summary.json','真实诊断故障测试')+' · '+link('runs/rtl/safety.vcd','RTL 安全仿真波形')+' · '+link('runs/kinematics/summary.json','B601 运动学验证')+'</p>'
html+='<h2>复现和继续联调</h2><pre>cd /home/aup/reflex-arm\n.venv/bin/python -m pytest -q\nbash scripts/test_rtl.sh\n.venv/bin/python scripts/check_b601_model.py\n.venv/bin/python -m reflex.experiment --out runs/abc_new\n.venv/bin/python -m reflex.benchmark --host 172.21.16.34 --out runs/timing_new</pre><p>KR260 目录 /home/ubuntu/reflex-arm；Vivado 构建目录 /home/hxt/reflex-arm 位于 172.18.20.66。诊断服务为用户进程，未设置系统自启；CAN 配置未设置开机自动执行。依赖 wheels 保留在 AI PC wheelhouse，ARM MotorBridge wheel 与解包依赖位于板端工程内。导出工程包包含源代码、锁定依赖列表、模型、bit/bin/XSA、时序资源报告和原始证据，不包含虚拟环境、第三方 wheels 或本机 SSH 凭据。</p><p>下一步需要定位速度反馈问题并核对夹爪状态，建立标定后的 RGB-D 感知、夹爪力与碰撞约束，将物理安全与电机控制接入经验证的 FPGA 接口。之后才可在限力现场工况下逐项开展实物抓放、真实在线学习对照、同轨迹实时控制对照和完整硬件安全实验。</p>'
html+='<p><small>参考 '+link('https://stable-baselines3.readthedocs.io/en/master/modules/sac.html','SB3 SAC')+'、'+link('https://docs.amd.com/r/en-US/ug1092-kr260-starter-kit','AMD KR260 指南')+'、'+link('https://github.com/RobStride/Python_Sample','RobStride 官方协议示例')+'。本报告中的量化结果均来自工程内保存的运行记录。</small></p></body></html>'
(root/'REPORT.html').write_text(html,encoding='utf-8')
print(root/'REPORT.html')
