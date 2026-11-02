"""Evidence-linked repair report and physical calibration handoff."""
import json
import math
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
def read(p):return json.loads((root/p).read_text())
base='runs/board/repair_01/'
j1=read(base+'move_j1_v3/summary.json');j7=read(base+'csp_j7_v4/summary.json')
final=read(base+'final_status/summary.json');watch=read(base+'watchdog_constant_effort/summary.json')
camera=read('runs/repair_01/orbbec_calibration_profile/summary.json')
raw=read(base+'disabled_raw/samples.json')
good=[r for r in raw if all(r[p].get('status')==0 for p in ('0x7019','0x701b'))]
positions=[r['0x7019']['f32'] for r in good];vel=[r['0x701b']['f32'] for r in good]
slopes=[(positions[i]-positions[i-10])/((good[i]['0x7019']['rx_ns']-good[i-10]['0x7019']['rx_ns'])/1e9) for i in range(10,len(good))]
stationary={'samples':len(good),'position_range_deg':math.degrees(max(positions)-min(positions)),
    'raw_velocity_max_deg_s':math.degrees(max(map(abs,vel))),
    'raw_spikes_gt3_deg_s':sum(abs(v)>math.radians(3) for v in vel),
    'encoder_100ms_velocity_max_deg_s':math.degrees(max(map(abs,slopes)))}
test_count=sum(int(s.get('tests',0)) for s in ET.parse(root/'runs/tests/junit.xml').iter('testsuite'))
summary={'evidence':'actual_KR260_and_AI_PC','date':'2026-10-19','j1':j1,'j7':j7,
    'stationary_j1':stationary,'final_all_disabled_fault_free':final['success'],
    'camera_factory_acquired':camera['success'],'external_calibration_complete':False,
    'physical_grasp_complete':False,'software_tests_passed':test_count,
    'caveat':'701B estimator noise remains in device telemetry; software guards use independent encoder motion checks'}
(root/'runs/repair_01/summary.json').write_text(json.dumps(summary,indent=2))
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(s))+'</th>' for s in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(s))+'</td>' for s in r)+'</tr>' for r in rows)+'</table>'
def link(url,title):return f'<a href="{escape(url)}">{escape(title)}</a>'
style='<style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1050px;margin:35px auto;padding:0 24px;color:#203047}h1,h2{color:#10243a}h2{margin-top:32px;font-size:22px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #cbd5e1;text-align:left}th{background:#e8eff5}a{color:#086b91}img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#eef2f6;padding:15px}.notice{border-left:4px solid #c77b12;padding:12px 18px;background:#fff8e8}</style>'
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>J1 与夹爪修复、标定准备</title>'+style+'<body>'
html+='<h1>J1 与夹爪修复、标定准备</h1><p>2026-10-19 · SEEED B601-RS · KR260 172.21.16.34 · AI PC 172.21.16.96</p>'
html+='<p class="notice"><strong>J1 和夹爪的受限微动复测已通过，最终 7 个电机均失能且故障位为零。相机取得真实 RGB-D 和厂家几何参数；外参标定和完整实物抓取尚未完成。</strong>电机的瞬时速度估计仍含噪声，修复针对反馈使用、保护判据及测试控制路径，没有宣称修改或修好电机固件。</p>'
html+='<h2>J1 速度反馈</h2><p>20 秒静止采样的原始 type-17 回复确实带有速度尖峰，参数索引与小端浮点解析正确。发送新请求前排空旧队列后现象仍存在。0x3017、0x3022、0x3034 等探测返回状态 1，判为不支持，不能把其中的零值视为有效反馈。旧采集程序因同时要求 0x3017 有效而把 valid_samples 写成 0；此处根据保存的原始回复重新计算，仅使用有效的 0x7019 和 0x701B。</p>'
html+=table(['静止观测','结果'],[[k,round(v,6) if isinstance(v,float) else v] for k,v in stationary.items()])
html+='<p>新保护同时检查 20/50 ms 编码器位移速度、持续和中值速度、原始极端值、反馈新鲜度、位置包络、供电和电机状态。J1 的编码器速度阈值仍为 3°/秒；J7 的静止速度估计峰值约 6°/秒，因此原始估计通道使用 12°/秒的诊断阈值，编码器速度阈值仍为 3°/秒。原始数据保留，未把异常样本改写为正常值。这是用户态诊断保护，不是独立硬件速度限制器。</p>'
html+='<h2>实际微动结果</h2>'
motionrows=[]
for label,path,r in [('J1 MIT，kp=200，kd=3','move_j1_v3',j1),('J7 CSP，内部限速 0.5°/秒','csp_j7_v4',j7)]:
    samples=read(base+path+'/samples.json')
    measured=max(abs(math.degrees(v)) for s in samples for v in s.get('velocity_guard',{}).get('encoder_rad_s',[]))
    motionrows.append([label,r['success'],f"{r['actual_amplitude_deg']:.4f}",f"{r['final_error_deg']:.4f}",f"{r['tracking_rmse_deg']:.4f}",f'{measured:.4f}',len(samples)])
html+=table(['受限测试','通过','实际幅度 °','结束偏差 °','全程 RMSE °','编码器窗口速度峰值 °/秒','采样'],motionrows)
html+='<p>J1 目标 0.5°，临时转矩上限 0.5 N·m。低刚度初测跟踪较弱，提高指令刚度后误差减小。夹爪 MIT 测试在克服静摩擦时触发了编码器确认的超速，保护停机；没有放宽编码器速度阈值。CSP 测试目标 2°，电机内部限速 0.5°/秒，电流上限 0.45 A，另有 0.5 A 软件检查。CSP 的输出约束依靠电流限制，不能把 MIT 转矩参数解释为已标定夹力。该 2° 指电机角度，尚未标定为夹口毫米。</p>'
html+='<p>0x10 对应堵转／过载标志。发送一次 clear=1 的停机帧后，标志仍持续；随后正常 stop=0 帧完成了这次观察到的清除过程。之后微动和最后 10 秒被动复查未再出现该标志。最初为什么进入过载保护缺少历史数据，不能由清除成功推断机械全行程或负载能力均正常。</p>'
html+='<p>只分别使能过 J1 和 J7；J2–J6 未使能。结束后模式、转矩限制、通信超时、CSP 速度及电流参数均回读恢复。固件会在失能时把动态 loc_ref 重设为编码器位置，因此不强制恢复历史目标；下次动作始终从实测位置起步。没有归零、保存闪存或运行学习策略控制。</p>'
html+='<h2>通信超时单位与实测行为</h2><p>厂家 RS00/RS06 手册规定 20000 计数为 1 秒；旧 API 请求 100 后实际回读 raw=100，只有约 5 ms。当前 100 ms 设置使用 raw=2000，避开该 API 的单位歧义。恒定 0.05 N·m 小信号测试记录如下。</p>'
html+=table(['参数 raw','手册标称 ms','反馈力矩首次恢复为精确零的观测 ms','模式进入 Reset'],[[t['raw'],t['manual_timeout_ms'],round(t['effort_zero_after_ms'],3),t['reset_after_ms']] for t in watch['trials']])
html+='<p>这是力矩反馈和低频状态帧上的观测时间，包含电机估计/滤波和报文间隔，不能当成精确断能时刻。两组试验中模式仍为 Run，故禁止用模式位断言超时已失能。明确 stop=0 后才回到 Reset。FPGA 诊断尚未接入实际电机，完整硬件急停链仍待验证。</p>'
html+='<h2>相机与打印标定板</h2><p>已离线安装官方 pyorbbecsdk2 2.1.2，并为 USB VID:PID 2bc5:0840 配置 plugdev 组访问，未开放全部 USB 设备。Gemini 305 固件 1.0.70，校准采集使用彩色 1280×800@10、原始深度 848×530@10，深度已对齐至彩色。实测深度尺度 '+str(camera['aligned_depth_scale_m'])+' 米/计数，约 0.1 mm/计数。保存完整 8 项有理畸变系数；厂家内参与机器人外参分别管理。</p>'
html+='<p>'+link('calibration/ChArUco_A4_5x7_24mm.pdf','下载 A4 ChArUco 标定板')+' · '+link('calibration/START_HERE.html','中文打印与放置步骤')+' · '+link('configs/camera_factory.json','实测厂家相机参数')+'</p>'
html+='<p>5×7 方格，格长 24 mm，标记边长 18 mm，字典 DICT_4X4_50，黑白区域 120×168 mm。PDF 已渲染检查并识别出全部 24 个角点、17 个标记。实际打印仍需用户测量底部 100 mm 标尺和 24 mm 方格。</p>'
html+='<img src="runs/repair_01/orbbec_calibration_profile/color.jpg" alt="Gemini 305 实际视野，标定板尚未放入"><p>上图为当前实际相机视野。先清空可见桌面，将板贴平整硬底板并置于镜头前，使完整黑白区域和白边可见。现在不指定物体的基座坐标，也不启动自动标定姿态序列。</p>'
html+='<h2>下一阶段验收</h2><p>先确认相机支架是否随腕部移动。固定相机需已知或实测的 T_base_board；随动相机需至少 15 个静止姿态/图像对，并包含两个独立转轴的变化，不能只转 J1。已实现 ChArUco 位姿和两种变换方向的求解函数，眼在手上候选按 80% 拟合、20% 留出验证。图像重投影 RMS 门限 0.7 px；留出位姿一致性要求 3 mm、1°以内。退化采样会被拒绝；实际标定还须使用独立测量点核验，不能只靠拟合误差。</p>'
html+='<p>首个抓取物建议为 30–40 mm、低于约 30 g 的轻质泡棉方块；这是待验证的首轮试验建议。标定完成、关节方向/净空和夹口方向/开度核验后，在相机视野内选择可达且无碰撞的固定位置。首次仅做低夹力抓取与短距离抬升，成功后再扩大重复次数；当前没有实物抓取成功率。</p>'
html+='<h2>代码与原始证据</h2><p>软件测试 '+str(test_count)+' 项通过，包括记录帧、排队旧回复、速度噪声与真实移动的区别、变换方向、退化姿态拒绝、留出样本异常和标定板识别。软件测试不能替代物理净空、夹力及硬件保护验证。</p>'
for path,title in [('disabled_raw','J1 静止原始帧'),('move_j1_v3','J1 最终微动'),('csp_j7_v4','夹爪最终 CSP 微动'),('watchdog_constant_effort','通信超时小信号测试'),('final_status','最后状态与参数回读')]:
    html+='<p>'+link(base+path+'/summary.json',title)+' · '+link(base+path,'原始数据目录')+'</p>'
html+='<p>'+link('runs/repair_01/summary.json','修复汇总 JSON')+' · '+link('runs/tests/junit.xml','软件测试结果')+' · '+link('REPORT.html','完整工程报告')+'</p>'
html+='<p>协议依据：'+link('https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS00/RS00User%20Manual251112.pdf','厂家 RS00 手册')+'、'+link('https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS06/RS06User%20Manual251112.pdf','厂家 RS06 手册')+'；相机 API：'+link('https://github.com/orbbec/pyorbbecsdk','Orbbec 官方 SDK')+'；坐标解算：'+link('https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html','OpenCV 官方文档')+'。</p></body></html>'
(root/'REPAIR_REPORT.html').write_text(html,encoding='utf-8')
guide='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>标定板打印与摆放</title>'+style+'<body><h1>先打印这一张 A4 标定板</h1>'
guide+='<p>'+link('ChArUco_A4_5x7_24mm.pdf','打开可打印 PDF')+'</p><ol><li>纸张 A4，纵向，选择“实际大小／100%”，关闭适合页面或缩放。</li><li>用尺子确认底部标尺 100 mm，单格 24 mm；黑白区域 120×168 mm。</li><li>贴到平整、坚硬的底板，保留四周白边，避免覆亮膜、弯曲和皱折。</li><li>清开 Gemini 305 看到的桌面和线缆，放入整张板。板的黑白区域应完整清晰可见，占图像宽度约 20–50%；不要遮挡在夹爪后。</li><li>回复“板已放好”，由程序读取实际图像检查识别和尺度。标定板放好后先固定住。</li></ol>'
guide+='<p>规格：ChArUco，5×7 方格，squareLength=0.024 m，markerLength=0.018 m，DICT_4X4_50。不是普通纯棋盘格。PDF 文件已经渲染和自动识别核对。</p><p>相机固定在基座/外部时，后续建立固定相机外参；相机随腕部移动时，采集多姿态手眼数据。两种方式使用同一张板。</p><p>建议准备一块 3–4 cm 的轻质泡棉方块。暂不需要把物体摆进机械臂动作区域，准确位置由实际标定和可达检查后确定。</p><p>当前相机内参已获取，外参尚未完成；标定采集程序不会自动使能机械臂。</p></body></html>'
(root/'calibration/START_HERE.html').write_text(guide,encoding='utf-8')
print(json.dumps({'report':'REPAIR_REPORT.html','tests':test_count,'final_status_ok':final['success']}))
