ReFlex 主从机械臂工作台 · 2026-10-31
访问 http://172.21.16.96:3002/ （原端口，同一内网，无登录）

已接入：7路主臂真实UART角度、7路从臂CAN反馈、全臂软件映射/预览，
以及J1底座轴实体跟随。J2–J6和夹爪尚未开放实体动作。

J1控制
- 每轮自动以主从当前位置为参考，不设电机零点。
- 主臂ID0 -> 从臂ID1。默认比例0.1：主臂转10°，从臂目标转1°。
- 从臂相对行程±1°，目标速度0.25°/秒，力矩上限1.0 Nm（用户本轮明确授权）。
- MIT Kp=200、Kd=3，沿用先前通过复验的J1设置。
- 仅J1使能和目标写入。其他轴不使能、不失能、不写运动目标。
- 每轮最多45秒；页面失去前台会请求停止，心跳1秒超时停止J1。
- 主臂数据200ms过期、输入跳变、反馈故障、偏移/速度超限停止。
- 初次单轴模式要求7轴初始失能，不接管运行中的电机。
- 停止只使J1失能，不代替全臂急停；现场防坠支撑仍须有效。
- 持久化参数、零位、固件、模式、其他轴限值均未写入。
- 启动时开启7轴主动状态上报(type24)，这是反馈配置，不是运动。

网页操作
1. 保持空载、净空、防坠支撑和急停值守，暂停3000网页手动控制。
2. 若读数提示跳变锁定，保持小主臂静止，点击“重新检查主臂读数”。
3. 勾选本轮现场条件，点击“开始 J1 受限跟随”。
4. 缓慢转小主臂底座5°～10°并停留5秒，观察本轮主臂/目标/从臂增量。
5. 缓慢回到起点，再点击“停止 J1 跟随”。
打开/刷新网页不会自动使能；重启后不自动恢复控制。
全臂预览的“已核对”记录不能解锁实体轴。

读数修正
SDK0.0.4零位确认计时反复重置，导致零位附近正常读数重复误判。
leader_reader.py只发送官方FashionStar command16，逐包检查应答头、
长度、校验和、命令和ID，区分有效新应答与缓存值；零位确认不再反复重置。
跳变锁定、数据过期和有效范围检查保留，不解锁主臂或复位多圈计数。

记录与部署
KR260：/home/ubuntu/reflex-arm/services/leader-follow/
  reflex-leader-board.service，127.0.0.1:9010，.venv-leader/bin/python
  /home/ubuntu/reflex-arm/runs/leader_web_20261031/j1_*/
    summary.json / samples.json / tx.json / rx.json 保存每轮实机记录。
  j1_driver_verified.json 与核心源文件SHA绑定，修改后必须重新验证。
  UART锁/tmp/reflex-leader-read.lock；CAN锁/tmp/reflex-can-actuation.lock。
AI PC：/home/aup/reflex-arm/services/leader-follow/
  reflex-leader-web.service，172.21.16.96:3002
  reflex-leader-tunnel.service，私有SSH转发到KR260 9010。
  Host/Origin校验保留，无公网端口映射。

测试
cd /home/ubuntu/reflex-arm/services/leader-follow
/home/ubuntu/reflex-arm/.venv-leader/bin/python -m unittest -v test_engine.py test_follow.py
run_pilot.py只用于显式CLI复验，不随服务启动，也没有HTTP入口。
维护时先结束联动，再在对应主机重启对应服务。

独立观察 · 2026-10-31
- 3002 已接入普通 USB 相机；/observer.jpg 与 /api/observer/status 为固定上游代理。
- 1°程序往返 j1_1793439835438611551：实际行程0.917970°，最大跟踪误差0.116043°，
  回位误差0.097657°，峰值电流0.622584 A；J1已失能、临时参数已恢复，其他轴运动写入为0。
- 40秒400张原始观察帧无采集错误。三处连杆特征扣除背景晃动后变化1.52–2.11像素，
  与编码器趋势相关系数0.967–0.972。当前小行程在缩小画面中不易看清。
- 该视觉结果支持程序往返存在实际运动；像素位移不是外部标定角度，
  相机与CAN无硬件同步，不能替代独立角度精度验收或全臂映射验收。
- 手动主臂完整往返与方向确认尚待本轮用户操作；六轴同时跟随及夹爪尚未开放。
