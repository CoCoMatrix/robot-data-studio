# Action–State 一致性算子 v1.0

面向机器人动作监督。一次读一个 episode，在原始采样率和源时钟上检查已配对的物理通道，输出哪些行、哪些动作通道可参与训练。程序只读源数据，CPU 运行，依赖 NumPy 与 PyArrow。

能力分两类：

- **语义**：识别 action 相对 state 的最佳滞后与物理含义。包括同帧 / 下一帧 / 时移状态复制、action 滞后于 state、符号反向、单位错误、速度与位置混淆、通道无指令，以及夹爪方向与量纲。
- **执行异常**：发指令但反馈无响应的窗口、首尾遥操接管跳变、灵巧手长期失配。

## 跑通一个真实样本

在仓库根目录执行（`PYTHONPATH=packages`，或先 `pip install -e .`）：

```bash
python -m robot_data_studio.quality.action_state.run_operator \
  --preset astribot-s1 \
  --parquet '/mnt/ccm-raw/vendor/VENDOR-000002/交付-星尘S1-110小时/星尘S1数据/家居/摆放餐具/data/chunk-000/episode_000022.parquet' \
  --single-episode --episode-id '家居/摆放餐具/episode_000022' \
  --out /tmp/astribot-22.json
```

`channels.right_arm_6.events` 中有两段 `unresponsive_feedback`（约 15.0–16.7 s、16.83–18.03 s），证据含指令变化范围、反馈变化范围和位置误差。完整示例见 [example-s1.json](validation/example-s1.json)。

## 输入与输出

| 输入 | 约定 |
|---|---|
| 身份与时间 | 稳定的 episode ID、严格递增的原生秒时间轴。多集 Parquet 用 `--episode-index N` 过滤；每集一文件用 `--single-episode`。 |
| 通道声明（profile） | 每通道声明 action/state 列与分量、源单位、物理量 `kind`（arm / hand / aperture / linear / *_velocity）、`role`（target / same_state / next_state / unknown / unsupported）和 `evidence`。夹爪另需原始值到开度 0–1 的标定折线和独立的开 / 闭锚点。 |
| 有效性 | 可提供每侧有效掩码和上游 `--keep-json`（源秒 `[[start,end], ...]`）。缺列、非有限值、无效帧与上游切口均保留边界。 |
| 检查范围 | 优先完整 episode。部分切片用 `--start-s` / `--end-s` 提供真实集首尾。异步设备先由适配器在原始时钟上配对。 |

输出为 JSON，含算子版本、profile SHA-256、源文件定位和逐通道证据。窗口同时给出半开行区间 `rows=[i,j)` 与秒区间。

| 字段 | 训练端用法 |
|---|---|
| `route` / `reason` | `command` 可作指令目标；`state_proxy` 可作已核实的同帧 / 下一帧状态目标；`review` 暂缓该通道动作监督。 |
| `command_keep_windows`、`command_mask_windows` | 可用指令行与需排除的行。 |
| `usable_proxy_windows`、`proxy_match_fraction` | 逐值复核通过的代理行及匹配率（低于 0.95 时整通道转复核）。 |
| `events` | 每个窗口的原因、作用范围、触发数值。 |
| `alignment`、`semantic` | 复制签名、最佳滞后与相关系数；语义探针结论、隐含超前帧数、斜率与零偏。 |
| `command_disposition` | `keep / keep_masked / no_command`。 |

```python
from robot_data_studio.quality.action_state import usable_mask

command_ok = usable_mask(report, ["right_arm_6"])
proxy_ok = usable_mask(report, ["right_arm_6"], supervision="state_proxy")
```

返回 `[原生行数, len(names)]` 布尔掩码。回写 30 Hz cache 时沿源行映射或 `source_seconds` 投影，再与原 `mask`、`row_valid` 取交集；通用输出的行号不能直接当重采样后的行号。

## 判定逻辑

| 问题 | 判据与处理 |
|---|---|
| 动作列是状态复制 | 运动帧上比较 `a[t]` 与 `s[t]`、`s[t+1]`，至少 20 个活动转移、一个时移近乎精确匹配且另一时移有区分度，记复制候选。声明为代理时逐值复核，匹配率 ≥0.95 才路由 `state_proxy`。 |
| 语义前置门（声明为 target 时） | `probe.py` 不依赖声明，在运动帧上估计：±1 s 内的时移精确复制；跟踪误差对速度回归得到的隐含超前帧数与显著性；水平相关、斜率与零偏；与 d(state)/dt 的相关。结论为同帧带噪复制、未来 / 过去状态复制、action 滞后于 state、符号反向、单位错误、速度当位置、action 恒定或与 state 无关时，该通道转 `review`（`reason=semantic_*`）。夹爪握持和手指接触会合理地造成脱节，这两类通道对“带噪复制 / 速度当位置 / 无关”只记复核事件。集首 2 s 初始化段不参与语义判定。确认合理的差异可在通道上声明 `accept_semantic: {verdicts: [...], evidence: ...}`。 |
| 指令变化、反馈长期不动 | 同一连续有效区间内检测“持续变化的命令 + 近乎静止的反馈”，以及“从已跟踪位置出现台阶、宽限期过后仍无响应”。须同时满足运动幅度、反馈噪声、持续时间和误差条件。已证实的停滞延续到反馈重新变化或误差消退。屏蔽命中通道。 |
| 接管跳变 | 在真实集首 / 尾附近检查同一关节的单帧指令跳变和反馈变化，要求误差同步增大，恢复需连续维持一段时间。profile 可选只记录或屏蔽、检查哪一端，以及按追上时刻或固定缓冲窗结束。 |
| 手部长期失配 | SharpaDex 专项规则：扣除已屏蔽的接管窗口后，逐手统计关节中位绝对误差，≥3 个关节超过 0.3 rad 时屏蔽该手整集动作监督。被语义门拦下的手指仍计入统计。 |
| 夹爪方向、量纲与接触 | 按标定表统一到开度 0–1，以独立开闭端点核对方向；检查截断前是否严重超出标定量程；统计充分开闭时的响应方向。端点反向、两次明确反向且无同向证据、动作恒定但反馈反复开闭、严重量程冲突，均暂缓该通道。 |
| 不可比较的物理量 | 角度 / 长度 / 速度按声明换算。跨物理量映射必须提供标定曲线和证据；把底盘位姿直接当速度不能通过。 |

## 接入新数据集

1. 对齐源时钟，指定上游 keep、源缺值与有效标志。明确每个通道的物理量、单位、方向，以及 action 是实际命令、同帧代理、下一帧代理还是未知。不确定时先声明为 `target`，由语义前置门给出证据。
2. 从[单通道 profile](example-profile.json) 开始，用 `--profile` 替换 `--preset`。夹爪标定折线与开闭锚点可参考 `python -m robot_data_studio.quality.action_state.profiles lingyu` 的输出。
3. 对 HDF5、ROS 等格式只需写读取适配器，调用 `analyze(episode_id, time_s, signals, profile, keep=...)`；`signals[name]` 含同长的一维 `action`、`state`，可选布尔 `action_valid`、`state_valid`，保持源单位。
4. 接入前按任务和设备采样，复核正常动作、真实坏窗、合法接触及时间断档，冻结 profile 后再批量运行。随代码提供 6 份预设：Astribot S1、SharpaDex、RoCo TaskBoard、LingYu、FineART、智元 G1。同一数据集含多个本体、控制模式或采样率时分别冻结。

## 阈值

单位：时间用秒，角度用 rad，开度用 0–1。基础关节配置为指令变化 0.06 rad、反馈峰峰值至多 0.002 rad、跟踪误差 0.08 rad、响应宽限 0.40 s、观察窗至少 0.65 s。覆盖顺序为 **物理量默认值 → 数据集同类通道 → 单通道**。

`python -m robot_data_studio.quality.action_state.calibrate good/*.json --out proposal.json` 可基于人工确认的正常集生成阈值建议：每集一票，默认每通道至少 10 个有激励的 episode。采用前，应在未参与标定的任务、设备或批次上比较坏窗检出、窗口覆盖和正常保留率。

## 重现验证

```bash
python -m pytest tests/quality/test_action_state.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m robot_data_studio.quality.action_state.validate_real --workers 8
```

单测只用合成数据。真实回放读取随包的[冻结案例](validation/cases.json)，依赖相同的只读数据挂载；逐例对照与代码哈希见[实际回放结果](validation/real-replay.json)。

### 本版实测范围

- 单测 20/20。
- 真实回放：6 个数据集共 156 个 episode，结果 155/156。其中 SharpaDex 接管坏窗 24/24、手部失配 24/24；S1、RoCo、SharpaDex 历史保留窗口各 24/24，LingYu 23/24。唯一分歧是 LingYu 叠积木第 278 集：右臂第 4 关节在 1.600–2.289 s 内指令跨度 0.182 rad、反馈仅 0.00077 rad，短时停滞和慢响应无法区分，暂列待复核。
- 故障注入留出集：64 个真实保留集、830 个臂通道，共 11,461 例，均未参与调参。语义故障检出率 96.9–100%，与 1126 原版的 0–74% 对比；通道错配单通道视角 73.1%；反馈冻结 97.5%，集首接管 98.7%；正常对照误报 1.0%。
- 14 个数据集 cache 抽检（15,423 个片段）：相对历史交付结果新增复核 0.34% 的指令通道片段。

### 已知限制

- 只凭 action/state 无法区分“控制器指令”和“未来若干帧读数”，例如 G1 双臂超前约 4 帧，这类问题需上游确认数据来源。
- 低运动片段无法判定语义，数据集 / 本体级汇总门禁尚未实现。
- 夹爪绝对开合极性仍需每数据集一个开闭锚点。
- 逐片段通道配对受联动关节干扰，只有数据集级的稳定错配可信；“声明为速度实为位姿”的门控规则尚未接入。
- 四元数、无对应反馈的底盘、物理量不匹配的槽位不做标量判定；未测策略训练效果。
