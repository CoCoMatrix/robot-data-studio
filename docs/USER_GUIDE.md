# Robot Data Studio 使用教程

这份教程面向第一版 MVP：导入一个 LeRobot 数据集，选择 episode，用 Rerun 回放可视化，并导出 ACT 风格 HDF5。

## 1. 这个版本能做什么

当前版本已经跑通了完整链路：

- 导入本地 LeRobot v3 数据集
- 在前端查看数据集规模、episode 列表和基础信息
- 选择单个 episode 后生成 Rerun `.rrd` 记录文件
- 在网页里嵌入 Rerun WebViewer 回放 action、state 和视频
- 将单个 episode 导出为 ACT 风格 HDF5

第一版还不是完整清洗平台。清洗规则、数据质量评分、坐标系转换、多格式互转会作为后续模块接进来。

## 2. 环境准备

需要本机已有：

- Python 3.11+
- Node.js
- pnpm，或使用 Codex 自带的 pnpm 路径

如果你在 Codex 当前项目目录里操作，项目路径是：

```bash
/Users/peterxie/Desktop/data platform 
```

注意：这个目录名最后有一个空格。如果你手动 `cd`，建议复制上面的路径，或者后面把目录重命名掉。

## 3. 安装依赖

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
pnpm install
```

如果你执行 `pnpm` 时看到：

```text
zsh: command not found: pnpm
```

说明你的 shell 里没有全局 pnpm。任选一种方式解决：

方式 A：全局安装 pnpm，之后命令最短。

```bash
npm install -g pnpm
pnpm install
```

方式 B：不安装全局 pnpm，直接用 Codex 自带的 pnpm。

```bash
/Users/peterxie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm install
```

安装完成后，Python 后端、测试工具、前端依赖都会准备好。

## 4. 下载 LeRobot 样例数据集

第一版默认使用 Hugging Face 上的 `lerobot/pusht` 样例数据。

执行：

```bash
.venv/bin/python scripts/download_sample.py
```

下载完成后，数据会放在：

```bash
data/samples/lerobot-pusht
```

当前已经验证过的数据规模是：

- LeRobot 版本：v3.0
- episodes：206
- frames：25,650
- fps：10
- 视频流：1 个，`observation.image`

## 5. 启动后端 API

打开一个终端，执行：

```bash
.venv/bin/uvicorn apps.api.main:app --reload --port 8000
```

默认后端地址：

```text
http://127.0.0.1:8000
```

如果启动成功，终端里会看到 Uvicorn 正在监听 `127.0.0.1:8000`。

## 6. 启动前端

再打开一个终端，执行：

```bash
pnpm dev:web
```

如果你没有全局 pnpm，用：

```bash
/Users/peterxie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm dev:web
```

默认前端地址：

```text
http://127.0.0.1:5173
```

浏览器打开这个地址后，会看到 Robot Data Studio 的工作台页面。

## 7. 导入数据集

页面顶部有一个数据集路径输入框，默认已经填好：

```text
data/samples/lerobot-pusht
```

点击导入按钮后，前端会调用后端 API 索引数据集。

导入成功后，页面会展示类似信息：

- 数据集格式：LeRobot
- episode 数量
- frame 总数
- fps
- 视频 keys

左侧会出现 episode 列表。第一版建议先选择 `episode 0` 做验证。

## 8. 运行清洗 Pipeline

导入数据集后，点击页面工具栏里的 `运行清洗 Pipeline`。

后台会读取每个 episode 的 action、state、timestamp 和视频可用性，生成一个 0 到 1 的质量分数。
前端会把 episode 分成三个虚拟文件夹：

- 待审查：分数处在中间区间，建议人工复核
- 排除：质量分数较低，默认不进入后续清洗结果
- 通过：质量分数较高，可直接保留

第一版不会复制、删除或改写源数据集。清洗状态保存在：

```bash
.rds-artifacts/projects/<project_id>/cleaning_state.json
```

选择待审查 episode 后，可以继续用 Rerun 回放，再在右侧 Quality Report 中点击 `通过` 或 `排除`。
人工决策会覆盖自动分桶，并在后续重新运行清洗时默认保留。

左侧 `搜索 / 筛选` 可以按 episode 编号或任务文本过滤列表，也可以勾选问题类型，例如模糊帧、
时间不同步、Action 跳变和 VLM 失败。右上角的 `VLM 设置` 可以直接设置 provider、model 和是否启用
VLM 检查；当前前端会把这些参数随清洗请求发送给本地 API，后续可在后端 scorer adapter 中接入真实 VLM。

## 8.1 动态运动学一致性与修正

打开 `Kinematic consistency`，先导入 URDF，再点击 `自动发现修正`。系统不依赖固定的 `tool0`、`eef` 或厂商字段名，而是枚举 URDF 运动链与数据列组合，用 held-out SE(3) 残差选择 profile：

- position residual 单位为米；
- orientation residual 同时显示弧度和角度；
- 支持 quaternion xyzw/wxyz、rotation 6D、rotation matrix、rotation vector 与 RPY；
- 可建议 base/TCP transform、joint sign 与 joint zero offset；
- 双臂或多臂 profile 按 chain 独立保存和评估。

先查看 before/after p95、confidence、candidate margin、evidence 和 warnings。`ambiguous`、`insufficient_excitation`、`no_candidates` 或 `fit_failed` 都是安全拒绝：补充更有激励的轨迹或明确映射后重跑，不要直接降低阈值。点击接受前会再次确认；URDF、数据或映射变化后旧 proposal 自动变为 stale。

接受只更新项目 profile。Filter 预览和导出会用修正后的 FK 重算派生 EEF pose，但不会改写源数据。`conversion_report.json` 会带上 proposal ID、fingerprint、profile 和 held-out 指标。高级用户仍可展开 `手动配置（高级）` 使用旧 XYZ-only 兼容路径；没有姿态字段时不要把结果称为完整 SE(3)。

命令行使用同一套 engine：

```bash
python scripts/qwen_filter_dataset.py /path/to/dataset \
  --urdf /path/to/robot.urdf \
  --discover-kinematics \
  --proposal-output /path/to/proposal.json
```

Codex 中可调用已安装的 `$align-robot-kinematics` skill。它的 wrapper 接受 repository、dataset、URDF、sample budget 和 output path，输出与 API 相同的 proposal/refusal JSON，并要求在接受前先人工授权。

### Astribot / 星尘智能 S1 验证清单

当前代码只在仓库的任意命名合成 URDF fixture 上通过，**尚未对未来提供的 S1 URDF 宣称验证通过**。拿到 S1 URDF 与一段包含末端姿态、关节充分运动的数据后：

1. 用 skill wrapper 生成 proposal，确认没有厂商 link 名硬编码；
2. 核对每条 arm chain、joint/state index、pose representation 和 frame；
3. 检查 held-out position/orientation before→after 与歧义警告；
4. 在 UI 预览后再接受；
5. 导出一份小样，核对源文件前后 SHA-256 不变，并检查 `conversion_report.json` correction manifest；
6. 若 S1 轨迹接近静止或候选近似等价，保留拒绝结果并补采 excitation trajectory。

## 9. 用 Rerun 回放 episode

选择一个 episode 后，点击 Rerun 回放按钮。

后台会做三件事：

1. 读取该 episode 的 parquet 数据
2. 生成 Rerun `.rrd` 文件
3. 把 `.rrd` 地址返回给前端

前端会把这个 `.rrd` 加载进 Rerun WebViewer。正常情况下，你会在页面里看到：

- action 曲线
- state 曲线
- 时间轴
- 视频实体
- Rerun 的 viewer 面板

生成的 `.rrd` 文件会保存在：

```bash
.rds-artifacts/
```

## 10. 导出 ACT 风格 HDF5

选择 episode 后，点击 HDF5 导出按钮。

第一版导出的 HDF5 结构是：

```text
/action
/observations/qpos
/observations/timestamp
```

同时写入这些属性：

```text
sim
source_format
episode_index
```

导出文件也会保存在：

```bash
.rds-artifacts/
```

文件名类似：

```text
e252f69849d4-episode-000000.hdf5
```

## 11. 验证整个流程

如果你想确认本地状态是健康的，执行：

```bash
.venv/bin/pytest -q
.venv/bin/ruff check apps packages tests scripts
pnpm test:web
pnpm build:web
```

如果没有全局 pnpm，把最后两行换成：

```bash
/Users/peterxie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm test:web
/Users/peterxie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm build:web
```

当前这套 MVP 已经验证过：

- 后端测试通过
- 前端测试通过
- 前端 production build 通过
- LeRobot 样例数据可导入
- 清洗 Pipeline 可生成通过、待审查、排除三类状态
- 待审查 episode 可人工通过或排除
- episode 0 可生成 Rerun `.rrd`
- Rerun WebViewer 可正常加载
- episode 0 可导出 HDF5

## 11. 常见问题

### 11.1 页面打不开

先确认前端服务是否启动：

```text
http://127.0.0.1:5173
```

如果打不开，重新执行：

```bash
pnpm dev:web
```

### 11.2 导入失败

检查后端是否启动：

```text
http://127.0.0.1:8000
```

再检查数据集目录是否存在：

```bash
ls data/samples/lerobot-pusht
```

正常应该能看到 `meta`、`data`、`videos` 这些目录。

### 11.3 Rerun 一直加载不出来

先确认已经运行过：

```bash
pnpm install
```

前端启动或构建时会自动复制 Rerun WebViewer 需要的 WASM 文件到：

```bash
apps/web/public/rerun/re_viewer_bg.wasm
```

如果缺失，可以重新启动前端：

```bash
pnpm dev:web
```

### 11.4 HDF5 导出后去哪里找

导出文件默认在：

```bash
.rds-artifacts/
```

可以用下面命令查看：

```bash
ls -lh .rds-artifacts/*.hdf5
```

## 12. 后续扩展方向

这个 MVP 的接口已经按 reader / viewer / exporter 分层，后面可以继续加：

- 数据质量检查：缺帧、NaN、时间戳不连续、episode 长度异常
- 清洗规则：过滤坏 episode、裁剪片段、重采样、字段修复
- 格式转换：LeRobot、HDF5、ROS bag、Zarr、RLDS 等
- 坐标系转换：相机坐标、机器人 base/world 坐标、end-effector pose 转换
- 评分系统：规则评分 + VLM 辅助检查
- 数据集报告：每个数据集自动生成质量 summary 和可导出的 report

第一版的目标不是把所有功能堆满，而是先把“导入 → 可视化回放 → 导出”这条最核心的闭环跑稳。

## 13. Visual Quality 与 DOVER-Mobile

Visual Quality 默认使用 `auto` backend：规则扫描始终运行；只有显式配置独立 Python 与本地 checkpoint 后，才会批量运行 DOVER-Mobile technical quality。RDS 不会自动下载、打包或提交模型权重。

内部非商业环境可固定使用 DOVER 官方仓库 commit `f1ddc96215bc7fbcf8f315c65d47905f339c3419`，并在隔离环境中安装依赖：

```bash
git clone https://github.com/VQAssessment/DOVER.git
cd DOVER
git checkout f1ddc96215bc7fbcf8f315c65d47905f339c3419
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

在项目的 `filter_config.json` 中配置本机绝对路径：

```json
{
  "visual_quality": {
    "backend": "dover_mobile",
    "dover_python_path": "/absolute/path/to/DOVER/.venv/bin/python",
    "checkpoint_path": "/absolute/path/to/DOVER-Mobile.pth",
    "model_device": "cuda",
    "batch_size": 8
  }
}
```

CPU 模式会自动将实际 batch size 限制为 1。runner 只执行 technical 分支；单相机失败会标记为 partial，runner 整体失败则回退 rules，filter run 仍会成功。完整 run 会写入带配置/checkpoint fingerprint 的 episode detail 和 camera cohort baseline；单 episode 重跑只复用最近一次匹配的完整 baseline。

DOVER 官方仓库使用非商业许可证，部署前请核对 [DOVER 仓库](https://github.com/VQAssessment/DOVER) 与 [S-Lab License](https://raw.githubusercontent.com/VQAssessment/DOVER/master/LICENSE)。
