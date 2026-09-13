# EgoHandKit 优化接入审阅与对比说明

更新时间：2026-09-13

## 结论摘要

本次 `observations` 前端优化是基于：

```text
/home/user/roboego-hand-vis/docs/pipeline/pre_hamer_observation_frontend_migration.md
```

迁移的核心内容是“同帧 observation 去重”和“跨帧 physical hand 关联”，而不是修改 HaWoR、HaMeR、WiLoR 或 HTM 的网络结构。

当前推荐链路为：

```text
视频
  -> session + camera 独立命名和抽帧缓存
  -> Detectron2 person proposals
  -> ViTPose whole-body keypoints
  -> official hand keypoint gate
  -> 原图坐标手部 bbox
  -> Hand Observation Consolidation
  -> Physical Hand Temporal Association
  -> selected_for_hamer
  -> 原有 backend crop generator
  -> HaWoR
  -> MANO 参数和 mesh
  -> 原图 overlay
```

当前命令中的 `--backend hawor` 表示后端使用 HaWoR；`--frontend observations` 表示使用新的前端。

## 接入前后是否可以对比

可以。接入前后应保持以下变量一致：

- 输入视频和 camera；
- backend，例如都使用 `hawor`；
- focal length；
- GPU 和 batch 参数；
- 是否启用 Omega；
- 输出目录分开，避免缓存复用。

接入前（legacy）：

```bash
python run.py \
  --input "/path/to/session/videos/left.mp4" \
  --backend hawor \
  --frontend legacy \
  --gpu 0 \
  --img_focal 553.9148003055425 \
  --force_detect \
  --output_root "/path/to/compare/before"
```

接入后（observations）：

```bash
python run.py \
  --input "/path/to/session/videos/left.mp4" \
  --backend hawor \
  --frontend observations \
  --gpu 0 \
  --img_focal 553.9148003055425 \
  --force_detect \
  --output_root "/path/to/compare/after"
```

主要对比文件：

```text
before/<sequence>/render_hawor.mp4
after/<sequence>_observations/render_hawor.mp4
```

以及：

```text
pass1_raw.pkl
pass2_cleaned.pkl
observations_raw.pkl
observations_selected.pkl
*_hawor.pkl
observation_selection.json
```

之前已有一份 60 秒选择层三路对比视频：

```text
/home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/three_way_baseline_comparison/full_60s/comparison/raw_vs_original_vs_optimized_rgb.mp4
```

该视频固定为：A 原始全部候选、B 原仓库选择、C 优化选择。它证明了前端候选选择层可以对比，但不等同于当前 EgoHandKit 重新完成 legacy HaWoR 与 observation HaWoR 的严格端到端基准。

## 新前端具体做了什么

### 1. Detectron2 + ViTPose

- 保留所有 score > 0.5 的 person proposal，而不是只取每帧最高分的人；
- 对每个人独立运行 ViTPose whole-body；
- 使用最后 42 个点拆分 left/right 两组 21 点；
- 仅保留超过 3 个 keypoint 的 confidence 严格大于 0.5 的手；
- keypoint 坐标最小值和最大值生成手 bbox。

### 2. 同帧 Observation Consolidation

同一真实手可能由多个 person proposal 重复产生。模块使用以下空间和质量证据聚类：

- bbox IoU；
- bbox 中心距离；
- 共同可靠手部 keypoint 的 median 距离；
- wrist/MCP palm 距离；
- handedness 软约束；
- keypoint 有效比例、均值、median、person score、bbox completeness。

每个 cluster 只保留一个原始 observation，不平均 bbox 或 keypoints，因此不会引入新的几何平滑偏差。

### 3. Physical Hand Temporal Association

对去重后的 0~2 个 observation 做全序列动态规划。代价包括：

- bbox 中心移动距离；
- bbox IoU；
- bbox 尺寸变化；
- 手部 keypoint 距离；
- palm 距离；
- handedness 变化惩罚；
- observation quality；
- missing gap、track start/restart 和 unassigned 代价。

轨迹是匿名 physical track slot，不直接等同于解剖学左手或右手。默认最多允许两帧短缺口连接；长缺口会开启新的 fragment。缺失帧仍保留为空状态，不合成 bbox 或 MANO 姿态。

### 4. HaWoR 适配

HaWoR 按 physical track 的连续 segment 推理，而不是简单把所有 left 帧和所有 right 帧拼在一起。遇到以下情况会切段：

- 缺失帧；
- fragment 改变；
- 原始 handedness 改变。

这样可避免跨不同物理手或跨长遮挡强行连接时污染 HaWoR 的 temporal context。

## 减少抖动的真实范围

本次优化主要减少的是“输入选择和身份跳变”：

- 不同 session 的同名 `left.mp4` 不再复用旧帧缓存；
- 多人 proposal 切换减少；
- 同一只手的重复候选减少；
- 左右手 physical track swap 减少；
- 遮挡后重新出现时错误接轨减少；
- focal 在推理、相机平移和渲染之间保持一致；
- overlay 保留完整原始时间轴，不因空手帧缩短视频。

本次没有做：

- Kalman filter；
- One Euro filter；
- EMA 或 moving average；
- MANO rotation/vertex 后处理平滑；
- 全局 orientation 修正；
- missing frame 的姿态插值。

因此，不能把本次优化描述成“最终 MANO pose 已经被滤波平滑”。它主要是减少错误目标、错误身份和错误 crop 输入造成的抖动。ViTPose 本身噪声、严重遮挡和 HaWoR 姿态抖动仍可能存在。

## 是否完全算法解耦

准确说法是：**前端算法逻辑独立，接口级解耦，但存在少量适配耦合。**

已经解耦的部分：

- consolidation 不读取 HaWoR/HaMeR 输出；
- temporal association 只使用 bbox、keypoints、置信度和帧序列；
- 不依赖 MANO、深度、VIO 或 Omega；
- 同一 `selected_for_hamer` 可交给 HaWoR、HaMeR、WiLoR 或 HTM；
- 后端通过 `HandInstance` / `Pass3Inputs` 接收统一输入。

仍存在的适配边界：

1. 前端 bbox 必须符合后端 crop generator 的坐标约定；
2. handedness 会影响后端 left-hand flip 和 `is_right`；
3. HaWoR 使用 physical track/fragment 决定 temporal segment；
4. HaWoR 的 focal 同时影响推理和 overlay；
5. association 是完整离线序列 DP，不是实时逐帧 tracker。

因此不能声称前端与后端“完全无关”，但可以称为“可复用的 pre-HMR observation selection 模块，通过统一数据接口接入多个 HMR backend”。

## 审阅证据

### 代码结构证据

迁移模块位于：

```text
observation_frontend/adapter.py
observation_frontend/hand_observation_consolidation.py
observation_frontend/physical_hand_temporal_association.py
observation_frontend/pre_hamer_observation_frontend.py
```

主入口通过 CLI 显式切换：

```text
run.py --frontend legacy
run.py --frontend observations
```

后端仍通过：

```text
run.py --backend hawor|hamer|wilor|htm
```

独立选择。

### 自动化测试

在 `egohandkit` 环境执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/user/miniconda3/envs/egohandkit/bin/python \
  -m pytest tests -q
```

结果：

```text
57 passed, 2 warnings, 6 subtests passed
```

覆盖内容包括：

- 同帧 observation consolidation；
- physical hand temporal association；
- observation frontend cache；
- 空帧和完整视频时间轴；
- HaWoR focal 解析；
- session/camera 输入命名；
- Omega chunk 参数解析。

### 前端迁移回放验证

执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/user/miniconda3/envs/egohandkit/bin/python \
  scripts/validate_frontend_migration.py \
  /home/user/roboego-hand-vis/tmp/vio_hand_pose_eval/failure_mining/full_60s/cache/full_sequence_temporal_pipeline.json
```

结果摘要：

```text
source_unchanged: true
frame_count: 1800
input_observation_count: 3489
consolidated_observation_count: 2973
hamer_input_observation_count: 2872
mismatched_frames: []
max_bbox_difference: 0.0
max_keypoint_difference: 0.0
network_inference: false
```

这证明迁移后的选择层对源缓存没有修改，且 replay 的选中 frame/candidate、bbox 和 keypoints 与参考结果一致。它验证的是前端选择等价性，不是姿态精度提升证明。

## 当前边界和审阅结论

- 这是 observation selection 和 track association 的优化，不是 HMR 网络升级；
- 主要收益是减少重复、错误切换和输入轨迹不连续；
- 需要用同一 backend 做 legacy/observations 端到端视频对照，才能报告最终 overlay 质量差异；
- 当前没有足够证据宣称所有数据集上 pose accuracy、FPS 或最终 jitter 都必然改善；
- Omega 默认关闭是针对手部 overlay 的运行策略，不能与 observation 前端优化混为一谈。

