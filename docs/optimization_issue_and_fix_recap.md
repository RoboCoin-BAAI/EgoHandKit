# Observations 前端效果问题与修复记录

## 1. 当前观察到的问题

在以下相机对比视频中，`observations` 前端的视觉结果没有稳定优于 Legacy：

```text
/home/user/ego_data/结果对比/egohandkit_left_batch/session_20260825_170413_将商品放到购物车里_20260828181649/camera_before_after.mp4
```

问题表现为手部 mesh 偶尔抖动、尺度或位置不稳定，以及左右手语义在部分帧不连续。

这不是视频播放速度、拼接顺序或 focal 不一致导致的。两条链路使用同一段 `left.mp4` 和相同的 `img_focal=600`。`camera_before_after.mp4` 也是相机坐标系 overlay，Omega 世界坐标恢复不参与这个视频的相对差异。

## 2. 根因分析

### 2.1 HaWoR 时序输入被切得过碎

Legacy 在该样本中只形成约 4 个 HaWoR 时序段，其中有一条约 430 帧的长段。

Observations 前端最初形成了 67 个 HaWoR segments，统计结果为：

```text
raw observations                 988
consolidated observations        626
selected for HaWoR               539
physical track fragments          31
handedness flip events            27
suspected track switch events     31
missing segments                  39
HaWoR segments                    67
```

当前 HaWoR adapter 会在缺帧、fragment 变化或 handedness 变化时切段。大量短段会反复重置 HaWoR 的 temporal attention，使输出在段与段之间不连续。

相关代码位于 [`mesh_recovery.py`](../mesh_recovery.py) 的 observation 输入构造部分。

### 2.2 这不是简单的“快速运动好、慢速运动差”

当前 DP 关联主要依赖相邻帧的：

```text
bbox 中心距离
bbox IoU
bbox 尺度变化
关键点距离
手掌关键点距离
```

它没有恒速模型、Kalman prediction 或 optical-flow prediction。因此快速运动会提高中心距离和关键点距离，并降低 IoU；如果同时有运动模糊或遮挡，反而更容易断轨。

快速运动只有在 Legacy 已经发生明显漏检或 ID switch、而 Detectron2/ViTPose 仍提供了可靠候选时，才可能从 Observations 前端中受益。

这个样本的主要问题更接近：候选手相互接近、遮挡、左右手候选混杂和物理轨迹身份交换，而不是运动速度本身。

### 2.3 all-person 检测提高召回，同时引入了错误候选

Observations 使用 Detectron2 检出所有人物，再对每个人运行 ViTPose。相比 Legacy 的 YOLO/清洗路径，它会保留更多候选。召回率提高，但候选中也可能包含其他人的手或同一人物的错误手部候选。

DP 可能为了降低局部运动代价，将另一只移动较慢的手关联到当前物理轨迹，从而造成 identity switch。

## 3. 已尝试的修复

### 3.1 增加稳定的 backend handedness

之前 HaWoR 直接使用每一帧的 ViTPose `handedness`。现在增加了 `backend_handedness`：

```text
原始 handedness       保留，用于诊断
backend_handedness    只在轨迹左右手标签稳定度 >= 85% 时使用
```

这样，单帧 handedness 抖动不会立即导致 HaWoR crop flip 和时序窗口重启；而左右标签混杂严重的轨迹不会被强制改成 dominant side，避免把真实左手错误当成右手。

实现位置：

- [`observation_frontend/pre_hamer_observation_frontend.py`](../observation_frontend/pre_hamer_observation_frontend.py)
- [`mesh_recovery.py`](../mesh_recovery.py)

### 3.2 验证结果

修复后的 HaWoR 输入段数从：

```text
67 段 → 48 段
```

测试结果：

```text
58 passed, 6 subtests passed
```

修复后的相机对比视频已经重新生成，路径仍为：

```text
/home/user/ego_data/结果对比/egohandkit_left_batch/session_20260825_170413_将商品放到购物车里_20260828181649/camera_before_after.mp4
```

当前代码提交：

```text
a547042 fix: stabilize HMR crop side across frontend flips
```

## 4. 当前修复的边界

这次修复解决的是：

```text
瞬时 handedness 抖动导致的 crop flip 不稳定
部分 HaWoR segment 被无意义切断
```

它还没有完全解决：

```text
物理轨迹在不同人物/不同手之间发生 identity switch
快速移动时缺少速度预测
all-person 候选中目标人物选择不稳定
```

因此目前不能宣称 Observations 已经在所有数据上优于 Legacy。当前版本更准确的定位是：

```text
高召回、物理轨迹建模的实验链路
```

而不是已经完成时序稳定化的最终链路。

## 5. 下一步建议

下一步应优先修改物理关联，而不是继续调整 focal 或简单做全局平均：

1. 引入短期速度预测，使快速移动的真实手不会因为位移大而被错误候选替代。
2. 对轨迹切换增加滞回机制，要求候选切换连续成立若干帧后才确认。
3. 对候选增加 bbox 尺度、关键点置信度、中心速度和遮挡状态的质量门控。
4. 在明确目标为“相机佩戴者的手”时，增加目标人物先验，避免 all-person 检测把其他人的手送入 HaWoR。
5. 做以下 ablation，区分收益和退化来源：

```text
Legacy
Observations + consolidation
Observations + consolidation + DP
Observations + DP + handedness stabilization
Observations + velocity-aware association
```

## 6. 世界坐标视频说明

本次修复后只重新生成了相机坐标系的 HMR overlay。原来的：

```text
world_before_after.mp4
```

仍然是旧的 Omega 结果。若要比较修复后的世界坐标轨迹，需要使用新的 HMR 输出重新运行 Omega；这与相机 overlay 的问题是两个独立环节。
