# Xiaoyang → Hang：IMU 端进度交接

Hang，我把 IMU expert 和 phase arbitrator 两块都跑通了，已经 push 到 main。
下面是你需要知道的事，分四块。

---

## 1. 我做完的

### IMU Expert（`code/models/imu_expert.py`）

- **Deep 1D-CNN + residual + BN**，输入 `(B, 6, 192)`，输出 `(B, T_i=12, 256)`
- **关键设计**：不做 global pool！保留时序维度，给 cross-attention 和 phase gating 用
- 单模态训练 80 epoch，**test acc = 82.33%**，比 midterm `IMUNet` 67.9% 高 14.43 个百分点
- Train/test split：subjects {1,3,5,7} / {2,4,6,8}（和你那边一致）

权重在 Drive：

```
/content/drive/MyDrive/pgmoe_ckpt/
├── imu_expert.pt              ← 只有 encoder，给 PG-MoE 加载用
├── imu_classifier_best.pt     ← 完整模型（含临时分类 head）
└── backups/*_v1_acc82_33.pt   ← 带版本号的备份
```

> 这个文件夹我需要 share 给你才能让你的 Colab 读到。我等下发你共享链接。

### Phase Arbitrator（`code/models/phase_arbitrator.py`）

这是整个 paper 的 novelty 模块。

- **输入**：raw IMU `(B, 6, 192)`（**不是** IMU expert 的输出！）
- **输出**：α(t) `(B, T_i=12)` ∈ [0, 1]
- **结构**：
  1. `PhaseFeatures`（无参数）抽 3 个物理量：|a(t)|、d²|a|/dt²、d/dt(|a|²)
  2. AdaptiveAvgPool 192 → 12
  3. PhaseEncoder：3 → 64 → 32（用 1×1 Conv 等价于逐 t 的 MLP）
  4. ArbitratorMLP：32 → 16 → 1 → Sigmoid
- 总参数仅 **2,881**

**为什么用物理量做输入而不是 deep feature**：让 α 可解释、可验证、不依赖 IMU expert 训练好坏。这就是我们和 MMTSA/DynMM 黑盒 gating 的本质区别。

### Cross-Modal Attention（`code/models/cross_attention.py`）

**这本来是你负责的**，我先写了一个可工作版本（ViLBERT Co-TRM 双向 cross-attention，2 层，4 heads），是为了让 PG-MoE 能 end-to-end forward pass 不报错。

**接口约定**：

```python
cross_attn(vision, imu) -> (vision_enhanced, imu_enhanced)
# vision: (B, T_v, 256), imu: (B, T_i, 256)
```

只要你最终的版本守这个接口，我们就能直接换。如果你已经写好了，告诉我，我替换就行。

---

## 2. 几个关键发现

### Finding 1：IMU 单模态的最差 3 类完美贴合 PG-MoE 假设

```
c4   right arm throw    acc=0.250
c18  right hand knock   acc=0.438
c16  tennis serve       acc=0.500
```

**全是快速冲击/弹道运动**——单点 IMU 看不到方向信息。而 100% 正确率的 13 类全是**周期性 / 加速度模式独特**的动作（walking, squat, draw circle, bowling, ...）。

**这直接是我们 method 部分的论据**：α(t) 应该在 throw/knock/serve 这些类上偏 vision，在 walking/squat 上偏 IMU。

### Finding 2：物理特征图教科书级清晰

我在 `figures/phase_features_throw.png` 画了一个 throw 样本的三个特征：

- 0–50 timestep：prep，|a|≈1（只有重力）
- 50–70：pre-accel，|a| 升到 2
- **75–80：impact，|a| 尖峰 5.4，d²|a|/dt² ±4 剧烈摆动，energy_rate 从 +14 跌到 -20**
- 85–130：follow-through 衰减

**这是 paper method figure 的核心素材**。未训练的 α(t) 在 `figures/alpha_untrained.png`，所有动作都贴在 0.527——sigmoid of 随机权重，符合预期。等联合训练完，这条曲线应该出现 phase 起伏，throw 在 timestep 75 附近 α 该升高。

---

## 3. 关于接口的注意事项

### d_model = 256

- plan 第三节约定。
- 你那边 ResNet3D 输出本来就是 256，OK。
- Qwen2.5-VL-3B 原生是 2048——如果你用 Qwen，**末尾要加一个 Linear(2048 → 256)** 投影到 256，否则我们对不上。

### Vision token shape

- 我们的 cross-attention 是用 `(B, T_v, 256)` 的标准格式。
- 你的方案 A（Qwen）原生输出 `(B, T_v=60, N_patch=64, 2048)`。如果保留 patch 维度，cross-attention 会变成 60×64=3840 个 vision token 对 12 个 IMU token，计算量爆炸。**建议你在 vision expert 内部先把 patch 维度 attention pool 掉**，输出 `(B, T_v, 256)`。
- 你的方案 B（ResNet3D）输出 `(B, T, 256)`，**直接能用**。

### 时间对齐

- 我们 IMU 是 T_i = 12。
- 你那边 T_v 是多少？60 还是 pooling 后更少？告诉我，我在 fusion 这里调适配。
- 当前 `pgmoe.py` 用 `AdaptiveAvgPool1d` 把 vision pool 到 T_i 做对齐，不依赖 T_v 具体值。

### 想用你的哪个 backbone

我个人建议**主线走 ResNet3D（89.3%）**，Qwen 当 ablation。我们 paper 的 novelty 在 phase arbitrator，不在 vision backbone。从 89.3% 起步整体上 92%+ 是有说服力的故事；从 58.6% 起步要靠 IMU 推 30+ 点，比较难。

但这是你的决定，你说了算。

---

## 4. 我需要你给我的东西

按急迫性排序：

1. **缓存的 vision tokens 文件**（最急）
   - 格式：每个样本一个 `(T_v, 256)` 的 tensor，或者全部样本打包成 `(N, T_v, 256)`
   - 用 filename `aA_sS_tT.pt` 或者 dict `{filename: tensor}` 都行
   - 给我 Drive 上的路径就能加载
   - 没有这个我没法做联合训练

2. **你 cross-attention 模块的实际代码**（如果你写了的话）
   - 直接替换 `code/models/cross_attention.py` 就行
   - 接口前面说了

3. **vision 单模态 baseline 的精确数字**
   - 你说的 ResNet3D 89.3% 是 best 还是 final？
   - 用了什么 split？也是 subjects {2,4,6,8} 做 test 吧？
   - 哪个 epoch、训了多久？

4. **vision expert 的代码**（如果还没 push 到 repo）
   - 放到 `code/models/vision_expert.py` 和 `code/data/extract_vision_tokens.py`
   - 不然只有缓存文件没法复现

---

## 5. 我接下来准备做什么

- 写 `code/models/pgmoe.py` 把所有模块串起来（Cross-attn 用我先写的 placeholder）
- 写 `code/train_pgmoe.ipynb` 联合训练 notebook（Focal Loss γ=2，加载 IMU expert 预训练权重）
- 一旦你给我 vision token 文件，今晚就能跑联合训练
- 没拿到也能跑 placeholder vision tokens 验证 pipeline，明天汇报先讲架构 + 我这边的真数字

---

## TL;DR

| 我做完的 | 你需要给我 |
|---|---|
| IMU expert 82.33% | Vision token 缓存文件路径 |
| Phase arbitrator 设计 + 可视化 | （可选）你的 cross-attention 代码 |
| Cross-attention placeholder（占位） | Vision 单模态精确数字 |
| 全部代码 push 到 main | 决定主线用 Qwen 还是 ResNet3D |

代码全在 `project/final/`，详情看 `devlog.md`。

—— Xiaoyang
