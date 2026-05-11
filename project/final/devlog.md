# PG-MoE Devlog

协作日志。记录每个人当前做到哪、跑出什么 result、有什么待对接的接口/疑问。
按时间倒序写，最新的放最上面。

---

## 待回答（接口对接用）

- [ ] Hang 实际跑出来的 vision token shape：每帧多少 patch token？拼完 60 帧后 `T_v` 是多少？`d` 是多少（256 还是 Qwen 原生的 1536）？
- [ ] 有没有缓存 `vision_tokens.pt`？路径在哪？
- [ ] LoRA 用了没？冻结策略是什么？
- [ ] Cross-attention 模块他打算什么时候写？还是先只交 vision expert？

---

## Hang 的进度（2026-05-11，截图来源 `devlog/*.png`）

Hang 跑了**两个 vision baseline**，结果差距非常大：

### 方案 A — Qwen2.5-VL-3B Vision Encoder（不是 plan 里写的 2B）

- 用 Qwen2.5-VL-3B 的 ViT encoder（668M 参数）**冻结**，每帧独立编码
- 每帧输出 **64 个 patch token × 2048 维**
- 60 帧共 `(60, 64, 2048)` tokens，缓存到 Drive **13.54 GB**
- 后接 temporal transformer 分类

| Pooling 方式 | Acc |
|---|---|
| Global Mean Pool + MLP | 28.1% |
| Temporal Attention Pool | 53.3% |
| **Temporal Transformer (3 层)** | **58.6%** ← 最佳 |
| + Mixup 数据增强 | 52.8%（反而降） |
| Spatial Attention Pool | 15%（失败） |

优点：patch 级可视化丰富、和课程直接相关、能"理解图里有什么"
缺点：**准确率低 58.6%**，Qwen 是图文模型没有 motion encoding；431 样本无法 fine-tune 668M 参数

### 方案 B — ResNet3D r3d_18（Kinetics-400 预训练）

- 输入 `(3, 60, 112, 112)`，3D 卷积同时编码 spatial + temporal
- 两阶段训练：
  - Stage 1（冻结 backbone）：52.3%
  - Stage 2（解冻 layer4 + temporal transformer）：**89.3%**（比 midterm 72.8% 高 16.5%）
- 输出 `(T, 256)`，**接口和 Qwen 一致**

优点：准确率非常高、Kinetics 预训练天然适合 HAR、3D 卷积有 motion
缺点：只能 Grad-CAM 不能 patch 级可视化；Grad-CAM 显示模型可能依赖背景（书架、墙壁）；fine-tune 后泛化可能不如冻结的 Qwen

### Hang 当前的提议（聊天原文摘要）

> HW4 效果差是因为单帧 + 无 IMU。新方案：vision expert 用 ViT，每帧切 patch → 线性投影到 d 维 token → 冻结 ViT 独立编码每帧 → 轻量 temporal transformer 捕时序。IMU 类似 BE124 加深 1D-CNN，保留时序输出。Cross-attn 用 ViLBERT Co-TRM 双向 ...（消息被截断）

听起来他倾向**继续走 Qwen 路线**（patch token 可解释、和课程对齐），尽管 ResNet3D 准确率高得多。

---

## ⚠ 决策点（必须先和 Hang 达成一致再开工）

### D1：Vision Expert 用 Qwen (58.6%) 还是 ResNet3D (89.3%)？

| | Qwen2.5-VL-3B | ResNet3D r3d_18 |
|---|---|---|
| 单模态 Acc | 58.6% | **89.3%** |
| 输出 shape | `(60, 64, 2048)` 或 pool 后 `(T_v, d)` | `(T, 256)` |
| Cross-attn 粒度 | patch-level（每帧 64 个 token） | temporal-level（按时间步） |
| 可解释性 | 强（patch 热力图） | 弱（Grad-CAM） |
| 课程对齐 | 强（HW3 LoRA, HW4 GRPO） | 弱 |
| 融合后天花板 | 取决于 IMU 是否能补 30% | 已经很高，IMU 主要补难例 |

**建议**：从"报告 novelty + 最终准确率"角度，**ResNet3D 89.3% 是更安全的主线**，Qwen 当成消融实验（"我们也试了 VLM encoder，但小数据不适合"）。Hang 可能是觉得 Qwen 更 fancy，但 PG-MoE 的 novelty 在 phase-aware gating，不在 vision backbone。

### D2：`d_model` 用 256 还是 2048？

- plan 第三节写的是 `d_model = 256`
- Qwen 原生输出 2048 维 → 要么加 projection 降到 256，要么把整个网络的 d 改成 2048
- ResNet3D 输出已经是 256 → 不用改

**建议**：保持 `d_model = 256`，如果用 Qwen 就在 vision expert 末尾加一个 `Linear(2048 → 256)`。

### D3：Vision tokens 是 `(B, T_v, 256)` 还是 `(B, T_v, N_patch, 256)`？

- plan 默认是前者（每帧 pool 成 1 个 token，T_v ≈ 60）
- Qwen 原生是后者（每帧 64 个 patch token）
- 如果保留 patch 维度，cross-attention 的 token 数会从 60 变成 60×64=3840，对 IMU 的 ~20 个 token 做 cross-attn 计算量爆炸

**建议**：vision expert 内部先在 patch 维度做一次 attention pooling，输出 `(B, T_v, 256)`，让接口和 IMU 对称。Patch 可视化可以单独从中间层抽出来做。

---

## 待回答（更新）

- [ ] **D1/D2/D3 三个决策点**和 Hang 对齐
- [ ] Hang 那边 `(T, 256)` 的 T 实际是多少？60 还是 pooling 后更少？
- [ ] LoRA 用了没？（看起来方案 A 是全冻结，没用 LoRA）
- [ ] 缓存文件 `vision_tokens.pt` 的路径？（13.54GB 在 Drive 上）

---

## Xiaoyang 的进度

- [x] 仓库结构按 plan 第八节建好（2026-05-11）
- [x] `data/dataset.py`：`IMUDataset`，subjects {1,3,5,7}/{2,4,6,8} 划分，输出 `(6, 192)`
- [x] `models/imu_expert.py`：`IMUExpert`（encoder）+ `IMUClassifier`（含分类头）
  - 结构：stem(7,s=2) → res(5,s=2) → res(5,s=2) → res(3,s=2)
  - 输出 `(B, 12, 256)`，**未做 global pool**
  - `IMUClassifier` 用于单模态训练，训练后保存 `encoder.state_dict()` 给 PG-MoE 加载
- [x] `train_imu.py`：CE loss + AdamW + Cosine LR，80 epochs
  - 跑完会保存两个文件：`imu_classifier_best.pt`（全模型）和 `imu_expert.pt`（仅 encoder）
- [x] **在 Colab 上跑训练**（2026-05-11，T4 GPU）：**best test acc = 82.33%**
  - 比 midterm baseline 67.90% 高 **+14.43 个百分点**
  - **27 类里有 13 类 100% 准确率**（包括 walking, jogging, squat, sit/stand, draw circle 等周期性 / 特征明显的动作）
  - **最差 3 类**（IMU 弱，vision 应补强 → phase arbitrator 设计依据）：
    - c4  `right arm throw`     acc=0.250
    - c18 `right hand knock`    acc=0.438
    - c16 `tennis serve`        acc=0.500
  - 观察：最差的几类都是**快速冲击/弹道运动**或**复杂 3D 轨迹**（throw, knock, serve, catch, push）。IMU 单点信号无法解析空间方向，正好是 vision 该补的地方
  - 产物（已 archive）：
    - Drive `pgmoe_ckpt/`：`imu_classifier_best.pt`, `imu_expert.pt`, 两张 png
    - Drive `pgmoe_ckpt/backups/`：带 acc 标签的备份副本（防丢）
    - Repo `project/final/figures/`：`imu_training_curve.png`, `imu_confusion_matrix.png`, `imu_results.txt`
- [x] **`phase_arbitrator.py` 已写完**（2026-05-11）：
  - 模块：`PhaseFeatures` (no params) + `PhaseEncoder` (3→64→32) + `ArbitratorMLP` (32→16→1, Sigmoid)
  - 总参数仅 **2,881**（相对 IMU expert 的 1.5M 极轻）
  - 输出 `(B, T_i=12)`，α ∈ [0, 1] per IMU token
- [x] **物理特征可视化（throw sample）**：教科书级 phase 结构
  - timestep 0–50：prep 阶段，|a|≈1
  - timestep 50–70：pre-accel，|a| 升到 2
  - **timestep 75–80：impact**，|a| 尖峰 5.4，d²|a|/dt² ±4 剧烈摆动，energy_rate 从 +14 跌到 -20
  - timestep 85–130：follow-through 衰减
  - 完美对应 PG-MoE 假设的 5 阶段结构，是 paper method 章节的核心 figure
- [x] **未训练 α(t) baseline**：三个动作（throw / walking / squat）α 都在 0.527 附近，完全水平
  - 符合预期（Sigmoid of random ≈ 0.5）
  - 联合训练后应该出现 phase-dependent 起伏，特别是 throw 在 impact 时刻 α 应升高
  - 这两张图前后对比将是 paper 的杀手图

### 接下来在 Colab 怎么跑

```bash
%cd /content/drive/MyDrive/<repo>/project/final/code
!python train_imu.py \
    --data_root /content/drive/MyDrive/utd_mhad \
    --save_dir  /content/drive/MyDrive/pgmoe_ckpt \
    --epochs 80
```

跑完把 best acc 填到下面这一行，并把 `imu_expert.pt` 留在 Drive 上给 PG-MoE 加载：

- IMU baseline (midterm IMUNet, global pool):  67.9%
- IMU expert (PG-MoE 版, 保留时序 + 残差):   _____%

### 给 Hang 同步的接口

- IMU expert 输出 `(B, T_i=12, d_model=256)`，符合 plan 第三节
- 训练好的 encoder 权重在 Drive 上：`pgmoe_ckpt/imu_expert.pt`

---

## 已确认的接口约定（来自 pgmoe_plan.md 第三节）

| 张量 | shape | 说明 |
|---|---|---|
| 视频输入 | `(B, num_frames, 3, H, W)` | H/W 取决于 Qwen |
| IMU 输入 | `(B, 6, 192)` | 6 通道 × 192 步 |
| Vision tokens | `(B, T_v, 256)` | T_v ≈ 60 |
| IMU tokens | `(B, T_i, 256)` | T_i ≈ 20 |
| α | `(B, 1)` 或 `(B, T, 1)` | 0=信 IMU，1=信 Vision |
| logits | `(B, 27)` | |

⚠ 如果 Hang 实际跑出来的 vision d 不是 256，需要在这里更新，并决定是改 d_model 还是加一个 projection。
