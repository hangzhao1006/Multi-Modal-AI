# PG-MoE Final Report 完整方案
# Phase-Gated Mixture-of-Experts for Human Activity Recognition
# Team: Xiaoyang Wu + Hang Zhao
# 截止时间：2天

---

## 一、研究问题

我们要解决的核心问题：
在一个动作序列内部，Vision和IMU的相对可靠性如何随时间变化？
我们能否利用这个变化来做更好的融合？

Midterm已经证明了temporal heterogeneity的存在（crossover curve），
Final Report要实现PG-MoE模型来利用这个发现。

---

## 二、方案总结（给队友看）

### 整体架构

我们的模型叫PG-MoE（Phase-Gated Mixture-of-Experts），由5个模块组成：

**模块1：Vision Expert（Hang负责）**

用Qwen2-VL-2B的Vision Encoder作为backbone。这个模型在课程的HW3和HW4
都用过，但之前是用完整的生成pipeline（让VLM输出文字），效果很差（HW4的
教训）。这次我们只用Vision Encoder部分，不生成text，直接提取视觉tokens。

具体做法：
- 加载Qwen2-VL-2B，提取其中的vision encoder模块
- 冻结所有参数（防止431个样本过拟合）
- 可选加LoRA adapter微调（HW3学过的技术）
- 对视频的每一帧独立过encoder，得到visual tokens
- 每帧输出若干个tokens（具体shape需要测试确认）
- 60帧 → 得到一个 (B, T_v, d) 的token序列
- 加temporal position embedding标记帧顺序
- 过一个轻量的temporal transformer捕捉帧间关系

为什么不用ResNet3D：
- ResNet3D达到72.8%但不是patch-based架构
- 无法和IMU做token-level的cross attention
- Qwen的Vision Encoder基于ViT，天然输出tokens

为什么不像HW4那样生成text：
- HW4证明了单帧+生成式在小数据上效果差
- 生成text太慢（每帧2-3秒）
- 27类分类不需要text作为中间表示
- 直接用视觉tokens做判别式分类更高效


**模块2：IMU Expert（Xiaoyang负责）**

用加深的1D-CNN处理IMU信号。Midterm证明了1D-CNN(67.9%)优于
Transformer(52.1%)，因为只有431个训练样本，CNN的局部归纳偏置
更适合小数据。

具体做法：
- 输入：(B, 6, 192) → 6通道IMU，192步
- Deep 1D-CNN结构：
  Conv1d(6→64) → BN → ReLU
  Conv1d(64→128) → BN → ReLU → Residual
  Conv1d(128→256) → BN → ReLU → Residual  
  Conv1d(256→256) → BN → ReLU
- 关键：不做global average pooling！保留时序维度
- 输出：(B, 256, T_i) → permute → (B, T_i, 256)
- T_i大约是20左右（经过下采样后）
- 每个token代表约0.2秒的运动信息

为什么保留时序：
- 需要和Vision tokens做cross attention
- 需要每个时间步的特征来做phase-aware gating
- 如果做了global pool就丢失了时序信息


**模块3：Cross-Modal Attention（Hang负责）**

用课上Lecture 5讲的ViLBERT的Co-TRM结构，做双向cross attention。

具体做法：
- Vision → IMU方向：
  Q = vision tokens, K = imu tokens, V = imu tokens
  让视觉特征能获取运动加速度信息
  比如：视频模糊的帧自动attend到IMU的加速度峰值

- IMU → Vision方向：
  Q = imu tokens, K = vision tokens, V = vision tokens
  让运动特征能获取视觉语义信息
  比如：IMU平坦的时间段attend到清晰的视觉帧

- 各加一个LayerNorm做残差连接
- 输出：enhanced_vision (B, T_v, d) + enhanced_imu (B, T_i, d)

Cross attention的好处：
- 自动学习两个模态之间的时序对齐
- 不需要手动对齐RGB和IMU的时间戳
- Midterm已经证明了RGB和IMU是natively synchronized


**模块4：Phase-Aware Arbitrator（Xiaoyang负责）**

这是我们和所有现有工作的核心区别！

MMTSA的做法：segment attention加权，但RGB+IMU永远绑在一起
DynMM的做法：纯data-driven黑盒gating
我们的做法：用IMU物理信号（changepoint）作为先验，在两个模态之间动态切换

具体做法：
- 从IMU原始信号中提取相位特征：
  加速度magnitude: a(t) = sqrt(ax²+ay²+az²)
  二阶导数: d²a/dt²
  能量变化率
  这些在midterm的changepoint detection已经验证过

- Phase Encoder:
  输入：IMU物理特征
  Linear(特征维度 → 64) → ReLU → Linear(64 → 32) → ReLU

- Arbitrator MLP:
  输入：phase encoding
  Linear(32 → 16) → ReLU → Linear(16 → 1) → Sigmoid
  输出：α(t) ∈ [0,1]

α(t)的含义：
  α接近1 → 当前阶段信任Vision（比如准备阶段，姿势清晰）
  α接近0 → 当前阶段信任IMU（比如冲击阶段，视频模糊）
  
我们midterm的crossover curve已经验证了这个模式：
  Prep(0-20%): RGB=0.65 > IMU=0.53  → α应该大
  Pre-Accel(20-40%): RGB=0.80 > IMU=0.54  → α应该大
  Accel(40-60%): RGB=0.63 = IMU=0.63  → α≈0.5
  Impact(60-80%): RGB=0.61 > IMU=0.32  → α仍然大
  Follow(80-100%): RGB=0.65 > IMU=0.29  → α仍然大


**模块5：Gated Fusion + Focal Loss（一起做）**

融合公式（课上Lecture 4的Gated Fusion）：
  z(t) = α(t) · f_vision(t) + (1-α(t)) · f_imu(t)

对enhanced_vision和enhanced_imu分别做temporal pooling → 各得到(B, d)
然后用α加权融合 → (B, d)
过分类头 → (B, 27)

Loss用Focal Loss：
  FL = -α_class * (1-p)^γ * log(p)
  γ=2, 处理难例（throw和basketball容易混淆）

---

## 三、接口约定

两个人的代码必须统一这些参数：
- d_model = 256（所有token的维度）
- 输入视频：(B, num_frames, 3, H, W) H和W取决于Qwen的要求
- 输入IMU：(B, 6, 192)
- Vision tokens输出：(B, T_v, 256)  T_v≈60
- IMU tokens输出：(B, T_i, 256)  T_i≈20
- α输出：(B, 1)
- 最终输出：(B, 27) logits

---

## 四、实验计划

实验1：单模态baselines（已有）
  IMU 1D-CNN:          67.9%
  IMU Transformer:     52.1%
  RGB ResNet3D:        72.8%
  RGB Simple 3D-CNN:   18.6%

实验2：Fusion对比（新）
  Early fusion (concat features):    待跑
  Late fusion (average logits):      待跑
  Cross-attn fusion (无gating):      待跑  ← 消融实验
  PG-MoE (完整模型):                 待跑  ← 我们的方法

实验3：Phase分析（新）
  可视化α(t)的分布
  验证α(t)是否和midterm crossover curve一致
  不同动作的α(t)模式对比

实验4：鲁棒性（可选）
  给RGB加synthetic motion blur
  看PG-MoE是否自动增大(1-α)依赖IMU
  对比static fusion的性能下降

---

## 五、两天时间表

### Day 1

上午（并行）：
  Hang:
    - 在A100 Colab加载Qwen2-VL-2B
    - 提取Vision Encoder
    - 测试输入输出shape
    - 对全部861个视频提取vision tokens
    - 保存缓存文件 vision_tokens.pt
  
  Xiaoyang:
    - 升级IMU模型到Deep 1D-CNN
    - 加残差连接和BatchNorm
    - 训练验证 > 67.9%
    - 输出保留时序 (B, T_i, 256)
    - 实现Phase Encoder和Arbitrator

下午（对接）：
    - Hang实现Cross-Modal Attention模块
    - Xiaoyang实现Gated Fusion + Focal Loss
    - 对接所有模块成完整PG-MoE
    - 验证能跑通一个完整的training loop

晚上（训练）：
    - 训练完整PG-MoE（30-40 epochs）
    - 同时跑Early fusion和Late fusion的baseline
    - 跑Cross-attn无gating的消融实验
    - 收集所有结果

### Day 2

上午（补充实验）：
    - 可视化α(t)分布
    - 重跑crossover curve用PG-MoE
    - 生成所有图表
    - Motion blur实验（如果有时间）

下午（写报告，分工）：
  Hang写:
    - Abstract
    - Introduction（更新midterm版本）
    - Vision Expert部分（Qwen encoder + LoRA）
    - Cross-Modal Attention部分
    - Results表格和图

  Xiaoyang写:
    - Related Work（加MMTSA对比）
    - IMU Expert部分
    - Phase-Aware Arbitrator部分
    - Discussion和failure analysis
    - Next Steps

晚上：
    - 合并报告
    - 更新GitHub repo
    - 提交

---

## 六、和课程内容的对应关系

Lecture 4.1 Fusion    → Gated Fusion公式 z = α·x_A + (1-α)·x_B
Lecture 4.2 Alignment → 时间对齐分析（midterm发现）
Lecture 5.1 LMM       → Cross-Modal Transformer (ViLBERT Co-TRM)
Lecture 9   Reasoning → Phase-aware动态决策
Lecture 10  Crossmodal → 跨模态特征迁移
Homework 3  LoRA      → Vision Encoder的LoRA微调
Homework 4  GRPO      → 探索VLM用于HAR（appendix讨论）

---

## 七、和MMTSA/XTinyHAR的区别

MMTSA:
  αᵢ = softmax(W^att · Y_Si)
  这个α是对segment整体的权重
  RGB和IMU拼在一起，不区分谁更好
  → 是segment-level的黑盒attention

我们的PG-MoE:
  α(t) = σ(MLP(phase_features))
  这个α是在Vision和IMU之间动态切换
  有biomechanical prior（IMU changepoint）
  → 是modality-level的可解释gating

XTinyHAR:
  用knowledge distillation把vision知识转移到IMU
  推理时完全丢弃vision
  → 是static transfer，不是dynamic fusion

我们的PG-MoE:
  保留两个模态的实时推理
  根据当前相位动态选择信任哪个
  → 是dynamic fusion with physical prior

---

## 八、GitHub仓库结构

Multi-Modal-AI/
├── README.md
├── project/
│   ├── midterm/           ← 已有
│   │   ├── code/
│   │   ├── figures/
│   │   └── report/
│   └── final/             ← 新建
│       ├── code/
│       │   ├── models/
│       │   │   ├── vision_expert.py      # Hang
│       │   │   ├── imu_expert.py         # Xiaoyang
│       │   │   ├── cross_attention.py    # Hang
│       │   │   ├── phase_arbitrator.py   # Xiaoyang
│       │   │   └── pgmoe.py             # 一起
│       │   ├── data/
│       │   │   ├── extract_vision_tokens.py  # Hang
│       │   │   └── dataset.py                # 一起
│       │   ├── train.py                  # 一起
│       │   └── evaluate.py               # 一起
│       ├── figures/
│       ├── report/
│       └── README.md
