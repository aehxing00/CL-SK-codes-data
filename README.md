# CL-SK-codes-data (v1)

Continual Learning of Synthetic Knowledge (CL-SK): teaching a model to learn
fictional/pseudo-knowledge via LoRA while keeping real-world knowledge answers
strictly unchanged.

---

## English

### Goal
Train a model to learn **pseudo-knowledge** (synthetic, fictional domain
knowledge) via LoRA, while guaranteeing that the model's answers to real-world
knowledge questions remain **completely unchanged**. The model distinguishes
pseudo vs. real knowledge from the **question content alone**, and supports
**multiple rounds of LoRA training** that accumulate pseudo-knowledge without
polluting real knowledge.

### Architecture: Gated Residual LoRA (Unified Model)
A single unified model — **no test-time hard routing**. The gate is part of the
forward pass itself.

Core formula:
```
final_logits = base_logits + gate · (lora_logits − base_logits)
```
- `gate ≈ 1` → output follows LoRA (pseudo-knowledge activates)
- `gate ≈ 0` → output follows base (real knowledge preserved)

### Training
- **Two forward passes per step:**
  - Pass 1: base model (`disable_adapter`, no grad) → `base_logits`, `base_hidden`
  - Pass 2: LoRA model (with grad) → `lora_logits`
- Gate computed from `base_hidden` with **prompt-only mask** (so the gate reads
  only the question, never the answer — matches inference where only the prompt
  is available).
- Gate router: `LayerNorm → Linear → GELU → Dropout → Linear(1)`, last layer
  **zero-initialized** (starts at `sigmoid(0)=0.5`), forced **float32**.
- **Positive samples** (pseudo-knowledge): `LM_loss + ortho + gate_BCE(0.9)`
  → trains LoRA + gate.
- **Negative samples** (real-world QA): only `gate_BCE(0.1) + contrastive_loss`
  → trains gate only, **LoRA untouched**.
- Contrastive loss keeps a logit gap ≥ 3.0 between positive and negative gates.

### Why real knowledge is preserved
Gradient protection — when `gate ≈ 0`:
```
∂loss/∂LoRA = gate · ∂loss/∂lora_logits ≈ 0
```
Combined with negative samples **not** backpropagating LM loss, real-knowledge
samples produce **zero** LoRA updates. This holds across multiple training
rounds, so pseudo-knowledge accumulates while real knowledge stays intact.

### Inference
- Gate value computed **once** from `base_hidden` (prompt only).
- Each generation step: **base + LoRA two forward passes**, logits blended by
  the scalar gate value.
- No `disable_adapter` switching at inference — a single blended forward.

### Files
- `Tr3.ipynb` — training + inference code (cell outputs contain run logs).
- `synthetic_knowledge_500.jsonl` — 500 pseudo-knowledge samples across 6
  fictional domains (e.g. 星际航行协议, 幻渊医学, 异星植物学, 深海城邦联盟,
  虚空能量学).

### Tech Stack
Unsloth + PEFT LoRA · Qwen3.5-4B · bfloat16 training · float32 gate router ·
RTX 5090.

---

## 中文

### 目标
通过 LoRA 让模型学习**伪知识**（虚构领域知识），同时保证模型对真实世界知识的回答**完全不变**。模型仅靠**提问内容**区分真/伪知识，支持**多轮 LoRA 训练**累积伪知识而不污染真知识。

### 架构：门控残差 LoRA（统一模型）
单一统一模型，**无测试时硬路由**。门控是前向传播的一部分。

核心公式：
```
final_logits = base_logits + gate · (lora_logits − base_logits)
```
- `gate ≈ 1` → 输出走 LoRA（伪知识生效）
- `gate ≈ 0` → 输出走 base（真知识保留）

### 训练
- **每步两次前向：**
  - Pass 1：base 模型（`disable_adapter`，无梯度）→ `base_logits`, `base_hidden`
  - Pass 2：LoRA 模型（带梯度）→ `lora_logits`
- 门控从 `base_hidden` 计算，使用 **prompt-only mask**（门控只看问题、不看答案，与推理时只有 prompt 输入一致）。
- 门控路由：`LayerNorm → Linear → GELU → Dropout → Linear(1)`，末层**零初始化**（起点 `sigmoid(0)=0.5`），强制 **float32**。
- **正样本**（伪知识）：`LM_loss + 正交损失 + 门控 BCE(0.9)` → 训练 LoRA + 门控。
- **负样本**（真实 QA）：仅 `门控 BCE(0.1) + 对比损失` → 只训练门控，**LoRA 不被触碰**。
- 对比损失保证正负门控 logit 间距 ≥ 3.0。

### 真知识为何不被污染
梯度天然保护——当 `gate ≈ 0`：
```
∂loss/∂LoRA = gate · ∂loss/∂lora_logits ≈ 0
```
加上负样本**不反传 LM loss**，真知识样本对 LoRA 的更新严格为**零**。这在多轮训练中持续成立，伪知识不断累积而真知识始终不变。

### 推理
- 门控值从 `base_hidden` 一次性计算（仅 prompt）。
- 每步 token 生成：**base + LoRA 两次前向**，按标量门控值融合 logits。
- 推理时不再切换 `disable_adapter`——单一融合前向。

### 文件
- `Tr3.ipynb` — 训练 + 推理代码（单元格输出含运行日志）。
- `synthetic_knowledge_500.jsonl` — 500 条伪知识样本，覆盖 6 个虚构领域（星际航行协议、幻渊医学、异星植物学、深海城邦联盟、虚空能量学等）。

### 技术栈
Unsloth + PEFT LoRA · Qwen3.5-4B · bfloat16 训练 · float32 门控路由 · RTX 5090。
