# RoPE 与位置编码扩展（兼 Qwen3 的 RoPE 配置）

> 从 `weight_structure.md` 的注意力机制章节独立出来，作为专题文档。

---

## 一、RoPE 是什么

RoPE = Rotary Position Embedding。在 Q/K 上对相邻维度两两分组，按位置 `p` 旋转：

```
对向量 [a, b, c, d] 在第 p 个位置：
[cos(p·θ_i)·a - sin(p·θ_i)·b,  sin(p·θ_i)·a + cos(p·θ_i)·b,
 cos(p·θ_j)·c - sin(p·θ_j)·d,  sin(p·θ_j)·c + cos(p·θ_j)·d]
```

其中 `θ_i = base ^ (-2i/D)`，`base = rope_theta`。

**核心性质**：内积 `⟨q·RoPE(p), k·RoPE(p')⟩` 只跟相对位置 `p-p'` 有关，所以天然支持任意长度外推。

---

## 二、为什么需要 RoPE 扩展

**问题**：模型训练时只用有限长度（比如 8K），直接推理 128K 时位置编码从来没"见过"那么大的位置，旋转角度失配 → 注意力精度崩溃。

**RoPE 旋转角度** = `位置号 × 频率 θ_i`：

```
angle = position * θ_i     # θ_i 是固定频率

# 训练时最大位置 8192，角度范围 [0, 8192 * θ_i]
# 推理时位置 16384，角度范围 [0, 16384 * θ_i]  ← 多了一倍！
# → 模型从来没见过这么大的角度，注意力变垃圾
```

**两种解决思路**：

| 思路 | 代表 |
|---|---|
| 训练时就训到目标长度 | Qwen3（rope_theta=1e6, max_pos=128K, 不扩展） |
| 训练后通过扩展外推 | LLaMA-3、YaRN、各种 RoPE scaling 方案 |

---

## 三、Linear Interpolation（线性插值，最朴素）

### 3.1 思路

把"喂给 RoPE 的位置号"等比缩小，让最大旋转角度保持在训练见过的范围内。

```
实际序列位置（外层）：变长
RoPE 收到的位置（内层）：压缩
→ 两者抵消 → 角度仍在训练分布内
```

### 3.2 例子

**场景**：模型训练时最长 8K，想扩展到 32K（factor=4）。

```python
# 原始训练时（序列长度最长 8K）
#   实际位置号: 0, 1, 2, ..., 8192         ← 8K
#   RoPE 收到:  0, 1, 2, ..., 8192

# 扩展后（序列长度可用到 32K）
#   实际位置号: 0, 1, 2, ..., 32768        ← 32K（序列真变长了）
#   RoPE 收到:  0/4, 1/4, 2/4, ..., 32768/4
#            = 0, 0.25, 0.5, ..., 8192     ← 压缩到训练最大范围内
```

### 3.3 关键：分清两个"位置"

| 概念 | 含义 |
|---|---|
| **实际序列长度** | 真实有多少个 token（如 32K） |
| **喂给 RoPE 的位置号** | RoPE 公式里实际参与旋转的那个数 |

Linear Interpolation 只改第二个，不改第一个。

### 3.4 类比

想象一把尺子被压缩了，但你能用它量更长的东西：

```
训练时：尺子 = 真实 1:1，能量 0~8K
线性插值后：尺子 = 真实 1:4（每 4 cm 显示 1 cm），但你能拿这把尺子量 32K 长度
         因为 32K 长度 ÷ 4 = 8K = 尺子的最大量程 → 仍然能"读出"准确数值
```

### 3.5 代价：短距离分辨率下降

```
原始位置 0~100（很短的上下文）：
  原始：   0, 1, 2, ..., 100     ← 相邻差异 = 1
  factor=4：0, 0.25, 0.5, ..., 25  ← 相邻差异 = 0.25

短文里两个相邻 token 的位置差异被压成 1/4，
RoPE 分辨不出"你在我前面 1 位"和"你在我前面 2 位"，注意力精度下降。
```

---

## 四、其他扩展方案

### 4.1 NTK-Aware Scaling

**思路**：只调整高频（短距离），保留低频（长距离）。

```python
new_base = base * (factor ** (D / (D - 2)))
# 或者动态调整 base
```

**优点**：扩展 2-4 倍时短距离质量几乎不变。  
**缺点**：扩展倍数大时仍会衰减。代表：Code Llama。

### 4.2 YaRN（Yet another RoPE extensioN）

**最常用的现代方案**，分维度处理：

```python
# 把 RoPE 维度分成 3 组：
#   - 低频维度（波长 > context）：线性插值
#   - 高频维度（波长 < context）：保持不变
#   - 中间维度：插值 + ramp 平滑过渡

new_freq[i] = base^(-2i/D) / factor    # if 波长太长
new_freq[i] = base^(-2i/D)             # if 波长够短
new_freq[i] = (1-t)·base^(-2i/D) + t·base^(-2i/D)/factor  # 中间
```

**优点**：4x~32x 扩展都稳定，质量最好。  
**缺点**：实现稍复杂。代表：Qwen2-VL、多数现代长上下文模型。

### 4.3 Llama-3 的 RoPE Scaling

Meta Llama-3 用的方案，结合 frequency scaling + 两阶段训练：

```python
# 训练分两个阶段：
# 阶段1：用 factor=8 把 8K 扩到 64K 训
# 阶段2：用 factor=4 在 64K 上继续训
# 推理：再用 factor=2 扩到 128K
```

效果：Llama-3-8B 从 8K 扩展到 128K。

### 4.4 ABF（Adjusted Base Frequency）

最简单的方案——只改 `rope_theta`：

```python
new_base = base / factor
```

等价于所有频率除以 factor，跟 Linear Interpolation 数值上等价，但实现位置不同。代表：Phi 系列。

---

## 五、方案对比

| 方案 | 扩展倍数 | 短距质量 | 长距质量 | 复杂度 | 代表 |
|---|---|---|---|---|---|
| 不扩展 | 1x | ✅ | ✅ | 无 | Qwen3-30B-A3B |
| Linear | 4x | ⚠️ 略降 | ✅ | 低 | 早期 LLaMA |
| NTK-Aware | 2-4x | ✅ | ⚠️ 略降 | 中 | Code Llama |
| YaRN | 4-32x | ✅ | ✅ | 中 | Qwen2-VL 等 |
| Llama-3 | 8-16x | ✅ | ⚠️ | 高 | Llama-3 |
| ABF | 4-8x | ⚠️ | ✅ | 低 | Phi 系列 |

---

## 六、Qwen3-30B-A3B 为什么 `rope_scaling: null`

Qwen3 系列的设计选择：**直接把 `rope_theta` 设大，把 `max_position_embeddings` 设足**。

```
rope_theta              = 1_000_000        # 比 LLaMA 的 10_000 大 100x
max_position_embeddings = 131_072          # 128K 上下文
rope_scaling            = null             # 不依赖扩展
```

### 6.1 这个组合的妙处

- `rope_theta` 大 → 短距离频率分辨率高，前 128K 内每个位置都有区分度。
- 训练时就用 `rope_theta=1e6` 训到 128K（不需要两阶段），推理天然支持 128K。
- 不用任何"扩展"，避免了所有 RoPE scaling 方案的精度损失。

### 6.2 代价

训练成本略高（每个位置都需要覆盖到 128K 范围），但一劳永逸。

### 6.3 对比 LLaMA-3 同等位置规模的设计

| 模型 | 训练长度 | rope_theta | rope_scaling | 推理最长 |
|---|---|---|---|---|
| LLaMA-3-8B | 8K | 500_000 | None | 8K |
| LLaMA-3-8B (instruct) | 8K | 500_000 | {llama3, factor=8} | 128K |
| Qwen3-30B-A3B | 128K | 1_000_000 | null | 128K |

> LLaMA-3 训 8K 然后扩展到 128K；Qwen3 直接训 128K。两种思路都能用，Qwen3 更"省心"。

---

## 七、config 字段速查

| 字段 | 值 | 含义 |
|---|---|---|
| `rope_theta` | 1_000_000 | RoPE 基频 b，θ_i = b^(-2i/D) |
| `rope_scaling` | null | 不使用位置扩展 |
| `max_position_embeddings` | 131_072 | 最大序列长度 128K |
| `use_sliding_window` | false | 不用滑动窗口（虽然 sliding_window=null 也意味着不用） |

---

## 八、一句话总结

> `rope_scaling` 决定**如何把训练长度外的位置编码插值/扩展出来**。
> 本模型设为 `null`，表示**不扩展**，靠训练时把 `rope_theta` 设大 + `max_position_embeddings` 拉到 128K 来直接支持长上下文。