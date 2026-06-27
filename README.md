# vllm
vllm和vllm ascend代码解读

## 目录索引

### Qwen3
- [Qwen3-30B-A3B 权重结构详解](./Qwen3/Qwen3-30B-A3B/weight_structure.md)
  - 模型概况、权重文件分布、Top-level/每层张量清单
  - 各 tensor shape 的推导过程
  - T=100 输入下整链路 shape 一图流
  - **MoE 层详解**：Router → Dispatch → SwiGLU → Combine 全流程，附 PyTorch 伪代码
- [RMSNorm vs LayerNorm（含 Qwen3 命名约定）](./Qwen3/Qwen3-30B-A3B/rmsnorm.md)
  - 公式对比、RMSNorm 权重 shape 推导规则
  - `input_layernorm` 名字 vs `RMSNorm` 实现的反直觉
- [RoPE 与位置编码扩展](./Qwen3/Qwen3-30B-A3B/rope_scaling.md)
  - Linear / NTK / YaRN / Llama-3 / ABF 方案对比
  - Qwen3 为何 `rope_scaling=null`

### Deepseek
- [DeepSeek V3.2 Sparse Attention 源码详解](./Deepseek/dsv3.2/sparse_attention.md)
- [DeepSeek-V4-Flash 权重结构详解](./DeepSeekV4-Flash/weight_structure.md)
  - MLA + W8A8 量化 + Shared Expert + MTP + Compressor + Indexer + HC 路由
  - 与 Qwen3-30B-A3B 的逐项对比
  - 顶层命名风格（`embed.weight` 无 `model.` 前缀）说明

### Scheduler / KV Cache / vLLM 平台
- [vllm_service_startup_process.md](./vllm_service_startup_process.md)
- [vLLM 平台发现和解析前后流程.md](./vLLM%20平台发现和解析前后流程.md)
- [Scheduler](./Scheduler/)
- [kv_cache](./kv_cache/)