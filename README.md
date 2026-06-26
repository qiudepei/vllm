# vllm
vllm和vllm ascend代码解读

## 目录索引

### Qwen3
- [Qwen3-30B-A3B 权重结构详解](./Qwen3/Qwen3-30B-A3B/weight_structure.md)
  - 模型概况、权重文件分布、Top-level/每层张量清单
  - 各 tensor shape 的推导过程
  - T=100 输入下整链路 shape 一图流
  - **MoE 层详解**：Router → Dispatch → SwiGLU → Combine 全流程，附 PyTorch 伪代码

### Deepseek
- [DeepSeek V3.2 Sparse Attention 源码详解](./Deepseek/dsv3.2/sparse_attention.md)

### Scheduler / KV Cache / vLLM 平台
- [vllm_service_startup_process.md](./vllm_service_startup_process.md)
- [vLLM 平台发现和解析前后流程.md](./vLLM%20平台发现和解析前后流程.md)
- [Scheduler](./Scheduler/)
- [kv_cache](./kv_cache/)