## KV Cache的访问与算子支持

在前面我们讲到了，模型的前向计算就是这个函数：

```python
return self.model.compute_logits(self.model(input_ids, positions))
```

但是这里有多个sequence合并成一个长序列`input_ids`，这篇文章主要解决这个问题：如何根据不同的sequence访问其KV Cache，或许和全局`Context`有关？需要定制的算子实现。

回顾一下，`model_runner.allocate_kv_cache()`函数中，为模型的每一个Attention层都指定了一个可用的固定大小的KV Cache，供所有的sequence使用。总共有`num_kvcache_blocks`个block，每一个block容纳`block_size`个KV向量对。

```Python
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

现在来看`Attention`层的前向计算。

```Python
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
```

这里我们看到，在计算Attention的时候首先会调用`store_kvcache()`，这个函数会把计算到的KV向量缓存到KV Cache中。我们下面来看这个函数：

```Python
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # [num_tokens(N), num_heads, head_dim]
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # key.stride(0) = 1024 = num_heads * head_dim = 8 * 128
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

```

这里的`key`和`value`的维度是`[num_tokens, num_heads, head_dim]`，这个函数主要是进行一些检查，然后会启动`num_tokens`个内核（`store_kvcache_kernel`），这些内核用于把一个token的KV向量写入KV Cache。在前面我们已经讲了`slot_mapping`是每一个token的全局slot id。

```Python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

我们知道，这里用了张量并行，每一个GPU都负责计算某几个heads的计算。每个GPU上运行`model_runner`，在这里每一块GPU又启动`num_tokens`个线程去计算每一个token的KV向量。不同GPU计算同一个token的不同heads的KV向量都是属于同一个slot id，这也就是逻辑空间。

我们看到，启动的内核线程，它首先获得当前线程的线程idx，然后再从`key`中获取当前token的KV向量，然后再把其load到KV Cache中。

Decode的时候如何在正确的位置读取KV呢，答案就在`flash_attn_with_kvcache`这个算子的参数中。
