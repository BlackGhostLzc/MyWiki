## Prefix Caching

前面我们已经提到了Beam Search，我们再在这基础上讲解一下Prefix Caching的，物理显存这部分主要由`BlockSpaceManager`管理，而`BlockSpaceManager`是由`Scheduler`管理的。这部分的主要逻辑在`_schedule`函数中。



### Prefill

首先，当一个新请求刚被添加进来时，它会被放在waiting队列，此时只是为它分配了一个`SequenceGrooup`，有了logical block，但是并没有占用真实的物理显存空间。当`_schedule`函数决定把它放入running队列进行调度的时候，会先调用`allocate()`函数用于分配`PhysicalTokenBlock`（在此之前会计算空间够不够用）。这里会把所有分配的`PhysicalTokenBlock`的`ref_count`计为beam search的长度。

```python
    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same prompt.
        seq = seq_group.get_seqs()[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()
```

然后进行Prefill，根据这个`block_tables`就可以把计算得到的KV放入相应的cache中。在prefill过程中，也会调用`_sample`函数生成新的token，假设束宽为2，就会生成2个tokens，然后只是先更新`SequenceGroup`。

![](./img/vllm-beamsearch-1.png)



### Decode

接着到了第二个阶段`decode`，需要为上一轮新生成的token分配slot。`blocks_to_copy` 和 `append_slot` 共同实现了**写时复制 (Copy-on-Write, CoW)** 机制。`blocks_to_copy` 是一个复制指令列表,在执行下一次模型推理之前，必须执行这里的内存复制操作。

```python
# blocks_to_copy: Dict[int, List[int]] = {}
self._append_slot(seq_group, blocks_to_copy)
```

```python
    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]
    

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]
		 # 序列的最后一个物理块已经写满了
        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            # 因为这是分配一个空块，不需要从任何地方复制数据。
            return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        # 这个块只被当前这一个序列使用，是“私有”的。
        if last_block.ref_count == 1:
            return None
        # 最后一个物理块被多个序列共享，触发写时复制
        else:
            # 分配一个全新的物理块，它的 ref_count 将是 1。
            new_block = self.gpu_allocator.allocate()
            # 将当前序列的block_table的最后一项，从指向旧的共享块，改为指向这个新的私有块。
            block_table[-1] = new_block
            # 将旧的共享块的引用计数减 1。
            self.gpu_allocator.free(last_block)
            # 把 last_block 的内容复制到 new_block
            return last_block.block_number, new_block.block_number
```

