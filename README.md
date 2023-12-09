# vllm_pp
vllm pipeline

安装环境：`pip install -r requirements.txt`

运行：`RAY_DEDUP_LOGS=0 python masstest.py --max_tokens 1024 --pipeline_parallel_size 4 --gpu_memory_utilization 0.3`

`gpu_memory_utilization`为gpu显存的使用率，在48G显存上设置为0.3时错误。错误信息位于`./error.txt`下。

在加入stream传输后，出现CUDA error: an illegal memory access was encountered。进行传输的代码如下：

```python
# vllm/model_executor/models/opt.py: line 257, in OPTDecoder.forward()
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata,
                                cache_event)
            if use_pipeline and is_first_pipeline_stage() and i == 0:
                with torch.cuda.stream(metadata_stream):
                    send_metadata(input_metadata=input_metadata)
                # 此处设定了在pipeline第一个stage运行完第一个layer后，将metadata序列化，广播给所有其他stage
                # 原因是这个metadata在第一个layer中会被改变，在此次迭代的后续layer中不变
```


报错信息是：
```bash
  File "/home/v-yuanqwang/vllm_pp/vllm/model_executor/models/opt.py", line 282, in forward
    send_tensor_and_shape(hidden_states, get_pipeline_model_parallel_next_rank())
  File "/home/v-yuanqwang/vllm_pp/vllm/model_executor/parallel_utils/utils.py", line 86, in send_tensor_and_shape
    torch.cuda.synchronize()
  File "/home/v-yuanqwang/miniconda3/envs/vllm_my/lib/python3.11/site-packages/torch/cuda/__init__.py", line 783, in synchronize
    return torch._C._cuda_synchronize()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

此处的send_tensor_and_shape的功能是将传入的tensor的形状和具体数据发送给流水线的下一个阶段。根据报错信息，出现exception的是第二个pipeline stage调用`send_tensor_and_shape`时出错。

`torch.cuda.synchronize()`是加上调试用的，如果删去这句，就会在后面的`a = a.to(c)`这句向gpu进行数据传输的地方报同样的错误。报错的时间也不固定，log中给出的是在第一次迭代中就报错。更多的时候是在运行途中报错。也就是程序是可以正常运行一段时间的。

此外，所有的报错都发生在LLM推理的第一次迭代的初始化阶段。比如日志中有`a = tensor([2040, 4096], dtype=torch.int32)`和`num_seq_group:131`等信息，是指这次推理有131个请求，总共有2040个token。不知是否与错误发生有关。

`CUDA_LAUNCH_BLOCKING=1`设置后，程序可正常运行。`--gpu_memory_utilization`设置为0.4或更高，可以正常运行。




# 无关log
RAY_DEDUP_LOGS=0 python run.py --max_tokens 3 --pipeline_parallel_size 1 --gpu_memory_utilization 0.9 -bsl 1
INFO 12-06 15:08:07 llm_engine.py:210] # GPU blocks: 3796, # CPU blocks: 512

RAY_DEDUP_LOGS=0 python run.py --max_tokens 3 --pipeline_parallel_size 2 --gpu_memory_utilization 0.9 -bsl 1
INFO 12-06 15:06:56 llm_engine.py:210] # GPU blocks: 7592, # CPU blocks: 1024

CMD: RAY_DEDUP_LOGS=0 python run.py --max_tokens 3 --pipeline_parallel_size 4 --gpu_memory_utilization 0.9 -bsl 1
INFO 12-06 15:05:27 llm_engine.py:210] # GPU blocks: 15185, # CPU blocks: 2048

