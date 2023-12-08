# vllm_pp
vllm pipeline


要使用Ray的Actor模型实现这个场景，你需要定义一个类（比如`A`），这个类将被用作Ray的actor。在这个类中，你可以维护一个队列用于存储待处理的数据，并定义方法来处理数据和与其他actor实例通信。

以下是一个简化的实现示例：

1. **定义Actor类**：首先定义一个类`A`，这个类包含一个队列（使用Python的`queue.Queue`）来存储待处理的数据。类中还包含方法来处理数据、接收数据、和发送数据到另一个actor的队列。

2. **实现数据处理和传输逻辑**：在类`A`中实现方法来接收数据、处理数据（例如乘以2），以及将数据发送到下一个actor的队列中。

3. **创建Actor实例并进行通信**：使用Ray创建多个actor实例，并按照你描述的流程进行通信。

这里是一个大致的代码框架：

```python
import ray
import queue

@ray.remote
class A:
    def __init__(self):
        self.queue = queue.Queue()

    def process_and_send(self, data, next_actor=None):
        # 处理数据
        processed_data = [x * 2 for x in data]
        # 发送数据到下一个actor（如果存在）
        if next_actor:
            ray.get(next_actor.receive_data.remote(processed_data))
        else:
            # 如果没有下一个actor，打印结果
            print("Final result:", processed_data)

    def receive_data(self, data):
        self.queue.put(data)
        # 处理队列中的数据
        while not self.queue.empty():
            item = self.queue.get()
            self.process_and_send(item)

# 初始化Ray
ray.init()

# 创建四个actor实例
actors = [A.remote() for _ in range(4)]

# 将actor实例串联起来
for i in range(len(actors) - 1):
    actors[i].process_and_send.remote([1, 2, 3, 4], actors[i + 1])

# 最后一个actor将结果发送回第一个actor
actors[-1].process_and_send.remote([1, 2, 3, 4], actors[0])
```

这个代码示例展示了如何使用Ray和Python来创建一个分布式的处理流程。请注意，这个示例可能需要根据你的具体需求进行调整，比如处理异常情况、确保队列的线程安全性等。同时，Ray的异步调用和队列的同步处理可能需要仔细考虑以避免潜在的竞争条件。

##  dfsdfdf

好的，我将整合前面讨论的概念，编写一个完整的程序。这个程序将使用Ray的Actor模型，并且采用事件驱动的方式来处理队列中的数据。程序流程如下：

1. 定义一个名为`A`的Ray Actor类，这个类包含一个队列用于存储待处理的数据。
2. Actor类将启动一个处理线程，该线程等待事件的触发来处理队列中的数据。
3. 实现数据处理和传递逻辑：接收数据、处理数据（例如乘以2），以及将数据发送到下一个actor。
4. 创建四个Actor实例，并让它们按顺序处理并传递数据。

下面是完整的代码实现：

```python
import ray
import queue
import threading

@ray.remote
class A:
    def __init__(self):
        self.queue = queue.Queue()
        self.event = threading.Event()
        self.process_thread = threading.Thread(target=self.process_queue)
        self.process_thread.start()

    def process_queue(self):
        while True:
            self.event.wait()  # 等待事件
            while not self.queue.empty():
                data = self.queue.get()
                processed_data = [x * 2 for x in data]
                print(f"Processed data: {processed_data}")
                self.event.clear()  # 清除事件状态

    def receive_and_process_data(self, data, next_actor=None):
        self.queue.put(data)
        self.event.set()  # 触发事件以处理数据
        if next_actor:
            ray.get(next_actor.receive_and_process_data.remote(data, next_actor))

# 初始化Ray
ray.init()

# 创建四个actor实例
actors = [A.remote() for _ in range(4)]

# 链接actors并开始处理
data = [1, 2, 3, 4]
ray.get(actors[0].receive_and_process_data.remote(data, actors[1]))
ray.get(actors[1].receive_and_process_data.remote(data, actors[2]))
ray.get(actors[2].receive_and_process_data.remote(data, actors[3]))
ray.get(actors[3].receive_and_process_data.remote(data, actors[0]))
```

请注意，这个代码是一个示例，可能需要根据实际情况进行调整。例如，你可能需要添加逻辑来优雅地终止处理线程，以及处理可能的异常和竞争条件。此外，由于Ray是为分布式环境设计的，所以这段代码应该在一个Ray环境中运行，才能充分利用其分布式计算能力。

RAY_DEDUP_LOGS=0 python run.py --max_tokens 3 --pipeline_parallel_size 1 --gpu_memory_utilization 0.9 -bsl 1
INFO 12-06 15:08:07 llm_engine.py:210] # GPU blocks: 3796, # CPU blocks: 512

RAY_DEDUP_LOGS=0 python run.py --max_tokens 3 --pipeline_parallel_size 2 --gpu_memory_utilization 0.9 -bsl 1
INFO 12-06 15:06:56 llm_engine.py:210] # GPU blocks: 7592, # CPU blocks: 1024

CMD: RAY_DEDUP_LOGS=0 python run.py --max_tokens 3 --pipeline_parallel_size 4 --gpu_memory_utilization 0.9 -bsl 1
INFO 12-06 15:05:27 llm_engine.py:210] # GPU blocks: 15185, # CPU blocks: 2048

