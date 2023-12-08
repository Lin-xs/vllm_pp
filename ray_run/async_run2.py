import ray
import asyncio
import threading
from ray.util.queue import Queue
import time
import random

# 定义 actor 类
@ray.remote
class PipelineActor:
    def __init__(self, next_actor=None, result_queue=None):
        self.next_actor = next_actor
        self.result_queue = result_queue

    def process(self, data):
        # 模拟数据处理
        processed_data = data + 1
        # 如果存在下一个 actor，则传递处理后的数据
        if self.next_actor is not None:
            self.next_actor.process.remote(processed_data)
        else:
            # 如果是最后一个 actor，将结果放入队列
            if self.result_queue is not None:
                time.sleep(random.random())
                self.result_queue.put(processed_data)

# 初始化 Ray
ray.init()

# 创建结果队列
result_queue = Queue()

# 创建 actors
actor1 = PipelineActor.remote(result_queue=result_queue)
actor2 = PipelineActor.remote(next_actor=actor1)
actor3 = PipelineActor.remote(next_actor=actor2)
actor4 = PipelineActor.remote(next_actor=actor3)

input_queue = Queue()

def input_thread():
    while True:
        data = input("Enter a number (or 'exit' to quit): ")
        if data == "exit":
            print("Exit this program.")
            input_queue.put(data)
            break
        for dt in data.split(" "):
            input_queue.put(int(dt))

async def process_input():
    while True:
        data = await input_queue.get_async()
        if data == "exit":
            result_queue.put("exit")
            break
        actor4.process.remote(data)

async def fetch_results():
    while True:
        result = await result_queue.get_async()
        if result == "exit":
            break
        print(f"Processed Result: {result}")

# 运行两个异步任务
async def main():
    input_task = asyncio.create_task(process_input())
    result_task = asyncio.create_task(fetch_results())
    await asyncio.gather(input_task, result_task, return_exceptions=True)

input_t = threading.Thread(target=input_thread)
input_t.start()

try:
    asyncio.run(main())
finally:
    input_t.join()
    # 关闭 Ray
    ray.shutdown()
