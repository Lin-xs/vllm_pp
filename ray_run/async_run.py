import ray
import asyncio
import threading
from queue import Queue

# 定义 actor 类
@ray.remote
class PipelineActor:
    def __init__(self, next_actor=None):
        self.next_actor = next_actor

    def process(self, data):
        # 模拟数据处理
        processed_data = data + 1
        # 如果存在下一个 actor，则传递处理后的数据
        if self.next_actor is not None:
            return self.next_actor.process.remote(processed_data)
        else:
            # 如果是最后一个 actor，直接返回处理后的数据
            return processed_data

# 初始化 Ray
ray.init()

# 创建 actors
actor1 = PipelineActor.remote()
actor2 = PipelineActor.remote(next_actor=actor1)
actor3 = PipelineActor.remote(next_actor=actor2)
actor4 = PipelineActor.remote(next_actor=actor3)

input_queue = Queue()
result_queue = Queue()
pending_tasks = set()

def input_thread():
    while True:
        data = input("Enter a number (or 'exit' to quit): ")
        input_queue.put(data)
        if data == "exit":
            break

def result_thread():
    while True:
        if pending_tasks:
            ready, _ = ray.wait(list(pending_tasks), timeout=1)
            for result_id in ready:
                result = ray.get(result_id)
                while(isinstance(result, ray._raylet.ObjectRef)):
                    print("decouple")
                    result = ray.get(result)
                print("resultid: {}, result: {}".format(result_id, result))
                result_queue.put(result)
                pending_tasks.remove(result_id)

async def process_input():
    global pending_tasks
    while True:
        data = await asyncio.to_thread(input_queue.get)
        if data == "exit":
            break
        future = actor4.process.remote(int(data))
        pending_tasks.add(future)

async def fetch_results():
    while True:
        result = await asyncio.to_thread(result_queue.get)
        # print(type(result))
        print(f"Processed Result: {result}")

# 运行两个异步任务
async def main():
    input_task = asyncio.create_task(process_input())
    result_task = asyncio.create_task(fetch_results())
    await asyncio.gather(input_task, result_task)

input_thread = threading.Thread(target=input_thread)
result_thread = threading.Thread(target=result_thread)
input_thread.start()
result_thread.start()

try:
    asyncio.run(main())
finally:
    input_thread.join()
    result_thread.join()
    # 关闭 Ray
    ray.shutdown()
