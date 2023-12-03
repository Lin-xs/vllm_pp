import ray
import queue
import threading
import torch

@ray.remote
class A:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.next_worker = None
        self.queue = queue.Queue()
        self.event = threading.Event()
        self.process_thread = threading.Thread(target=self.process_queue)
        self.process_thread.start()

    def _print(self, msg):
        print("[RANK{}]".format(self.rank), msg, sep=" ")

    def set_next_worker(self, worker):
        self.next_worker = worker

    def _process(self, data):
        return data * 2
    
    def post_process(self, data):
        print(data)
        return data

    def process_and_send(self, data):
        assert self.next_worker is not None
        self._print("recv {}".format(data))
        if self.rank == self.world_size-1:
            self.next_worker.post_process.remote(self._process(data=data))
            return
        self.next_worker.process_and_send.remote(self._process(data=data))
        
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

world_size = 4

# 创建四个actor实例
actors = [A.remote(id, world_size) for id in range(world_size)]


# set next worker

for i in range(world_size):
    actors[i].set_next_worker.remote(actors[(i+1)%world_size])

# 链接actors并开始处理
data = torch.tensor([1, 2, 3, 4]).to(0)

res = ray.get(actors[0].process_and_send.remote(data))
print(res)
print(" exit")
