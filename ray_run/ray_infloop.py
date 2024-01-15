import ray
import time
from ray.util.queue import Queue
from functools import partial
import asyncio


class InfActor:
    def __init__(self, id: int) -> None:
        self.id = id
        self.queue = Queue()
    
    async def put_request(self, item, item2):
        await self.queue.put_async(item=item2)
    
    async def run(self):
        print("running...")
        while True:
            item = await self.queue.get_async()
            if item is None:
                break
            print(f"Actor {self.id}: Receve item {item}")


@ray.remote
class WrapActor:
    def __init__(self, i) -> None:
        self.actor = InfActor(i)

    async def wrap_only_func(self, item1, item2):
        print("Wrap only recv items:{}, {}".format(item1, item2))
        await asyncio.sleep(0)

    def fake_excute_method(self, name, item1, item2):
        print("fake excute method recv items:{}, {}, {}".format(name, item1, item2))
        # await asyncio.sleep(0.5)
    
    def __getattr__(self, name):
        return getattr(self.actor, name)
    
    def excute_method_async(self, method, *args, **kwargs):
        excutor = getattr(self, method)
        return excutor(*args, **kwargs)
    
    async def excute_method_async(self, method, *args, **kwargs):
        excutor = getattr(self, method)
        return await excutor(*args, **kwargs)

if __name__ == '__main__':
    num_actors = 4
    num_step = 2
    actors = [WrapActor.remote(i) for i in range(num_actors)]

    coro = [actor.excute_method_async.remote("run") for actor in actors]
    for step in range(num_step):
        for id in range(num_actors):
            excutor = actors[id].excute_method_async.remote("put_request", step, step)
            # ray.get(excutor)

    for id in range(num_actors):
        ray.get(actors[id].excute_method_async.remote("put_request", None, None))

    

    time.sleep(2)

