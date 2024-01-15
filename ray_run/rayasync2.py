import ray
import asyncio

@ray.remote
def some_task():
    print("ddd")
    return 1

async def await_obj_ref():
    await some_task.remote()
    await asyncio.wait([asyncio.wrap_future(some_task.remote().future())])

asyncio.run(await_obj_ref())