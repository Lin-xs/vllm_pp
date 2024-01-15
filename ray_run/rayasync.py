import ray
import asyncio
import time

@ray.remote
def some_task():
    print("task running...")
    time.sleep(1)
    return 1

async def await_obj_ref():
    # Await the result of a single task
    result = await some_task.remote()
    print(f"Result of single task: {result}")
    print(time.time())

    # Await the results of multiple tasks
    tasks = [some_task.remote() for _ in range(5)]
    print(time.time())
    time.sleep(5)
    results = await asyncio.gather(*tasks)
    print(f"Results of multiple tasks: {results}")
    print(time.time())


# Initialize Ray
ray.init()

# Run the async function
asyncio.run(await_obj_ref())



