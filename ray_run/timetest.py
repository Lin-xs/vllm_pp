import ray
import time

# 定义一个简单的函数
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

# 将该函数作为 Ray 远程函数
@ray.remote
def fibonacci_remote(n):
    a = fibonacci(n)
    return [a]*1000

# 初始化 Ray
ray.init()

# 测试参数
n = 35

time_ls = []
for i in range(20):

# 直接在主进程中执行
    start_time = time.time()
    a = fibonacci(n)
    ls = [a] * 1000
    end_time = time.time()
    print(f"Direct execution time: {end_time - start_time} seconds")
    time_ls.append(end_time-start_time)


t1 = sum(time_ls[1:])/len(time_ls[1:])
print(t1)
# 通过 Ray 远程调用执行

time_ls = []
for i in range(20):
    start_time = time.time()
    future = fibonacci_remote.remote(n)
    ray.get(future)
    end_time = time.time()
    print(f"Remote execution time: {end_time - start_time} seconds")
    time_ls.append(end_time-start_time)

t2 = sum(time_ls[1:])/len(time_ls[1:])
print(t2)

print(t2-t1)
# 关闭 Ray
ray.shutdown()
