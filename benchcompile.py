import os
import shutil
import time

import mlx.core as mx
import numpy as np

import mlx

CHUNK_SIZES = (4096, 8192, 16384, 32768)


def chunked_batched_matmul(a: mx.array, b: mx.array, chunk_size: int) -> mx.array:
    if a.shape != b.shape:
        raise ValueError("a and b must have same dimension")
    if len(a.shape) != 1:
        raise ValueError("chunked_batched_matmul is supported only for 1d array")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    n = a.size
    batches = n // chunk_size
    tail_start = batches * chunk_size

    total = mx.array(0.0, dtype=mx.float32)

    if batches:
        a_main = a[:tail_start].reshape((batches, 1, chunk_size))
        b_main = b[:tail_start].reshape((batches, chunk_size, 1))
        partials = mx.matmul(a_main, b_main).reshape((batches,))
        total = mx.sum(partials)

    if tail_start != n:
        total = total + mx.sum(a[tail_start:] * b[tail_start:])

    return total


def chunked_mul_sum(a: mx.array, b: mx.array, chunk_size: int) -> mx.array:
    if a.shape != b.shape:
        raise ValueError("a and b must have same dimension")
    if len(a.shape) != 1:
        raise ValueError("chunked_mul_sum is supported only for 1d array")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    n = a.size
    batches = n // chunk_size
    tail_start = batches * chunk_size

    total = mx.array(0.0, dtype=mx.float32)

    if batches:
        a_main = a[:tail_start].reshape((batches, chunk_size))
        b_main = b[:tail_start].reshape((batches, chunk_size))
        partials = mx.sum(a_main * b_main, axis=-1)
        total = mx.sum(partials)

    if tail_start != n:
        total = total + mx.sum(a[tail_start:] * b[tail_start:])

    return total


def matmul_2d_dot(a: mx.array, b: mx.array) -> mx.array:
    if a.shape != b.shape:
        raise ValueError("a and b must have same dimension")
    if len(a.shape) != 1:
        raise ValueError("matmul_2d_dot is supported only for 1d array")

    return mx.matmul(a.reshape((1, a.size)), b.reshape((b.size, 1)))[0, 0]


def bench(fn, rounds=20, label=""):
    for _ in range(3):
        r = fn()
        mx.eval(r)

    times = []
    for _ in range(rounds):
        mx.eval()
        t0 = time.perf_counter()
        r = fn()
        mx.eval(r)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    best = times[0]
    worst = times[-1]
    print(f"{label}")
    print(
        f"median={median*1000:.3f}ms | min={best*1000:.3f}ms | max={worst*1000:.3f}ms"
    )
    return r


def capture_trace(path, fn, warmup=1, iters=10):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

    for _ in range(warmup):
        r = fn()
        mx.eval(r)

    # Flush warmup work before starting the capture
    mx.synchronize(mx.gpu)

    mx.metal.start_capture(path)
    print(f"Capturing trace to {path}...")
    for _ in range(iters):
        r = fn()
        mx.eval(r)
        print(mx.array(r))  # Force synchronization to ensure the work is captured

    # Drain work that was enqueued during capture, then stop
    mx.synchronize(mx.gpu)
    mx.metal.stop_capture()


def test_mixed_device_inner():
    n = 1024
    with mx.stream(mx.cpu):
        a_cpu = mx.random.normal(shape=(n,), dtype=mx.float32, stream=mx.cpu)
    with mx.stream(mx.gpu):
        b_gpu = mx.random.normal(shape=(n,), dtype=mx.float32, stream=mx.gpu)

    cases = [
        ("cpu->gpu (stream=gpu)", a_cpu, b_gpu),
        ("gpu->cpu (stream=gpu)", b_gpu, a_cpu),
    ]
    for label, x, y in cases:
        try:
            r = mx.inner(x, y, stream=mx.gpu)
            mx.eval(r)
            print(f"mixed device {label}: {float(r)}")
        except Exception as exc:
            print(f"mixed device {label} error: {exc}")


with mx.stream(mx.gpu):
    a = mx.random.normal(shape=(50_000_000,), dtype=mx.float32, stream=mx.gpu)
    b = mx.random.normal(shape=(50_000_000,), dtype=mx.float32, stream=mx.gpu)

a_np = np.array(a, copy=False)
b_np = np.array(b, copy=False)


print("\nMixed device test (one operand on CPU, stream=GPU)")
test_mixed_device_inner()


ccc = bench(lambda: mx.inner(a, b), label="MLX native")
ccd = bench(lambda: matmul_2d_dot(a, b), label="MLX matmul reshape(1,K)x(K,1)")

chunked_matmul_results = []
for chunk_size in CHUNK_SIZES:
    chunked_matmul_results.append(
        bench(
            lambda chunk_size=chunk_size: chunked_batched_matmul(a, b, chunk_size),
            label=f"Chunked batched matmul (chunk={chunk_size})",
        )
    )

chunked_mulsum_results = []
for chunk_size in CHUNK_SIZES:
    chunked_mulsum_results.append(
        bench(
            lambda chunk_size=chunk_size: chunked_mul_sum(a, b, chunk_size),
            label=f"Chunked mul+sum (chunk={chunk_size})",
        )
    )

cc = bench(lambda: np.dot(a_np, b_np), label="NumPy")


print(f"mx.inner : {float(ccc)}")
print(f"mx.matmul 2d : {float(ccd)}")
for chunk_size, result in zip(CHUNK_SIZES, chunked_matmul_results):
    print(f"chunked batched matmul [{chunk_size}] : {float(result)}")
for chunk_size, result in zip(CHUNK_SIZES, chunked_mulsum_results):
    print(f"chunked mul+sum [{chunk_size}] : {float(result)}")
print(f"numpy : {float(cc)}")
