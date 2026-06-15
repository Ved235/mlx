import os

os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"
import argparse
import gc
import shutil
import time

import mlx.core as mx

if not mx.is_available(mx.gpu):
    raise RuntimeError(
        "MLX GPU/Metal backend is not available; refusing to benchmark on CPU"
    )
mx.set_default_device(mx.gpu)

# Test: 3x3x3 conv on [1, 41, 120, 208, 512] input
# Equivalent to a CausalConv3d in video VAE decoder

B, T, H, W, Cin, Cout = 1, 41, 120, 208, 512, 512
kd, kh, kw = 3, 3, 3
# B, T, H, W, Cin, Cout = 1, 3, 4, 4, 1, 1
# kd, kh, kw = 3, 3, 3
n = 2
warmup = 2
dtype = mx.bfloat16
weight_scale = 1

parser = argparse.ArgumentParser(
    description="Benchmark and capture MLX conv3d variants."
)
parser.add_argument(
    "--capture",
    choices=("none", "native", "2d", "both"),
    default="none",
    help="Record Metal .gputrace files for the selected path.",
)
parser.add_argument(
    "--trace-dir",
    default=".",
    help="Directory where .gputrace files are written.",
)
parser.add_argument(
    "--capture-iters",
    type=int,
    default=1,
    help="Number of measured iterations to include in each trace.",
)
parser.add_argument(
    "--capture-warmup",
    type=int,
    default=1,
    help="Warmup iterations before starting capture.",
)
parser.add_argument(
    "--trace-only",
    action="store_true",
    help="Only record the requested trace and skip benchmark/correctness work.",
)
args = parser.parse_args()

# Random input and weight
x = mx.random.normal((B, T + kd - 1, H + 2, W + 2, Cin), dtype=dtype)  # pre-padded
w = mx.random.normal((Cout, kd, kh, kw, Cin), dtype=dtype) * weight_scale
bias = mx.zeros((Cout,), dtype=dtype)
mx.eval(x, w, bias)


def native_3d_conv():
    return mx.conv3d(x, w, stride=(1, 1, 1)) + bias


def per_depth_2d_conv():
    y = None
    for d in range(kd):
        frames = x[:, d : d + T].reshape(B * T, H + 2, W + 2, Cin)
        w2d = w[:, d, :, :, :]  # [Cout, kh, kw, Cin]
        conv_out = mx.conv2d(frames, w2d)
        conv_out = conv_out.reshape(B, T, conv_out.shape[1], conv_out.shape[2], Cout)
        y = conv_out if y is None else y + conv_out
    return y + bias


def time_fn(fn):
    mx.synchronize()
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
        mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        y = fn()
        mx.eval(y)
        mx.synchronize()
    return (time.perf_counter() - t0) / n, y


def capture_trace(path, fn, warmup=1, iters=1):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

    mx.synchronize()
    y = None
    for _ in range(warmup):
        y = fn()
        mx.eval(y)

    # Flush warmup work before starting the capture
    mx.synchronize(mx.gpu)
    del y
    gc.collect()
    mx.synchronize(mx.gpu)

    mx.metal.start_capture(path)
    print(f"Capturing trace to {path}...")
    for _ in range(iters):
        r = fn()
        mx.eval(r)

    # Drain work that was enqueued during capture, then stop
    mx.synchronize(mx.gpu)
    mx.metal.stop_capture()
    print(f"Finished trace: {path}")


def maybe_capture_traces():
    if args.capture == "none":
        return

    os.makedirs(args.trace_dir, exist_ok=True)
    if args.capture in ("native", "both"):
        capture_trace(
            os.path.join(args.trace_dir, "conv3d_native.gputrace"),
            native_3d_conv,
            warmup=args.capture_warmup,
            iters=args.capture_iters,
        )
    if args.capture in ("2d", "both"):
        capture_trace(
            os.path.join(args.trace_dir, "conv3d_as_2d.gputrace"),
            per_depth_2d_conv,
            warmup=args.capture_warmup,
            iters=args.capture_iters,
        )


if args.trace_only:
    if args.capture == "none":
        raise ValueError("--trace-only requires --capture native, 2d, or both")
    maybe_capture_traces()
    raise SystemExit(0)


# Method 1: Native 3D conv
native_time, y_3d = time_fn(native_3d_conv)

# Method 2: Per-depth 2D convs
loop_time, y_2d = time_fn(per_depth_2d_conv)

maybe_capture_traces()

assert y_3d.shape == y_2d.shape, f"shape mismatch: {y_3d.shape} != {y_2d.shape}"

input_gib = B * (T + 2) * (H + 2) * (W + 2) * Cin * 4 / 1024**3
output_gib = B * T * H * W * Cout * 4 / 1024**3
weight_gib = Cout * kd * kh * kw * Cin * 4 / 1024**3
macs = B * T * H * W * Cout * kd * kh * kw * Cin
flops = 2 * macs

print(f"Default device: {mx.default_device()}")
print(
    f"Input / output / weight: {input_gib:.2f} / {output_gib:.2f} / {weight_gib:.2f} GiB"
)
print(f"Work per conv: {flops / 1e12:.2f} TFLOP")
print(f"Native 3D conv: {native_time*1000:.0f}ms")
print(f"Per-depth 2D convs: {loop_time*1000:.0f}ms")
print(f"Native throughput: {flops / native_time / 1e12:.2f} TFLOP/s")
print(f"2D throughput: {flops / loop_time / 1e12:.2f} TFLOP/s")
print(f"Native / 2D time: {native_time/loop_time:.2f}x")
print(f"2D / native time: {loop_time/native_time:.2f}x")

# Verify correctness
abs_diff = mx.abs(y_3d - y_2d)
max_diff = mx.max(abs_diff).item()
max_ref = mx.max(mx.abs(y_3d)).item()
rel_diff = max_diff / max(max_ref, 1e-12)
print(f"Max abs diff: {max_diff:.6f}")
print(f"Max relative diff: {rel_diff:.6e}")

# print(y_3d)
# print(y_2d)
