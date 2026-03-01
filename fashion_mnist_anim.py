"""PaCMAP-MLX Fashion-MNIST 70K animation. 450 iters, 120fps, real timestamps."""
import os, time
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import fetch_openml

FPS = 120
SEED = 42

print("Loading Fashion-MNIST 70K...")
fm = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X = fm.data.astype(np.float32) / 255.0
y = fm.target.astype(np.int32)
n = X.shape[0]

np.random.seed(SEED)
viz_idx = np.random.choice(n, 15000, replace=False)
y_viz = y[viz_idx]
colors = plt.cm.tab10(y_viz / 9.0)

# Monkey-patch PaCMAP to capture per-iteration snapshots
from pacmap_mlx.pacmap import PaCMAP, _brute_knn, _sample_neighbors, _sample_MN_pairs, _sample_FP_pairs
import math

snaps = []
snap_times = []

print("Running PaCMAP-MLX...")
t_global = time.time()

p = PaCMAP(n_components=2, n_neighbors=10, random_state=SEED, verbose=True)

# Manually run the pipeline to inject snapshot capture
X_np = X.copy().astype(np.float32)
rng = np.random.default_rng(SEED)

X_mx = p._preprocess(X_np)
mx.eval(X_mx)
X_proc_np = np.array(X_mx)

n_neighbors = min(10, n - 1)
n_MN = int(0.5 * n_neighbors)
n_FP = int(2.0 * n_neighbors)

knn_indices, knn_distances = _brute_knn(X_mx, n_neighbors + 50)
knn_indices_np = np.array(knn_indices, dtype=np.int32)
knn_distances_np = np.array(knn_distances, dtype=np.float32)

pair_neighbors = _sample_neighbors(knn_distances_np, knn_indices_np, n_neighbors)
pair_MN = _sample_MN_pairs(X_mx, n_MN, rng)
pair_FP = _sample_FP_pairs(n, pair_neighbors, n_neighbors, n_FP, rng)

# PCA init
Y = mx.array(X_proc_np[:, :2]) * 0.01
mx.eval(Y)

# Capture init
snaps.append(np.array(Y)[viz_idx])
snap_times.append(time.time() - t_global)

# Run optimization with per-iteration snapshots
src_nn = mx.array(pair_neighbors[:, 0].copy())
dst_nn = mx.array(pair_neighbors[:, 1].copy())
src_mn = mx.array(pair_MN[:, 0].copy())
dst_mn = mx.array(pair_MN[:, 1].copy())
src_fp = mx.array(pair_FP[:, 0].copy())
dst_fp = mx.array(pair_FP[:, 1].copy())
mx.eval(src_nn, dst_nn, src_mn, dst_mn, src_fp, dst_fp)

m = mx.zeros_like(Y)
v = mx.zeros_like(Y)
beta1, beta2, lr = 0.9, 0.999, 1.0
num_iters = (100, 100, 250)
num_iters_total = sum(num_iters)
phase1, phase2, _ = num_iters
w_MN_init = 1000.0

for itr in range(num_iters_total):
    if itr < phase1:
        w_MN = (1.0 - itr / phase1) * w_MN_init + (itr / phase1) * 3.0
        w_nb, w_fp = 2.0, 1.0
    elif itr < phase1 + phase2:
        w_MN, w_nb, w_fp = 3.0, 3.0, 1.0
    else:
        w_MN, w_nb, w_fp = 0.0, 1.0, 1.0
    
    grad = mx.zeros_like(Y)
    
    diff = Y[src_nn] - Y[dst_nn]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_nb * 20.0 / ((10.0 + d) * (10.0 + d))) * diff
    grad = grad.at[src_nn].add(g)
    grad = grad.at[dst_nn].add(-g)
    
    if w_MN > 0:
        diff = Y[src_mn] - Y[dst_mn]
        d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
        g = (w_MN * 20000.0 / ((10000.0 + d) * (10000.0 + d))) * diff
        grad = grad.at[src_mn].add(g)
        grad = grad.at[dst_mn].add(-g)
    
    diff = Y[src_fp] - Y[dst_fp]
    d = mx.sum(diff * diff, axis=1, keepdims=True) + 1.0
    g = (w_fp * 2.0 / ((1.0 + d) * (1.0 + d))) * diff
    grad = grad.at[src_fp].add(-g)
    grad = grad.at[dst_fp].add(g)
    
    lr_t = lr * math.sqrt(1.0 - beta2 ** (itr + 1)) / (1.0 - beta1 ** (itr + 1))
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * (grad * grad)
    Y = Y - lr_t * m / (mx.sqrt(v) + 1e-7)
    
    mx.eval(Y, m, v)
    snaps.append(np.array(Y)[viz_idx])
    snap_times.append(time.time() - t_global)

t_total = time.time() - t_global
print(f"Done: {t_total:.2f}s, {len(snaps)} snapshots")

# Build animation
n_snap = len(snaps)

def get_square_lims(emb, margin=0.1):
    cx = (emb[:, 0].min() + emb[:, 0].max()) / 2
    cy = (emb[:, 1].min() + emb[:, 1].max()) / 2
    span = max(emb[:, 0].max() - emb[:, 0].min(), emb[:, 1].max() - emb[:, 1].min())
    hs = span / 2 * (1 + margin)
    return (cx - hs, cx + hs), (cy - hs, cy + hs)

xlim, ylim = get_square_lims(snaps[-1])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect('equal')
ax.axis('off')

scatter = ax.scatter([], [], s=1.5, alpha=0.6)
title = ax.set_title('', color='white', fontsize=14, pad=10, fontfamily='monospace')

init_f = 60     # 0.5s hold on init
hold_f = 240    # 2s hold on final
total_f = init_f + n_snap + hold_f

print(f"Rendering {total_f} frames at {FPS}fps...")

def update(frame):
    if frame < init_f:
        idx = 0
        t = snap_times[0]
        label = f'pacmap-mlx  Fashion-MNIST  70,000 x 784  init  t={t:.2f}s'
    elif frame < init_f + n_snap:
        idx = frame - init_f
        t = snap_times[idx]
        label = f'pacmap-mlx  Fashion-MNIST  70,000 x 784  iter {idx}/{num_iters_total}  t={t:.2f}s'
    else:
        idx = n_snap - 1
        label = f'pacmap-mlx  Fashion-MNIST  70,000 x 784  done in {t_total:.1f}s'

    scatter.set_offsets(snaps[idx])
    scatter.set_color(colors)
    title.set_text(label)
    return scatter, title

anim = animation.FuncAnimation(fig, update, frames=total_f, blit=True, interval=1000 // FPS)
outpath = '/Users/hanxiao/.openclaw/workspace/pacmap-mlx/animation.mp4'
anim.save(outpath, writer=animation.FFMpegWriter(fps=FPS, bitrate=8000,
          extra_args=['-pix_fmt', 'yuv420p']))
plt.close()
print(f"Saved {outpath} ({os.path.getsize(outpath) / 1024 / 1024:.1f} MB)")
