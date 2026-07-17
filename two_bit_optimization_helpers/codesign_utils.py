from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import math

def plot_event(X, timeslices_range, index):
    if index < 0 or index >= X.shape[0]:
        print("Index out of range.")
        return
    
    timeframe_num = X[index].shape[2]
    fig, axes = plt.subplots(1, timeframe_num, figsize=(timeframe_num*4, 16))

    vmin = int(np.floor(X[index].min()))
    vmax = int(np.ceil(X[index].max()))
    bounds = np.arange(vmin, vmax + 2)
    n_colors = len(bounds) - 1

    colors_list = []
    for val in bounds[:-1]:
        if val == 0:
            colors_list.append((1, 1, 1))
        else:
            intensity = (val - 0) / (vmax - 0) if vmax > 0 else 1
            colors_list.append((1, 1 - intensity, 1 - intensity))

    cmap = ListedColormap(colors_list)
    norm = BoundaryNorm(bounds, n_colors)

    imgs = []
    for i_slice in range(timeframe_num):
        img = axes[i_slice].imshow(
            X[index, :, :, i_slice],
            interpolation='nearest',
            origin='lower',
            cmap=cmap,
            norm=norm
        )
        imgs.append(img)
        axes[i_slice].set_title(f'Event {index} – Timeframe {timeslices_range[i_slice]}')
        axes[i_slice].set_xticks(range(0, X.shape[2], max(1, X.shape[2] // 5)))
        axes[i_slice].set_yticks(range(0, X.shape[1], max(1, X.shape[1] // 5)))

    cbar = fig.colorbar(
        imgs[0], ax=axes.ravel().tolist(),
        orientation='vertical', fraction=0.025, pad=0.08, aspect=20,
        boundaries=bounds, ticks=bounds
    )
    cbar.set_label("Value")

    plt.show()

def animate_event(X, timeslices_range, index):
    if index < 0 or index >= X.shape[0]:
        print("Index out of range.")
        return
    
    timeframe_num = X[index].shape[2]
    fig, ax = plt.subplots(figsize=(6, 6))

    vmin = int(np.floor(X[index].min()))
    vmax = int(np.ceil(X[index].max()))
    bounds = np.arange(vmin, vmax + 2)
    n_colors = len(bounds) - 1

    colors_list = []
    for val in bounds[:-1]:
        if val == 0:
            colors_list.append((1, 1, 1))
        else:
            intensity = (val - 0) / (vmax - 0) if vmax > 0 else 1
            colors_list.append((1, 1 - intensity, 1 - intensity))

    cmap = ListedColormap(colors_list)
    norm = BoundaryNorm(bounds, n_colors)

    img = ax.imshow(
        X[index, :, :, 0],
        interpolation='nearest',
        origin='lower',
        cmap=cmap,
        norm=norm
    )

    ax.set_title(f'Event {index} – Timeframe {timeslices_range[0]}')
    ax.set_xticks(range(0, X.shape[2], max(1, X.shape[2] // 5)))
    ax.set_yticks(range(0, X.shape[1], max(1, X.shape[1] // 5)))

    cbar = fig.colorbar(
        img, ax=ax, orientation='vertical',
        fraction=0.025, pad=0.02, aspect=40,
        boundaries=bounds, ticks=bounds
    )
    cbar.set_label('Value')

    def update(frame):
        img.set_data(X[index, :, :, frame])
        ax.set_title(f'Event {index} – Timeframe {timeslices_range[frame]}')
        return img,

    ani = FuncAnimation(fig, update, frames=timeframe_num, blit=False, interval=500)
    plt.close(fig)
    return ani

def plot_layer_comparisons(trace_hls4ml, trace_qkeras, save_path="comparison_qkeras_hls4ml.jpeg"):
    layers = [
        layer for layer in trace_hls4ml.keys()
        if "_alpha" not in layer and layer != "q_separable_conv2d_depthwise"
    ]
    
    num_layers = len(layers)
    num_columns = 2
    num_rows = math.ceil(num_layers / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        klayer = layer.replace("_pointwise", "").replace("_linear", "")

        min_x = min(np.amin(trace_hls4ml[layer]), np.amin(trace_qkeras[klayer]))
        max_x = max(np.amax(trace_hls4ml[layer]), np.amax(trace_qkeras[klayer]))

        x = trace_hls4ml[layer].flatten()
        y = trace_qkeras[klayer].flatten()

        golden_min_x = np.amin(trace_qkeras[klayer])
        golden_max_x = np.amax(trace_qkeras[klayer])
        range_size = abs(golden_min_x) + abs(golden_max_x)

        integer_bits = math.ceil(math.log2(range_size)) if range_size > 0 else 0

        nonzero = trace_qkeras[klayer][trace_qkeras[klayer] != 0]
        min_abs_value = np.min(np.abs(nonzero)) if len(nonzero) > 0 else 0
        decimal_bits = abs(math.floor(math.log2(min_abs_value))) if min_abs_value > 0 else 0

        mse_value = np.mean((x - y) ** 2)

        ax = axes[i]
        ax.plot([min_x, max_x], [min_x, max_x], color="gray", linestyle="--")
        ax.scatter(x, y, s=0.2, color="red")

        stats = (
            f"Range: [{golden_min_x:.2f}, {golden_max_x:.2f}]\n"
            f"Range Size: {range_size:.2f}\n"
            f"Bits (I): {integer_bits}\n"
            f"Bits (D): {decimal_bits}\n"
            f"MSE: {mse_value:.2e}"
        )

        ax.text(
            0.05, 0.95, stats,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )

        ax.set_xlabel(f"hls4ml {layer}")
        ax.set_ylabel(f"QKeras {klayer}")
        ax.set_title(f"Comparison QKeras vs. hls4ml ({klayer})")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()