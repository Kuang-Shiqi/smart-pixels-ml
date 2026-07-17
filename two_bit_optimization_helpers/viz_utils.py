import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def viz_history(history, title=None, start_epoch=1, prefix=""):

    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])


    if len(history.epoch) < start_epoch:
        print(f"WARNING: not enough epochs: {len(history.epoch)}")
        return

    start_idx   = start_epoch - 1
    training_validation_loss_png = f"{prefix}training_validation_loss_e{start_epoch}.png"

    # Best val_loss within the plotted range
    local_rel_idx = int(np.nanargmin(val_loss[start_idx:]))
    best_epoch    = start_epoch + local_rel_idx
    best_val      = float(val_loss[start_idx + local_rel_idx])

    epochs = np.arange(start_epoch, len(loss) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, loss[start_idx:], label='Training Loss')
    ax.plot(epochs, val_loss[start_idx:], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Vertical marker
    ax.axvline(best_epoch, color='red', linestyle='--', linewidth=1.5)
    ax.scatter([best_epoch], [best_val], s=30, color='red', zorder=3)

    # Red “legend/box”
    txt = f'best val_loss = {best_val:.6g}\nepoch = {best_epoch}'
    at = AnchoredText(txt, loc='upper right', prop=dict(color='red'), frameon=True, borderpad=0.5)
    at.patch.set_edgecolor('red')
    at.patch.set_alpha(0.85)
    ax.add_artist(at)
    if title:
        plt.title(title)

    fig.tight_layout()
    fig.savefig(training_validation_loss_png, bbox_inches='tight', pad_inches=0.5)
    plt.show()

    print(f"Save history as: {training_validation_loss_png}")
