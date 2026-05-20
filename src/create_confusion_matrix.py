from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix_from_csv(
    csv_path,
    output_path,
    title="Confusion Matrix",
    normalize=False,
    figsize=(12, 10),
    dpi=300,
    cmap="Blues",
):
    """
    Load a confusion matrix from a CSV and save it as an image.

    Expected CSV format:
        - First row contains column labels
        - First column contains row labels
        - Remaining values are numeric confusion matrix entries

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file.
    output_path : str or Path
        Path to save the output image (e.g. .png).
    title : str
        Plot title.
    normalize : bool
        If True, normalise each row to percentages.
    figsize : tuple
        Figure size passed to matplotlib.
    dpi : int
        Output resolution.
    cmap : str
        Matplotlib colormap.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    # Read CSV with first column as index
    df = pd.read_csv(csv_path, index_col=0)

    # Force numeric just in case
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        raise ValueError("CSV contains non-numeric values in the matrix body.")

    # Convert class labels to ALL CAPS
    labels_x = [str(label).upper() for label in df.columns]
    labels_y = [str(label).upper() for label in df.index]

    cm = df.to_numpy()

    # Optional row normalisation
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm / row_sums
        value_fmt = ".2f"
    else:
        cm_plot = cm
        value_fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels_x)),
        yticks=np.arange(len(labels_y)),
        xticklabels=labels_x,
        yticklabels=labels_y,
        xlabel="PREDICTED LABEL",
        ylabel="TRUE LABEL",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    threshold = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            if normalize:
                text_value = format(cm_plot[i, j], value_fmt)
            else:
                text_value = f"{int(cm[i, j])}"

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix image to: {output_path}")


if __name__ == "__main__":
    # csv_path = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\out\mfd_stream_train\pulse_tail_pad\mfd_stream_resnet18_dual_pulseonly_windowstateaux_20260327_020333\cm_best.csv"
    csv_path = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\out\mfd_stream_train\jan_fused_buffer_power_phase_gap_noise_images_20260415_235941\cm_best.csv"
    
    # output_path = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\out\mfd_stream_train\jan_fused\jan_fused_buffer_power_phase_20260404_133336\cm_best.png"
    output_path = csv_path[:-3] + "png"

    plot_confusion_matrix_from_csv(
        csv_path=csv_path,
        output_path=output_path,
        title="Confusion Matrix",
        normalize=False,   # set True if you want row-normalised values
        figsize=(12, 10),
        dpi=300,
        cmap="Blues",
    )