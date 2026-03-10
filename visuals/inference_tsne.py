import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
from matplotlib.colors import ListedColormap
import seaborn as sns
from utils.data_utils import save, load

# Generate a diverse and unique color palette without sequential similarity
# Use hue shifts to ensure visually distinct colors
def generate_unique_colors(num_colors):
    color_palette = [
        # "#FF5733",  # Bright Red-Orange
        "#99a832", # 111    Book
        "#FFBD33",  # Bright Orange
        # "#FFD700",  # Bright Yellow
        "#ADFF2F",  # Bright Lime Green
        "#32CD32",  # Bright Green
        "#20B2AA",  # Bright Aqua
        "#00CED1",  # Bright Turquoise
        "#1E90FF",  # Bright Sky Blue
        "#4169E1",  # Bright Royal Blue
        "#6A5ACD",  # Bright Indigo 111 Movie
        "#9932CC",  # Bright Purple Person
        "#C71585",  # Bright Deep Pink
        "#FF69B4",  # Bright Pink Music recoding
        "#FF1493",  # Bright Hot Pink
        "#FFA07A",  # Bright Salmon
        "#FF4500",  # Bright Red
        "#FFDAB9"  # Bright Peach
    ]
    random.shuffle(color_palette)
    colors = [color_palette[i] for i in range(num_colors)]

    # colors = ["#9932CC", "#C71585", "#1E90FF", "#FF69B4", "#FF4500", "#99a832", "#ADFF2F", "#6A5ACD", "#FFDAB9", "#FFBD33", "#FF1493", "#4169E1", "#20B2AA", "#00CED1", "#32CD32", "#FFA07A"]

    return colors

def plot_opentsne_2d(gnn_embeddings, cta_ts_pred_col, cta_ts_true_col, subj_class_idxs, subj_class, save_path="opentsne_plot.png"):

    # Validate and convert inputs to numpy arrays
    try:
        gnn_embeddings = np.array(gnn_embeddings, dtype=np.float32)
        cta_ts_pred_col = np.array(cta_ts_pred_col, dtype=np.float32)
        cta_ts_true_col = np.array(cta_ts_true_col, dtype=np.int32)
    except ValueError as e:
        raise ValueError(f"Error converting inputs to numpy arrays: {e}")

    if gnn_embeddings.shape[1] != cta_ts_pred_col.shape[1]:
        raise ValueError(f"Dimension mismatch: gnn_embeddings have {gnn_embeddings.shape[1]} features, "
                         f"but cta_ts_pred_col have {cta_ts_pred_col.shape[1]} features")

    if len(cta_ts_true_col) != len(cta_ts_pred_col):
        raise ValueError("cta_ts_true_col must match the number of rows in cta_ts_pred_col.")

    if gnn_embeddings.ndim != 2 or cta_ts_pred_col.ndim != 2:
        raise ValueError(
            "Both gnn_embeddings and cta_ts_pred_col must be 2D arrays or lists of shape (n_samples, n_features)")

    # Filter GNN embeddings based on indices in subj_class_idxs
    filtered_gnn_embeddings = gnn_embeddings[subj_class_idxs]

    # Verify compatibility of shapes for stacking
    if filtered_gnn_embeddings.shape[1] != cta_ts_pred_col.shape[1]:
        raise ValueError(f"Dimension mismatch after filtering: filtered_gnn_embeddings have {filtered_gnn_embeddings.shape[1]} features, "
                         f"but cta_ts_pred_col have {cta_ts_pred_col.shape[1]} features")

    # Combine data for TSNE transformation
    data_combined = np.vstack((filtered_gnn_embeddings, cta_ts_pred_col))

    # Apply openTSNE
    tsne = TSNE(n_components=2, perplexity=100, n_iter=300, initialization='pca', random_state=42)

    data_2d = tsne.fit(data_combined)

    # Split transformed data back to gnn and true_col
    gnn_2d = data_2d[:len(filtered_gnn_embeddings)]
    true_col_2d = data_2d[len(filtered_gnn_embeddings):]

    # Assign colors based on ele_idx
    unique_indices = np.unique(cta_ts_true_col)
    num_colors = len(unique_indices)
    color_palette = generate_unique_colors(num_colors)

    # Create a color mapping for the indices
    idx_to_color = {idx: color_palette[i] for i, idx in enumerate(unique_indices)}

    # Prepare GNN embedding colors (default gray)
    gnn_colors = ['gray'] * len(filtered_gnn_embeddings)

    # Assign colors for GNN embeddings referred by cta_ts_true_col
    for i, idx in enumerate(subj_class_idxs):
        if idx < len(filtered_gnn_embeddings):  # Ensure idx is in range
            gnn_colors[i] = idx_to_color.get(idx, 'gray')

    # Plot True Embeddings (Close to Anchors) with circle marker first
    plt.figure(figsize=(17, 8))
    for i, idx in enumerate(cta_ts_true_col):
        plt.scatter(true_col_2d[i, 0], true_col_2d[i, 1], color=idx_to_color[idx], s=40, marker='o', alpha=0.4)

    # Plot GNN Embeddings (Anchors) with star marker on top
    plt.scatter(gnn_2d[:, 0], gnn_2d[:, 1], c='#FF2E2E', alpha=1.0, s=90, marker='d', edgecolor='#333333')

    # Attach labels for GNN embeddings
    for i, (x, y) in enumerate(gnn_2d):
        plt.text(x + 1, y + 0.5, subj_class[i], fontsize=14, ha='left', color='black')

    # Add legend based on subj_class
    legend_handles = []
    legend_handles.append(plt.Line2D([0], [0], marker='d', color='#FF2E2E', label='node embedding', markersize=12, linestyle='None'))
    legend_handles.append(plt.Line2D([0], [0], label='(semantic type)', linestyle='None'))
    for i, idx in enumerate(subj_class_idxs):
        if idx in idx_to_color:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color=idx_to_color[idx], label='column - ' + subj_class[i], markersize=12, linestyle='None'))
    # plt.legend(handles=legend_handles, title="column embeddings", loc="best", fontsize=10)
    plt.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.5, 1.0), fontsize=18)

    # Add titles and grid
    plt.xlabel('t-SNE Component 1', fontsize=18)
    plt.ylabel('t-SNE Component 2', fontsize=18)

    # Set background color
    # plt.gca().set_facecolor('#f9f9f9')

    # Dynamically adjust axis limits for better visualization
    x_min, x_max = data_2d[:, 0].min() - 2, data_2d[:, 0].max() + 2
    y_min, y_max = data_2d[:, 1].min() - 2, data_2d[:, 1].max() + 2
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.subplots_adjust(right=4)
    plt.tight_layout()

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def main():

    gnn_embeddings, cta_ts_pred_col, cta_ts_true_col, subj_class_idxs, subj_class = load('cta_embeddings.pkl')
    subj_class = [subj.replace('/name', '') for subj in subj_class]
    plot_opentsne_2d(gnn_embeddings, cta_ts_pred_col, cta_ts_true_col, subj_class_idxs, subj_class, save_path="opentsne_plot_cta.png")

if __name__ == '__main__':
    main()