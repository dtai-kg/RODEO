import os
import sys
import csv
import pprint
import numpy as np
import matplotlib.pyplot as plt

from graph.build_pg_sotab_schema_org import table_gt_info

from utils.file_registry import get_path

def collect_p_subj(dataset_path, topic2S):

    node_train_gt_path = get_path(dataset_path, 'CTA_SCH_TRAIN_GT_CSV', 'sotab')
    node_val_gt_path = get_path(dataset_path, 'CTA_SCH_VAL_GT_CSV', 'sotab')
    edge_train_gt_path = get_path(dataset_path, 'CPA_SCH_TRAIN_GT_CSV', 'sotab')
    edge_val_gt_path = get_path(dataset_path, 'CPA_SCH_VAL_GT_CSV', 'sotab')

    file_paths = [node_train_gt_path, node_val_gt_path, edge_train_gt_path, edge_val_gt_path]

    result = {}
    for file_path in file_paths:
        if 'CTA' in file_path:
            continue

        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:

                if row[0] == 'table_name':
                    continue

                topic = row[0].split('_')[0]
                subj = topic2S[topic]['subj']

                if subj not in result.keys():
                    result[subj] = dict()

                property = row[3]
                if property not in result[subj]:
                    result[subj][property] = 0

                result[subj][property] += 1

    # Flatten the structure to directly use result[property]
    flattened_result = {}
    for subj, properties in result.items():
        for property, value in properties.items():
            if property not in flattened_result:
                flattened_result[property] = value
            else:
                # Handle conflicts if the same property exists across different subjects
                flattened_result[property] += value  # Example: Summing values

    result_percentage = {}
    for subj, properties in result.items():
        result_percentage[subj] = {}
        for property, value in properties.items():
            if property in flattened_result and flattened_result[property] > 0:
                result_percentage[subj][property] = (value / flattened_result[property])*100
            else:
                result_percentage[subj][property] = 0.0  # Avoid division by zero

    # pprint.pprint (result_percentage)
    return result_percentage

def plot_constraint_heatmap(SPO, info):

    filtered_SPO = [(s, p, o) for (s, p, o) in SPO if not p.startswith("??")]

    # Rebuild the types and properties based on filtered SPO
    types = sorted(set(s for (s, _, _) in filtered_SPO))
    properties = sorted(set(p for (_, p, _) in filtered_SPO))

    # Update label mappings based on filtered data
    type_to_idx = {t: i for i, t in enumerate(types)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    property_to_idx = {p: i for i, p in enumerate(properties)}
    idx_to_property = {i: p for p, i in property_to_idx.items()}

    # Initialize the new matrix with dimensions (types x properties)
    matrix_size_x = len(properties)  # Number of properties
    matrix_size_y = len(types)  # Number of types
    property_type_matrix = np.zeros((matrix_size_y, matrix_size_x), dtype=int)

    # Populate the matrix with co-occurrence counts for type -> property (source)
    # for (s, p, _) in filtered_SPO:
    #     type_id = type_to_idx[s]
    #     property_id = property_to_idx[p]
    #     property_type_matrix[type_id, property_id] += 1

    # Fill in the matrix with percentages
    for subj, properties in info.items():
        type_id = type_to_idx[subj]  # Map subj to type_id
        for property, percentage in properties.items():
            property_id = property_to_idx[property]  # Map property to property_id
            property_type_matrix[type_id, property_id] = percentage

    # Adjust figure size for larger matrices
    fig_width = min(20, max(10, matrix_size_x / 2))  # Dynamic width
    fig_height = matrix_size_y / 2  # Ensure full y-axis height

    # Plot the adjusted matrix
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.matshow(property_type_matrix, cmap='Blues', vmin=0)

    # Adjust the colorbar to align with the data matrix and set appropriate length
    cbar = fig.colorbar(cax, ax=ax, pad=0.01, shrink=0.32, aspect= 10)  # Reduced pad for closer alignment
    cbar.ax.tick_params(labelsize=8)

    # Set axis labels with ticks based on filtered results
    y_ticks = np.arange(matrix_size_y)
    ax.set_yticks(y_ticks)  # Ensure all y-ticks are set
    ax.set_yticklabels([idx_to_type[i][:-5] for i in y_ticks], fontsize=9)

    for i, label in enumerate([idx_to_property[i] for i in range(matrix_size_x)]):
        ax.text(i, -1.05, label, ha='left', va='bottom', fontweight='normal', fontsize=6.6, rotation=45, rotation_mode='anchor')
    ax.set_xticks([])  # Remove x-ticks

    ax.set_xlabel('Semantic Properties', fontsize=18)
    ax.set_ylabel('Semantic Types', fontsize=18)

    # Save the plot as an image
    plt.savefig('constraint_heatmap.png', dpi=300)
    plt.show()


def plot_constraint_heatmap_hl(SPO, info, highlight_type=None, highlight_property=None, highlight_column_property=None):
    # Filter SPO based on "/name" in the subject and "??" in properties
    filtered_SPO = [(s, p, o) for (s, p, o) in SPO if not p.startswith("??")]

    # Rebuild the types and properties based on filtered SPO
    types = sorted(set(s for (s, _, _) in filtered_SPO))
    properties = sorted(set(p for (_, p, _) in filtered_SPO))

    # Update label mappings based on filtered data
    type_to_idx = {t: i for i, t in enumerate(types)}
    # print (type_to_idx)
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    property_to_idx = {p: i for i, p in enumerate(properties)}
    idx_to_property = {i: p for p, i in property_to_idx.items()}

    # Initialize the new matrix with dimensions (types x properties)
    matrix_size_x = len(properties)  # Number of properties
    matrix_size_y = len(types)  # Number of types
    property_type_matrix = np.zeros((matrix_size_y, matrix_size_x), dtype=int)

    # Fill in the matrix with percentages
    for subj, properties_dict in info.items():
        type_id = type_to_idx[subj]  # Map subj to type_id
        for property, percentage in properties_dict.items():
            property_id = property_to_idx[property]  # Map property to property_id
            property_type_matrix[type_id, property_id] = percentage

    # Adjust figure size for larger matrices
    fig_width = min(20, max(10, matrix_size_x / 2))
    fig_height = matrix_size_y / 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.matshow(property_type_matrix, cmap='Blues', vmin=0)

    # --- NEW: highlight whole column (drawn first so cell outline sits on top) ---
    if highlight_column_property and (highlight_column_property in property_to_idx):
        col_idx = property_to_idx[highlight_column_property]
        # Rectangle that spans the whole column height
        ax.add_patch(plt.Rectangle(
            (col_idx - 0.5, -0.5),     # (x, y) bottom-left in data coords
            1,                         # width = 1 column
            matrix_size_y,             # height = number of rows
            fill=False,
            edgecolor='red',
            lw=3
        ))

    # Highlight crossed block if both highlight_type and highlight_property are provided
    if highlight_type and highlight_property:
        if highlight_type in type_to_idx and highlight_property in property_to_idx:
            ty_idx = type_to_idx[highlight_type]
            pr_idx = property_to_idx[highlight_property]
            # Add a red rectangle around the selected cell
            ax.add_patch(plt.Rectangle((pr_idx - 0.5, ty_idx - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))
            # Optional: override cell color for visibility
            property_type_matrix[ty_idx, pr_idx] = property_type_matrix[ty_idx, pr_idx]  # can increase value if needed

    # Colorbar
    cbar = fig.colorbar(cax, ax=ax, pad=0.01, shrink=0.32, aspect=10)
    cbar.ax.tick_params(labelsize=8)

    # Set axis labels with ticks based on filtered results
    y_ticks = np.arange(matrix_size_y)
    y_labels = [idx_to_type[i][:-5] for i in y_ticks]
    ax.set_yticks(y_ticks)  # Ensure all y-ticks are set
    ax.set_yticklabels(y_labels, fontsize=9)

    # Highlight y-axis label if needed
    for i, label in enumerate(y_labels):
        if highlight_type and types[i] == highlight_type:
            ax.text(-0.8, i, label, va='center', ha='right', fontsize=8.5, color='red',
                    fontweight='bold')  # highlighted label
        else:
            ax.text(-0.8, i, label, va='center', ha='right', fontsize=8.5)

    # Set x-axis labels manually
    for i, label in enumerate([idx_to_property[i] for i in range(matrix_size_x)]):
        if highlight_property and idx_to_property[i] == highlight_property:
            ax.text(i, -1.05, label, ha='left', va='bottom', fontweight='bold', fontsize=6.6, rotation=45,
                    rotation_mode='anchor', color='red')  # highlighted label
        elif highlight_column_property and idx_to_property[i] == highlight_column_property:
            ax.text(i, -1.05, label, ha='left', va='bottom', fontweight='bold', fontsize=6.6, rotation=45,
                    rotation_mode='anchor', color='red')  # highlighted label
        else:
            ax.text(i, -1.05, label, ha='left', va='bottom', fontweight='normal', fontsize=6.6, rotation=45,
                    rotation_mode='anchor')

    ax.set_xticks([])  # Remove x-ticks
    ax.set_xlabel('Semantic Properties (Relations)', fontsize=18)
    ax.set_yticks([])
    ax.set_ylabel('Semantic Types', fontsize=18, labelpad=70)

    plt.savefig('constraint_heatmap_hl2.png', dpi=300)
    plt.show()


def main():

    dataset_path = "/apollo/users/dya/dataset/semtab"
    SPO, topic2S = table_gt_info(dataset_path)
    info = collect_p_subj(dataset_path, topic2S)

    # plot_constraint_heatmap(SPO, info)
    plot_constraint_heatmap_hl(SPO, info, 'Restaurant/name', 'servesCuisine', 'description')

if __name__ == "__main__":
    main()
