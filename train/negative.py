import torch

def select_negative(training_mode, batch_idx, distances, p_distances_expanded, gnn_embeddings):

    if training_mode == '1_1':
        condition = batch_idx % 2 != 0
    elif training_mode == '1_2':
        condition = batch_idx % 3 == 0
    elif training_mode == '2_1':
        condition = batch_idx % 3 != 0
    elif training_mode == '0':
        condition = False
    elif training_mode == '1':
        condition = True
    else:
        raise ("Could not find the training mode for negative selection")

    if condition:
        # hard
        remove_positive = distances != p_distances_expanded
        valid_distances = torch.where(remove_positive, distances, torch.max(distances))
        min_distances, min_indices = torch.min(valid_distances, dim=-1)
        negative = gnn_embeddings[min_indices]
    else:
        # semi-hard
        eps = 1e-04
        comparison_semi_hard = p_distances_expanded + eps < distances
        valid_distances = torch.where(comparison_semi_hard, distances, torch.max(distances))
        min_distances, min_indices = torch.min(valid_distances, dim=-1)
        negative = gnn_embeddings[min_indices]

    return negative

def select_negative_multilabel(training_mode, batch_idx, anchor, positive, gnn_embeddings, positives_in_batch, dist_func):

    if training_mode == '1_1':
        condition = batch_idx % 2 != 0
    elif training_mode == '1_2':
        condition = batch_idx % 3 == 0
    elif training_mode == '2_1':
        condition = batch_idx % 3 != 0
    elif training_mode == '0':
        condition = False
    elif training_mode == '1':
        condition = True
    else:
        raise ("Could not find the training mode for negative selection")

    if condition:
        # hard
        distances = torch.square(torch.cdist(anchor.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))
        mask = torch.ones_like(distances, dtype=torch.bool)
        for i, pos_indices in enumerate(positives_in_batch):
            mask[i, pos_indices] = False  # Set the positive positions for each anchor to False
        valid_distances = torch.where(mask, distances, torch.max(distances))
        min_distances, min_indices = torch.min(valid_distances, dim=-1)
        negative = gnn_embeddings[min_indices]
    else:
        # semi-hard
        distances = torch.square(torch.cdist(anchor.unsqueeze(1), gnn_embeddings.unsqueeze(0), p=2).squeeze(1))
        p_distances = torch.square(dist_func(anchor, positive))
        p_distances_expanded = p_distances.view(-1, 1).expand(distances.shape[0], distances.shape[1])
        # Create a mask to exclude all positive distances
        mask = torch.ones_like(distances, dtype=torch.bool)  # Start with all distances as valid (True)
        for i, pos_indices in enumerate(positives_in_batch):
            mask[i, pos_indices] = False  # Set the positive positions for each anchor to False
        # Apply semi-hard negative condition: distances greater than positive distances plus epsilon
        eps = 1e-04
        comparison_semi_hard = p_distances_expanded + eps < distances
        # Combine mask and semi-hard condition to get valid distances
        combined_mask = mask & comparison_semi_hard
        valid_distances = torch.where(combined_mask, distances, torch.max(distances))
        min_distances, min_indices = torch.min(valid_distances, dim=-1)
        negative = gnn_embeddings[min_indices]

    return negative, distances
