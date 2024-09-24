import numpy as np
import torch


def label_flipping_untargeted(y, flip_ratio=0.5):
    """Apply label flipping attack to randomly change labels of traffic samples."""
    print('y type:', type(y))
    num_samples = len(y)
    num_flips = int(num_samples * flip_ratio)
    
    # Generate random indices based on the length of the series
    flip_indices = np.random.choice(num_samples, num_flips, replace=False)
    print("Generated flip_indices:", flip_indices)
    
    y_flipped = y.copy()  # Ensure y_flipped is a Series, not an ndarray
    
    for idx in flip_indices:
        # Use iloc to access by position instead of index
        new_label = np.random.choice(np.setdiff1d(np.unique(y), y.iloc[idx]))
        y_flipped.iloc[idx] = new_label
    
    return y_flipped


import torch
import numpy as np
import pandas as pd

def flip_labels(args, y, flip_fraction=0.1):
    num_classes = args.num_classes
    num_samples = len(y)
    num_flips = int(flip_fraction * num_samples)
    
    # Select random indices to flip using PyTorch
    indices_to_flip = torch.randperm(num_samples)[:num_flips]
    
    # Convert the PyTorch tensor to a NumPy array or list for Pandas indexing
    indices_to_flip = indices_to_flip.numpy()
    
    # Get the current labels for these indices using .iloc for positional indexing
    current_labels = y.iloc[indices_to_flip]
    
    # Ensure current labels are numeric
    if current_labels.dtype == object:  # This checks if the data type is 'object', which usually indicates mixed types
        # Try converting to numeric, coerce errors to NaN, which we can handle separately if needed
        current_labels = pd.to_numeric(current_labels, errors='coerce')
    
    # Create a mask for choosing a new label
    new_labels = torch.randint(0, num_classes, (num_flips,))
    
    # Ensure new labels are different from the current labels, handle NaNs if any
    if current_labels.isna().any():
        print("Warning: Some labels could not be converted to numeric and will not be flipped.")
        new_labels = np.where(current_labels.notna(), (current_labels + 1 + new_labels.numpy()) % num_classes, current_labels)
    else:
        new_labels = (current_labels + 1 + new_labels.numpy()) % num_classes
    
    # Assign the new labels back to the original Series
    y.iloc[indices_to_flip] = new_labels

    return y

def label_flipping_(dataset, idxs, ratio=0.1):
    """Apply label flipping attack to randomly change labels of traffic samples."""
    print("inside label_flipping")
    print("num_samples:", len(idxs))
    print("ratio:", ratio)
    num_samples = len(idxs)
    num_flips = int(num_samples * ratio)
    print("num_flips:", num_flips)
    
    # Generate random indices based on the length of the series
    flip_indices = np.random.choice(idxs, num_flips, replace=False)
    #print("Generated flip_indices:", flip_indices)
    
    y_flipped = dataset.targets.copy()  # Ensure y_flipped is a Series, not an ndarray
    
    for idx in flip_indices:    
        # Use iloc to access by position instead of index
        #print("dataset.labels[idx]:", dataset.labels[idx])
        new_label = np.random.choice(np.setdiff1d(np.unique(dataset.targets), dataset.targets[idx]))
        #print("new_label:", new_label)
        y_flipped[idx] = new_label
    dataset.targets = y_flipped
    return dataset

import numpy as np

def label_flipping(dataset, idxs, ratio=0.1):
    """Apply label flipping attack to randomly change labels of traffic samples."""
    
    num_flips = int(len(idxs) * ratio)
    np.random.seed(1234)

    # Generate random indices to flip
    flip_indices = np.random.choice(idxs, num_flips, replace=False)
    
    # Convert targets to a NumPy array for efficient processing
    targets = np.array(dataset.targets)
    unique_labels = np.unique(targets)

    # Get the current labels of the selected indices
    current_labels = targets[flip_indices]
    
    # Generate new labels for the selected indices
    new_labels = np.array([
        np.random.choice(unique_labels[unique_labels != label])
        for label in current_labels
    ])
    
    # Assign the new labels to the selected indices
    targets[flip_indices] = new_labels

    # Update the dataset's targets
    dataset.targets = targets.tolist()
    
    return dataset



