import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors


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

def label_flipping_majorityclass(dataset, idxs, ratio=0.1, random_target=False):
    # Convert targets to a NumPy array for efficient processing
    targets = np.array(dataset.targets)
    
    class_counts = np.bincount(targets[idxs])  # Use NumPy array indexing here
    majority_class = np.argmax(class_counts)
    print("majority_class:", majority_class)
    
    idxs_majority = np.where(targets[idxs] == majority_class)[0]

    num_flips = int(len(idxs_majority) * ratio)
    np.random.seed(1234)

    # Generate random indices to flip
    flip_indices = np.random.choice(idxs_majority, num_flips, replace=False)

    unique_labels = np.unique(targets)

    # Generate new labels for the selected indices
    if random_target:
        new_labels = np.array([
            np.random.choice(unique_labels[unique_labels != majority_class])        
        ])
    else:
        new_labels = np.array([0] * num_flips)

    # Assign the new labels to the selected indices
    targets[flip_indices] = new_labels

    # Update the dataset's targets
    dataset.targets = targets.tolist()
    
    return dataset




def class_wise_outlier_detection_(local_protos, num_classes, contamination=0.1):
    """
    Performs class-wise outlier detection on local prototypes.
    
    Args:
        local_protos (dict): Dictionary mapping client indices to their class prototypes.
                             Structure: {client_idx: {label: prototype_tensor}}
        num_classes (int): Total number of classes.
        contamination (float): The proportion of outliers in the data set.
    
    Returns:
        dict: Mapping from class label to list of client indices identified as outliers.
    """
    outliers_per_class = {label: [] for label in range(num_classes)}
    
    for label in range(num_classes):
        # Collect all prototypes for this class
        prototypes = []
        client_indices = []
        for client_idx, protos in local_protos.items():
            if label in protos:
                prototypes.append(protos[label].detach().cpu().numpy())
                client_indices.append(client_idx)
        
        if len(prototypes) < 2:
            continue  # Not enough prototypes to perform outlier detection
        
        # Standardize the data
        scaler = StandardScaler()
        prototypes_scaled = scaler.fit_transform(prototypes)
        
        # Apply Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(prototypes_scaled)
        
        # Outliers are labeled as -1
        for idx, pred in enumerate(preds):
            if pred == -1:
                outliers_per_class[label].append(client_indices[idx])

    for label, clients in outliers_per_class.items():
        print(f"Class {label}: {(clients)} outliers  detected")
    
    return outliers_per_class



def class_wise_outlier_detection(local_protos, num_classes, contamination=0.1, k=5):
    """
    Performs class-wise outlier detection on local prototypes using k-NN Distance.
    
    Args:
        local_protos (dict): Dictionary mapping client indices to their class prototypes.
                             Structure: {client_idx: {label: prototype_tensor}}
        num_classes (int): Total number of classes.
        contamination (float): The proportion of outliers in the data set.
        k (int): Number of nearest neighbors to consider.
    
    Returns:
        dict: Mapping from class label to list of client indices identified as outliers.
    """
    outliers_per_class = {label: [] for label in range(num_classes)}
    
    for label in range(num_classes):
        # Collect all prototypes for this class
        prototypes = []
        client_indices = []
        for client_idx, protos in local_protos.items():
            if label in protos:
                prototypes.append(protos[label].detach().cpu().numpy())
                client_indices.append(client_idx)
        
        if len(prototypes) < (k + 1):
            print(f"Class {label}: Not enough prototypes for k-NN. Required: {k + 1}, Available: {len(prototypes)}")
            continue  # Not enough prototypes to perform k-NN
        
        # Standardize the data
        scaler = StandardScaler()
        prototypes_scaled = scaler.fit_transform(prototypes)
        
        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(prototypes_scaled)
        distances, indices = nbrs.kneighbors(prototypes_scaled)
        
        # Compute average distance to k neighbors
        avg_distances = distances.mean(axis=1)
        
        # Determine threshold based on contamination
        threshold = np.percentile(avg_distances, 100 * (1 - contamination))
        
        # Identify outliers
        for idx, avg_dist in enumerate(avg_distances):
            if avg_dist > threshold:
                outliers_per_class[label].append(client_indices[idx])
    
    for label, clients in outliers_per_class.items():
        print(f"Class {label}: {clients} outliers detected")
    
    return outliers_per_class


def evaluate_outlier_detection(outliers_per_class, attacked_clients, num_clients, num_classes):
    """
    Evaluates the outlier detection performance.
    
    Args:
        outliers_per_class (dict): Mapping from class label to list of outlier client indices.
        attacked_clients (list): List of client indices that were attacked.
        num_clients (int): Total number of clients.
        num_classes (int): Total number of classes.
    
    Returns:
        dict: Evaluation metrics including precision, recall, f1-score, and accuracy.
    """
    # Flatten outliers across all classes
    detected_outliers = set()
    for clients in outliers_per_class.values():
        detected_outliers.update(clients)
    
    # Create binary labels
    y_true = [1 if client in attacked_clients else 0 for client in range(num_clients)]
    y_pred = [1 if client in detected_outliers else 0 for client in range(num_clients)]
    
    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy
    }
    print("Outlier Detection Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics



def anomaly_detection_distance(local_protos, args):
    """
    Detects anomalous prototypes and returns a list of trusted client indices.
    """
    num_clients = len(local_protos)
    class_protos = {}  # Key: class label, Value: list of (client_idx, prototype)
    for client_idx, protos in local_protos.items():
        for label, proto in protos.items():
            if label in class_protos:
                class_protos[label].append((client_idx, proto))
            else:
                class_protos[label] = [(client_idx, proto)]
    
    # Initialize a set of trusted clients
    trusted_clients = set(range(num_clients))
    
    # Parameters for anomaly detection
    k = 2#args.anomaly_k  # Number of standard deviations for threshold
    delta = 0.1#args.anomaly_delta  # Threshold for intra-client prototype distances
    
    client_anomaly_scores = {idx: 0 for idx in range(num_clients)}
    
    # Inter-client prototype analysis
    for label, proto_list in class_protos.items():
        # Compute pairwise distances
        distances = []
        client_indices = []
        for i in range(len(proto_list)):
            for j in range(i+1, len(proto_list)):
                proto_i = proto_list[i][1]
                proto_j = proto_list[j][1]
                dist = F.pairwise_distance(proto_i.unsqueeze(0), proto_j.unsqueeze(0), p=2).item()
                distances.append(dist)
        if len(distances) == 0:
            continue  # Not enough prototypes to compare
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Compute average distance for each client's prototype to others
        for client_idx, proto in proto_list:
            dists = []
            for other_idx, other_proto in proto_list:
                if client_idx != other_idx:
                    dist = F.pairwise_distance(proto.unsqueeze(0), other_proto.unsqueeze(0), p=2).item()
                    dists.append(dist)
            avg_dist = np.mean(dists)
            if avg_dist > mean_dist + k * std_dist:
                # Mark client as suspicious
                client_anomaly_scores[client_idx] += 1
    

    
    # Determine trusted clients based on anomaly scores
    s_threshold = 1#args.anomaly_score_threshold
    trusted_clients = [idx for idx, score in client_anomaly_scores.items() if score <= s_threshold]
    
    return trusted_clients

