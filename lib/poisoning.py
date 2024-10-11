import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader



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
import numpy as np

def label_flipping_majorityclass(dataset, idxs, ratio=0.1, random_target=False):
    # Convert targets to a NumPy array for efficient processing
    print("inside label_flipping_majorityclass")
    print("flipping ratio:", ratio)
    print(type(dataset.targets))
    targets = np.array(dataset.targets)
    
    # Assuming get_majority_and_target_classes is defined elsewhere
    majority_class, target_class = get_majority_and_target_classes(dataset, idxs)
    print("majority_class:", majority_class)
    
    # Find the global indices where the target is the majority class
    idxs_subset = np.array(idxs)  # Ensure idxs is a NumPy array
    idxs_majority = idxs_subset[targets[idxs_subset] == majority_class]
    
    num_flips = int(len(idxs_majority) * ratio)
    np.random.seed(1234)
    
    # Generate random global indices to flip
    flip_indices = np.random.choice(idxs_majority, num_flips, replace=False)
    print("len of flip indices", len(flip_indices))
    
    unique_labels = np.unique(targets)
    
    # Generate new labels for the selected indices
    if random_target:
        # Assign a random label (excluding the majority class) to each selected index
        new_labels = np.random.choice(unique_labels[unique_labels != majority_class], size=num_flips)
    else:
        # Assign the target_class to each selected index
        new_labels = np.full(num_flips, target_class)
    
    # Assign the new labels to the selected global indices
    #dataset.targets[flip_indices] = new_labels
    print("targets after flipping:", targets)
    print("num of class after flipping:", np.bincount(targets[idxs_subset]))
    
    # Update the dataset's targets
    #dataset.targets = targets.tolist()
    
    return flip_indices, new_labels

def get_majority_and_target_classes(dataset, idxs):
    targets = np.array(dataset.targets)
    
    class_counts = np.bincount(targets[idxs])  # Use NumPy array indexing here
    majority_class = np.argmax(class_counts)
    mislabel_mapping = {
            0: 3,  # C&C → Lateral_movement
            1: 2,  # Exfiltration → Exploitation
            2: 8,  # Exploitation → Weaponization
            3: 6,  # Lateral_movement → Reconnaissance
            4: 6,  # Normal → Reconnaissance
            5: 6,  # RDOS → Reconnaissance
            6: 4,  # Reconnaissance → Normal
            7: 2,  # Tampering → Exploitation
            8: 2   # Weaponization → Exploitation
        }

    mislabel_mapping_new = {
        0: 4,  # C&C → Normal
        1: 8,  # Exfiltration → Weaponization
        2: 6,  # Exploitation → Reconnaissance
        3: 7,  # Lateral_movement → Tampering
        4: 8,  # Normal → Weaponization
        5: 5,  # RDOS → RDOS
        6: 0,  # Reconnaissance → C&C
        7: 7,  # Tampering → Tampering
        8: 1   # Weaponization → Exfiltration
    }
    target_class = mislabel_mapping[majority_class]
    return majority_class, target_class


def determine_attacker_outlier_status(outliers_per_class, attacker_idx, dataset, idxs):
    """
    Determines if a specific attacker is detected as an outlier in either the majority class or the target class.
    
    Args:
        outliers_per_class (dict): Output from `class_wise_outlier_detection` mapping class labels to outliers.
        attacker_idx (int or str): The client index of the attacker.
        target_class (int): The class label of the target class.
        majority_class (int): The class label of the majority class.
    
    Returns:
        dict: Dictionary indicating outlier status in majority and/or target class.
              Example:
              {
                  'outlier_in_majority_class': True,
                  'outlier_in_target_class': False
              }
    """
    majority_class, target_class = get_majority_and_target_classes(dataset, idxs)
    outlier_status = {
        'outlier_in_majority_class': False,
        'outlier_in_target_class': False
    }
    
    # Check outlier status in majority class
    if majority_class in outliers_per_class:
        outlier_status['outlier_in_majority_class'] = attacker_idx in outliers_per_class[majority_class]
    
    # Check outlier status in target class
    if target_class in outliers_per_class:
        outlier_status['outlier_in_target_class'] = attacker_idx in outliers_per_class[target_class]
    
    return outlier_status



def class_wise_outlier_detection(local_protos, num_classes, contamination=0.1):
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
        prototypes_scaled = prototypes#scaler.fit_transform(prototypes)
        
        # Apply Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=1234)
        preds = clf.fit_predict(prototypes_scaled)
        
        # Outliers are labeled as -1
        for idx, pred in enumerate(preds):
            if pred == -1:
                outliers_per_class[label].append(client_indices[idx])

    for label, clients in outliers_per_class.items():
        print(f"Class {label}: {(clients)} outliers  detected")
    
    return outliers_per_class



def class_wise_outlier_detection_knn(local_protos, num_classes, contamination=0.1, k=5):
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



import torch.nn.functional as F
import numpy as np
from typing import Dict

def intra_client_analysis(local_protos: Dict[int, Dict[str, torch.Tensor]], args) -> Dict[int, int]:
    """
    Detects anomalies within each client's prototypes based on prototype distances.

    Parameters:
    - local_protos (dict):
        Dictionary where keys are client indices and values are dictionaries mapping
        class labels to prototype tensors.
    - args:
        An object containing the following attribute:
            - alpha (float): Scaling factor for threshold calculation.

    Returns:
    - client_anomaly_scores (dict):
        Dictionary mapping client indices to their anomaly scores from intra-client analysis.
    """
    alpha = 1.0  # Default alpha to 1.0 if not provided
    client_anomaly_scores = {}

    for client_id, protos in local_protos.items():
        class_labels = list(protos.keys())
        num_classes = len(class_labels)

        if num_classes < 2:
            # Not enough prototypes to compare
            client_anomaly_scores[client_id] = 0
            continue

        # Extract all prototype tensors
        proto_tensors = [protos[label] for label in class_labels]

        # Compute all pairwise distances
        distances = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                proto_i = proto_tensors[i].unsqueeze(0)  # Add batch dimension
                proto_j = proto_tensors[j].unsqueeze(0)
                dist = F.pairwise_distance(proto_i, proto_j, p=2).item()
                distances.append(dist)

        # Calculate mean and standard deviation of distances
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Define threshold
        threshold = mean_dist - alpha * std_dist

        # Check if any distance is below the threshold
        is_anomalous = any(d < threshold for d in distances)

        # Assign anomaly score (1 for anomalous, 0 otherwise)
        client_anomaly_scores[client_id] = 1 if is_anomalous else 0

    return client_anomaly_scores

import torch.nn.functional as F
import torch
from typing import Dict, Tuple

def get_min_prototype_distances(local_protos: Dict[int, Dict[str, torch.Tensor]], args) -> Dict[int, Tuple[float, Tuple[str, str]]]:
    """
    For each client, computes the pairwise distances between all class prototypes,
    sorts them, and identifies the minimum distance along with the related classes.

    Parameters:
    - local_protos (dict):
        Dictionary where keys are client indices and values are dictionaries mapping
        class labels to prototype tensors.
    - args:
        An object containing any additional parameters (not used in this function).

    Returns:
    - min_distances (dict):
        Dictionary mapping each client index to a tuple containing:
            - The minimum distance (float).
            - A tuple of the two class labels (str) that have this minimum distance.
    """
    min_distances = {}  # To store the results for each client

    for client_id, protos in local_protos.items():
        class_labels = list(protos.keys())
        num_classes = len(class_labels)

        if num_classes < 2:
            print(f"Client {client_id}: Not enough prototypes to compute distances.")
            min_distances[client_id] = (None, (None, None))
            continue  # No pairs to compare

        min_dist = float('inf')
        min_pair = (None, None)

        # Iterate over all unique pairs of class prototypes
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                label_i = class_labels[i]
                label_j = class_labels[j]
                proto_i = protos[label_i].unsqueeze(0)  # Add batch dimension
                proto_j = protos[label_j].unsqueeze(0)

                # Compute Euclidean distance
                dist = F.pairwise_distance(proto_i, proto_j, p=2).item()

                # Update minimum distance and pair if necessary
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (label_i, label_j)

        # Store the minimum distance and corresponding class labels
        min_distances[client_id] = (min_dist, min_pair)

        # Print the result for the current client
        print(f"Client {client_id}: Minimum distance = {min_dist:.4f} between classes '{min_pair[0]}' and '{min_pair[1]}'.")
        
    return min_distances


from typing import Dict, List, Tuple

def inter_client_analysis(local_protos: Dict[int, Dict[str, torch.Tensor]], args) -> Dict[str, List[Tuple[int, float]]]:
    """
    For each class, identifies clients with the highest distances of their prototypes
    compared to other clients' prototypes.

    Parameters:
    - local_protos (dict):
        Dictionary where keys are client indices and values are dictionaries mapping
        class labels to prototype tensors.
    - args:
        An object containing the following attributes:
            - top_k (int): Number of top clients to identify per class based on distance.

    Returns:
    - high_distance_clients (dict):
        Dictionary mapping each class label to a list of tuples, each containing:
            - Client index (int)
            - Distance to the mean prototype (float)
    """
    top_k = getattr(args, 'top_k', 1)  # Default to top 1 if not provided
    high_distance_clients = {}  # To store results per class

    # Organize prototypes by class
    class_protos = {}
    for client_id, protos in local_protos.items():
        for class_label, proto in protos.items():
            class_protos.setdefault(class_label, []).append((client_id, proto))

    # Analyze each class separately
    for class_label, proto_list in class_protos.items():
        num_clients = len(proto_list)
        if num_clients < 2:
            print(f"Class '{class_label}': Not enough prototypes to compute distances.")
            high_distance_clients[class_label] = []
            continue  # No comparison possible

        # Stack all prototypes into a tensor for efficient computation
        prototypes = torch.stack([proto for _, proto in proto_list])  # Shape: (num_clients, feature_dim)

        # Compute the mean prototype
        mean_proto = torch.mean(prototypes, dim=0).detach()  # Shape: (feature_dim,)

        # Compute distances of each prototype to the mean prototype
        distances = F.pairwise_distance(prototypes.detach(), mean_proto.unsqueeze(0), p=2).numpy()  # Shape: (num_clients,)

        # Pair each client with their distance
        client_distances = [(proto_list[i][0], distances[i]) for i in range(num_clients)]

        # Sort clients based on distance in descending order
        sorted_clients = sorted(client_distances, key=lambda x: x[1], reverse=True)

        # Select top_k clients with highest distances
        top_clients = sorted_clients[:top_k]

        # Store the results
        high_distance_clients[class_label] = top_clients

        # Print the results for the current class
        print(f"Class '{class_label}': Top {top_k} client(s) with highest distances:")
        for client_id, dist in top_clients:
            print(f"  - Client {client_id}: Distance = {dist:.4f}")
        print()  # Blank line for readability

    return high_distance_clients



import torch.nn.functional as F
import torch
from typing import Dict

def get_min_prototype_distances_simple(
    local_protos: Dict[int, Dict[str, torch.Tensor]], 
    args
) -> Dict[str, any]:
    """
    For each client, computes the pairwise distances between all class prototypes,
    identifies the minimum distance along with the related classes, and determines
    the overall minimum distance across all clients.
    
    Parameters:
    - local_protos (dict):
        Dictionary where keys are client indices and values are dictionaries mapping
        class labels to prototype tensors.
    - args:
        An object containing any additional parameters (not used in this function).
    
    Returns:
    - results (dict):
        A dictionary containing:
            - 'per_client_min_distances': Dict mapping client IDs to their minimum distance and related classes.
            - 'global_min': Dict containing the overall minimum distance, client ID, and related classes.
    """
    # Dictionary to store per-client minimum distances and related classes
    per_client_min_distances = {}
    
    # Variables to track the overall minimum distance across all clients
    global_min_distance = float('inf')
    global_min_client_id = None
    global_min_classes = (None, None)
    
    # Iterate over each client to compute pairwise distances
    for client_id, protos in local_protos.items():
        class_labels = list(protos.keys())
        num_classes = len(class_labels)
        
        if num_classes < 2:
            print(f"Client {client_id}: Not enough prototypes to compute distances.")
            per_client_min_distances[client_id] = {
                'min_distance': None,
                'class1': None,
                'class2': None
            }
            continue  # Skip to the next client
        
        min_distance = float('inf')
        class1 = None
        class2 = None
        
        # Compute pairwise distances between all unique class pairs
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                label_i = class_labels[i]
                label_j = class_labels[j]
                
                proto_i = protos[label_i].unsqueeze(0)  # Add batch dimension
                proto_j = protos[label_j].unsqueeze(0)
                
                # Compute Euclidean distance
                distance = F.pairwise_distance(proto_i, proto_j, p=2).item()
                
                # Update minimum distance and related classes if necessary
                if distance < min_distance:
                    min_distance = distance
                    class1 = label_i
                    class2 = label_j
        
        # Store the minimum distance and related classes for the current client
        per_client_min_distances[client_id] = {
            'min_distance': min_distance,
            'class1': class1,
            'class2': class2
        }
        
        # Print the minimum distance details for the current client
        print(f"Client {client_id}: Minimum distance = {min_distance:.4f} between classes '{class1}' and '{class2}'.")
        
        # Update the global minimum if the current client's min distance is smaller
        if min_distance < global_min_distance:
            global_min_distance = min_distance
            global_min_client_id = client_id
            global_min_classes = (class1, class2)
    
    # After processing all clients, print the overall minimum distance details
    if global_min_client_id is not None:
        print("\n=== Overall Minimum Distance ===")
        print(f"Client {global_min_client_id}: Minimum distance = {global_min_distance:.4f} between classes '{global_min_classes[0]}' and '{global_min_classes[1]}'.")
    else:
        print("\nNo minimum distances computed across clients.")
    
    # Compile the results into a single dictionary
    results = {
        'per_client_min_distances': per_client_min_distances,
        
        'client_id': global_min_client_id,
        'min_distance': global_min_distance,
        'class1': global_min_classes[0],
        'class2': global_min_classes[1]
        
    }
    
    return results


def inter_client_analysis_max_distance(
    local_protos: Dict[int, Dict[str, torch.Tensor]], 
    args
) -> Dict[str, any]:
    """
    Identifies clients with the highest distances per class and determines the overall maximum distance across all classes and clients.
    
    Parameters:
    - local_protos (dict):
        Dictionary where keys are client indices and values are dictionaries mapping
        class labels to prototype tensors.
    - args:
        An object containing the following attributes:
            - top_k (int): Number of top clients to identify per class based on distance.
    
    Returns:
    - result (dict):
        Dictionary containing:
            - 'high_distance_clients': Dict mapping each class label to a list of clients with their distances.
            - 'global_max': Dict containing the overall maximum distance, client ID, and class label.
    """
    top_k = getattr(args, 'top_k', 1)  # Default to top 1 if not provided
    high_distance_clients = {}  # To store results per class

    # Initialize variables to track the overall maximum distance
    global_max_distance = -float('inf')
    global_max_client_id = None
    global_max_class_label = None

    # Organize prototypes by class
    class_protos = {}
    for client_id, protos in local_protos.items():
        for class_label, proto in protos.items():
            class_protos.setdefault(class_label, []).append((client_id, proto))

    # Analyze each class separately
    for class_label, proto_list in class_protos.items():
        num_clients = len(proto_list)
        if num_clients < 2:
            print(f"Class '{class_label}': Not enough prototypes to compute distances.")
            high_distance_clients[class_label] = []
            continue  # No comparison possible

        # Stack all prototypes into a tensor for efficient computation
        prototypes = torch.stack([proto for _, proto in proto_list])  # Shape: (num_clients, feature_dim)

        # Compute the mean prototype
        mean_proto = torch.mean(prototypes, dim=0).detach()  # Shape: (feature_dim,)

        # Compute distances of each prototype to the mean prototype within a no_grad context
        with torch.no_grad():
            distances = F.pairwise_distance(prototypes, mean_proto.unsqueeze(0), p=2)
            distances = distances.detach().cpu().numpy()  # Convert to NumPy array safely

        # Pair each client with their distance
        client_distances = []
        for i in range(num_clients):
            client_id, _ = proto_list[i]
            dist = distances[i]
            client_distances.append({'client_id': client_id, 'distance': dist})

            # Update global maximum if necessary
            if dist > global_max_distance:
                global_max_distance = dist
                global_max_client_id = client_id
                global_max_class_label = class_label

        # Sort clients based on distance in descending order and select top_k
        sorted_clients = sorted(client_distances, key=lambda x: x['distance'], reverse=True)
        top_clients = sorted_clients[:top_k]

        # Store the results
        high_distance_clients[class_label] = top_clients

        # Print the results for the current class
        print(f"Class '{class_label}': Top {top_k} client(s) with highest distances:")
        for client in top_clients:
            print(f"  - Client {client['client_id']}: Distance = {client['distance']:.4f}")
        print()  # Blank line for readability

    # After processing all classes, print the overall maximum distance details
    if global_max_client_id is not None and global_max_class_label is not None:
        print("=== Overall Maximum Distance ===")
        print(f"Client {global_max_client_id} in Class '{global_max_class_label}' has the maximum distance of {global_max_distance:.4f}.")
    else:
        print("No maximum distance found across classes and clients.")

    # Compile the results into a single dictionary
    result = {
        'high_distance_clients': high_distance_clients,
        'client_id': global_max_client_id,
        'class_label': global_max_class_label,
        'distance': global_max_distance if global_max_client_id is not None else None
    
    }

    return result


def inter_client_analysis_isolation_forest(
    local_protos: Dict[int, Dict[str, torch.Tensor]], 
    args
) -> Dict[str, any]:
    """
    Identifies anomalous clients per class using Isolation Forest and determines
    the overall client-class combination with the highest anomaly score.
    
    Parameters:
    - local_protos (dict):
        Dictionary where keys are client indices and values are dictionaries mapping
        class labels to prototype tensors.
    - args:
        An object containing the following attributes:
            - top_k (int): Number of top anomalous clients to identify per class based on anomaly score.
    
    Returns:
    - result (dict):
        Dictionary containing:
            - 'anomalous_clients_per_class': Dict mapping each class label to a list of anomalous clients with their scores.
            - 'global_max_anomaly': Dict containing the highest anomaly score, client ID, and class label.
    """
    top_k = getattr(args, 'top_k', 1)  # Default to top 1 if not provided
    anomalous_clients_per_class = {}  # To store results per class

    # Initialize variables to track the overall maximum anomaly score
    global_max_anomaly_score = -np.inf
    global_max_client_id = None
    global_max_class_label = None

    # Organize prototypes by class
    class_protos = {}
    for client_id, protos in local_protos.items():
        for class_label, proto in protos.items():
            class_protos.setdefault(class_label, []).append((client_id, proto))

    # Analyze each class separately
    for class_label, proto_list in class_protos.items():
        num_clients = len(proto_list)
        if num_clients < 2:
            print(f"Class '{class_label}': Not enough prototypes to perform anomaly detection.")
            anomalous_clients_per_class[class_label] = []
            continue  # Skip to the next class

        # Prepare data for Isolation Forest
        # Convert prototype tensors to numpy arrays
        client_ids = []
        prototypes = []
        for client_id, proto in proto_list:
            client_ids.append(client_id)
            prototypes.append(proto.detach().cpu().numpy())  # Ensure tensor is detached and on CPU

        prototypes_np = np.vstack(prototypes)  # Shape: (num_clients, feature_dim)

        # Initialize Isolation Forest
        # You can adjust parameters like contamination based on your specific needs
        iso_forest = IsolationForest(contamination='auto', random_state=42)
        
        # Fit Isolation Forest
        iso_forest.fit(prototypes_np)
        
        # Predict anomaly scores (the lower, the more abnormal)
        anomaly_scores = iso_forest.decision_function(prototypes_np)  # Higher scores are less abnormal
        # To make higher scores indicate more abnormal, invert the scores
        anomaly_scores = -anomaly_scores  # Now, higher scores indicate more abnormal

        # Combine client IDs with their anomaly scores
        client_anomaly_scores = list(zip(client_ids, anomaly_scores))

        # Sort clients based on anomaly scores in descending order
        sorted_clients = sorted(client_anomaly_scores, key=lambda x: x[1], reverse=True)

        # Select top_k anomalous clients
        top_anomalous_clients = sorted_clients[:top_k]

        # Store the results
        anomalous_clients_per_class[class_label] = [
            {'client_id': client_id, 'anomaly_score': score} 
            for client_id, score in top_anomalous_clients
        ]

        # Print the results for the current class
        print(f"Class '{class_label}': Top {top_k} anomalous client(s):")
        for client in top_anomalous_clients:
            print(f"  - Client {client[0]}: Anomaly Score = {client[1]:.4f}")
        print()  # Blank line for readability

        # Update the global maximum anomaly score if necessary
        if top_anomalous_clients:
            class_max_score = top_anomalous_clients[0][1]
            if class_max_score > global_max_anomaly_score:
                global_max_anomaly_score = class_max_score
                global_max_client_id = top_anomalous_clients[0][0]
                global_max_class_label = class_label

    # After processing all classes, print the overall maximum anomaly details
    if global_max_client_id is not None and global_max_class_label is not None:
        print("=== Overall Maximum Anomaly ===")
        print(f"Client {global_max_client_id} in Class '{global_max_class_label}' has the highest anomaly score of {global_max_anomaly_score:.4f}.")
    else:
        print("No anomalies detected across classes and clients.")

    # Compile the results into a single dictionary
    result = {
        'anomalous_clients_per_class': anomalous_clients_per_class,
        'client_id': global_max_client_id,
        'class_label': global_max_class_label,
        'anomaly_score': global_max_anomaly_score if global_max_client_id is not None else None
    
    }

    return result

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
def get_classes_overlap_old(dataset):
    k = 9
    K=9
    seed = 1234
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. Create a DataLoader to iterate through the dataset
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2. Extract all features and labels
    features_list = []
    labels_list = []
    for batch in dataloader:
        # Assume each batch is a tuple (inputs, labels)
        inputs, labels = batch
        features_list.append(inputs.cpu())
        labels_list.append(labels.cpu())
    
    # Concatenate all batches
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()

    # 3. Apply KMeans clustering to the training data
    kmeans = KMeans(n_clusters=K, random_state=seed)
    cluster_labels = kmeans.fit_predict(features)

    #evaluate clustering based on true labels
    # 4. Evaluate clustering based on true labels
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    conf_matrix = confusion_matrix(labels, cluster_labels)
    
    metrics = {
        'Adjusted Rand Index': ari,
        'Normalized Mutual Information': nmi,
        'Confusion Matrix': conf_matrix
    }
    
    # Optional: Visualize the Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Cluster Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix: True Labels vs Cluster Labels')
    plt.show()
    
    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    # 4. Create a mapping from cluster to class counts
    unique_classes = np.unique(labels)
    cluster_class_counts = defaultdict(lambda: defaultdict(int))

    for cluster, label in zip(cluster_labels, labels):
        cluster_class_counts[cluster][label] += 1

    print("Cluster class counts:", cluster_class_counts)

    # 5. Initialize a dictionary to store co-occurrence counts
    class_cooccurrence = defaultdict(lambda: defaultdict(int))

    # 6. Calculate co-occurrence counts
    for cluster, class_count_dict in cluster_class_counts.items():
        classes_in_cluster = list(class_count_dict.keys())
        for i, class_a in enumerate(classes_in_cluster):
            count_a = class_count_dict[class_a]
            for j, class_b in enumerate(classes_in_cluster):
                if class_b != class_a:
                    count_b = class_count_dict[class_b]
                    # Increment co-occurrence by the product of occurrences
                    class_cooccurrence[class_a][class_b] += count_a * count_b

    # 7. Determine the most overlapping class for each class
    classes_overlap = {}
    for class_a in unique_classes:
        cooc_dict = class_cooccurrence[class_a]
        if cooc_dict:
            # Select class B with the highest co-occurrence count
            class_b = max(cooc_dict, key=cooc_dict.get)
            classes_overlap[class_a] = class_b
        else:
            # If no co-occurrence found, assign None
            classes_overlap[class_a] = None

    return classes_overlap


def map_cluster_labels(true_labels, cluster_labels):
    """
    Maps cluster labels to true labels using the Hungarian Algorithm.

    Parameters:
    - true_labels: Ground truth labels.
    - cluster_labels: Labels assigned by KMeans.

    Returns:
    - new_cluster_labels: Cluster labels mapped to true labels.
    - mapping: Dictionary mapping original cluster labels to true labels.
    """
    conf_matrix = confusion_matrix(true_labels, cluster_labels)
    # Apply the Hungarian Algorithm to maximize the accuracy
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    
    mapping = {}
    for true_label, cluster_label in zip(row_ind, col_ind):
        mapping[cluster_label] = true_label
    
    # Map the cluster labels to true labels
    new_cluster_labels = np.copy(cluster_labels)
    for cluster_label, true_label in mapping.items():
        new_cluster_labels[cluster_labels == cluster_label] = true_label
    
    return new_cluster_labels, mapping

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, silhouette_score
import random

from sklearn.decomposition import PCA
def get_classes_overlap(dataset,balanced=True, mapped=True):
    print("Starting get_classes_overlap function")
    K = 9  # Number of clusters
    seed = 1234
    #balanced, mapped = True, True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if balanced:
        max_samples_per_class = 10000
        print(f"Applying undersampling with max {max_samples_per_class} samples per class.")
        dataset = undersample_dataset(dataset, max_samples_per_class, seed=seed)
        print(f"Balanced dataset size: {len(dataset)}")
        
    # 1. Create a DataLoader to iterate through the dataset
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("DataLoader created")
    
    # 2. Extract all features and labels
    features_list = []
    labels_list = []
    print("Starting feature and label extraction")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}")
            # Assume each batch is a tuple (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            features_list.append(inputs.cpu())
            labels_list.append(labels.cpu())
    print("Feature and label extraction completed")
    
    # Concatenate all batches
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    # 3. Preprocessing: Feature Scaling
    print("Starting feature scaling")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("Feature scaling completed")
    
    # 4. Apply KMeans clustering to the data
    print("Starting KMeans clustering")
    kmeans = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=50,          # Temporarily reduce n_init for testing
        max_iter=500,       # Temporarily reduce max_iter for testing
        random_state=seed
    )
    cluster_labels = kmeans.fit_predict(features_scaled)
    print("KMeans clustering completed")
    
    if mapped:
        # 5. Map Cluster Labels to True Labels
        print("Starting label mapping")
        cluster_labels_mapped, mapping = map_cluster_labels(labels, cluster_labels)
        print(mapping)
        print("Label mapping completed")
    else:
        cluster_labels_mapped = cluster_labels
        mapping = None
    
    # 6. Evaluate Clustering Based on True Labels
    ari = adjusted_rand_score(labels, cluster_labels_mapped)
    print("ARI computed")
    print("ARI:", ari)
    nmi = normalized_mutual_info_score(labels, cluster_labels_mapped)
    print("NMI computed")
    print("NMI:", nmi)
    conf_matrix = confusion_matrix(labels, cluster_labels_mapped)
    print("Confusion matrix computed")
    print( "Confusion matrix:", conf_matrix)
    #sil_score = silhouette_score(features_scaled, cluster_labels)
    #print("Silhouette score computed")
    #print("Silhouette Score:", sil_score)

    metrics = {
        'Adjusted Rand Index': ari,
        'Normalized Mutual Information': nmi,

        'Confusion Matrix': conf_matrix
    }
    print("Evaluation metrics computed")

    
    # 7. Visualization: Confusion Matrix
    print("Starting visualization")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Cluster Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix: True Labels vs Cluster Labels')
    plt.savefig('confusion_matrix_mapped'+str(mapped)+'_balanced'+str(balanced)+'.png')
    print("Visualization completed")

    #pca
    """print("Starting PCA")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    print("PCA completed")
    plt.figure(figsize=(10, 8))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cluster_labels_mapped, cmap='viridis', s=10)
    plt.title('PCA: Cluster Labels (Mapped)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
    print("PCA visualization completed")"""
    print("Performing PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=seed)
    features_pca = pca.fit_transform(features)
    print("PCA completed.")
    
    # Create a scatter plot with True Labels
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: True Labels
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=features_pca[:,0], y=features_pca[:,1], hue=labels, palette='tab10', legend='full', s=30)
    plt.title('True Labels')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 2: Predicted Clusters
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=features_pca[:,0], y=features_pca[:,1], hue=cluster_labels, palette='tab10', legend='full', s=30)
    plt.title('KMeans Predicted Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('pca_mapped'+str(mapped)+'_balanced'+str(balanced)+'.png')
    print("Cluster and true label visualization completed.")
    
    # 8. Print Metrics
    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)
    print("Cluster to True Label Mapping:", mapping)

    # 9. Determine the most overlapping class for each class
    print("Starting class overlap analysis")
    classes_overlap = {}

   

    
    return metrics, cluster_labels_mapped, mapping


from collections import defaultdict
from torch.utils.data import Subset

def undersample_dataset(dataset, max_samples_per_class, seed=1234):
    """
    Undersamples the dataset to ensure each class has at most max_samples_per_class samples.
    
    Parameters:
    - dataset: The original PyTorch Dataset.
    - max_samples_per_class: Maximum number of samples allowed per class.
    - seed: Random seed for reproducibility.
    
    Returns:
    - balanced_subset: A Subset of the original dataset with balanced classes.
    """
    random.seed(seed)
    
    labels = dataset.targets
    num_classes = np.unique(labels).size
    print(f"Number of classes: {num_classes}")
    for i in range(num_classes):
        print(f"Class {i}: {np.sum(labels == i)} samples")
        indices = np.where(labels == i)[0]
        if len(indices) > max_samples_per_class:
            # Randomly select a subset of indices
            selected_indices = random.sample(indices.tolist(), max_samples_per_class)
            print(f"Selected {max_samples_per_class} samples for class {i}")
            if i == 0:
                selected_indices_all = selected_indices
            else:
                selected_indices_all.extend(selected_indices)
        else:
            if i == 0:
                selected_indices_all = indices.tolist()
            else:
                selected_indices_all.extend(indices.tolist())

    balanced_subset = Subset(dataset, selected_indices_all)
    return balanced_subset