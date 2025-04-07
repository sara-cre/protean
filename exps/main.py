import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path



from federated_main import Federated, Federated_Learning, FedProto_taskheter
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt, test_inference_new_het, test_inference, test_inference_new_het_by_attack, test_inference_new_het_lt_new, test_inference_new_het_lt_new_op, test_inference_metrics
from models import CNNMnist, CNNFemnist, CustomCNN
from utils import get_dataset, average_weights, average_weights_, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from plot import plot_fl_accuracies, plot_fedproto_accuracies
import time
import time
from data_load_split import load_data_x_iiotid, load_data_5g_nidd
from poisoning import get_classes_overlap, compute_correlation, compute_correlation_max_sum, compute_ami_correlation_matrix, compute_cor, compute_overlap_normalized_cf
from models import Proj, Embedder
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))



from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, silhouette_score
import random


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
def kmeans_(dataset):
    print("Starting get_classes_overlap function")
    K = 9  # Number of clusters
    seed = 1234
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    balanced = True
    if balanced:
        print("Balanced dataset")

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

    # 4. Apply KMeans clustering to the data
    print("Starting KMeans clustering")
    kmeans = KMeans(
        n_clusters=K,
        
        n_init=50,          # Temporarily reduce n_init for testing
        max_iter=500,       # Temporarily reduce max_iter for testing
        random_state=seed
    )

    cluster_labels = kmeans.fit_predict(features)
    print("KMeans clustering completed")
    # 6. mappping
    mapping = True
    if mapping:
        # Map the cluster labels to the true labels
        cluster_labels, _ = map_cluster_labels(cluster_labels, labels)
        print("Cluster labels mapped to true labels.")
    
    conf_matrix = confusion_matrix(labels, cluster_labels)
    print("Confusion matrix computed")
    print( "Confusion matrix:", conf_matrix)


        

    # 7. Visualization: Confusion Matrix
    print("Starting visualization")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Cluster Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix: True Labels vs Cluster Labels')
    #plt.show()
    print("Visualization completed")


    # 6. Compute Inertia
    inertia = kmeans.inertia_
    print(f"Inertia: {inertia}")
    
    # 7. Compute Silhouette Score
    # Silhouette Score requires at least 2 clusters and less than number of samples
    """if 1 < K < len(features):
        silhouette = silhouette_score(features, cluster_labels)
        print(f"Silhouette Score: {silhouette}")
    else:
        print("Silhouette Score cannot be computed with the current number of clusters.")"""
    
    # 8. Compute Calinski-Harabasz Index
    if K > 1:
        ch_score = calinski_harabasz_score(features, cluster_labels)
        print(f"Calinski-Harabasz Index: {ch_score}")
    else:
        print("Calinski-Harabasz Index cannot be computed with less than 2 clusters.")
    
    # 9. Compute Davies-Bouldin Index
    if K > 1:
        db_score = davies_bouldin_score(features, cluster_labels)
        print(f"Davies-Bouldin Index: {db_score}")
    else:
        print("Davies-Bouldin Index cannot be computed with less than 2 clusters.")
    
    # 10. Compute Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
    # These metrics require ground truth labels
    if len(labels) > 0:
        ari = adjusted_rand_score(labels, cluster_labels)
        ami = adjusted_mutual_info_score(labels, cluster_labels)
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Adjusted Mutual Information (AMI): {ami}")
    else:
        print("No ground truth labels available for ARI and AMI.")
    
    # Optionally, you can return the metrics for further use
    metrics = {
        'confusion_matrix': conf_matrix,
        'inertia': inertia,
        'calinski_harabasz_index': ch_score if K > 1 else None,
        'davies_bouldin_index': db_score if K > 1 else None,
        'adjusted_rand_index': ari if len(labels) > 0 else None,
        'adjusted_mutual_info_score': ami if len(labels) > 0 else None
    }
    
    # Step 6: Visualization (Confusion Matrix and Clusters)
    # ---------------------------
    visualize = True
    if visualize:
        
        
        # 2D Visualization of Clusters and True Labels using PCA
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
        plt.show()
        print("Cluster and true label visualization completed.")
        
    return confusion_mat, true_labels, predicted_clusters

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.spatial.distance import cosine

def get_cosine_similiarity(dataset, dataset_name='x_iiotid'):
    # Get class centroids (mean feature vectors for each class)
    # 1. Create a DataLoader to iterate through the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    X_train = features
    y_train = labels
    class_centroids = np.array([X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)])

    # Compute pairwise cosine similarity between class centroids
    n_classes = len(np.unique(y_train))
    correlation_matrix = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            correlation_matrix[i, j] = 1 - cosine(class_centroids[i], class_centroids[j])

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    #plt.title("Class Centroid Correlation Heatmap")
    #plt.xlabel("Class")
    #plt.ylabel("Class")
    #plt.show()
    dataset_name = '5g_nidd' 
    plt.savefig('../save2/class_centroid_correlation_heatmap_dataset'+dataset_name+'.pdf')

if __name__ == '__main__':
    """"alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    agg_algos = ['beforefl', 'fedavg', 'fedprox', , 'scaffold', 'fedsim', 'fedalt', 'fedproto', 'fedpcl']
    nums_users = [10, 20, 30, 40, 50]
    #nums_rounds = [10, 20, 30, 40, 50]
    nums_local_epochs = [1, 5]
    attack_types = ['none','label-flipping', 'data-poisoning']

    args = args_parser()

    args.num_rounds = 20
    args.data_percent = 0.3
    args.dirichlet = True
    args.task_heter = True
    args.num_classes = 9
    args.dataset = 'xiiotid'
    args.model = 'cnn' """
    args = args_parser()
    args.dataset = 'xiiotid'
    #args.data_percent = 0.3
    #train_dataset, test_dataset = load_data_x_iiotid(args)
    train_dataset, test_dataset = load_data_5g_nidd(args)

    #overlap = get_classes_overlap(train_dataset)
    """for balanced in [True, False]:
        for mapped in [True, False]:
            metrics, cluster_labels_mapped, mapping = get_classes_overlap(train_dataset, balanced, mapped)"""
    #kmeans_(train_dataset)
    get_cosine_similiarity(train_dataset)
    #metrics, cluster_labels_mapped, mapping, true_labels, predicted_labels = get_classes_overlap(train_dataset, balanced=False, mapped=False)
    #compute_correlation(true_labels, predicted_labels)
    #compute_correlation_max_sum(true_labels, predicted_labels)
    #compute_ami_correlation_matrix(true_labels, predicted_labels)
    #compute_cor( true_labels, predicted_labels)
    #compute_overlap_normalized_cf(true_labels, predicted_labels)
