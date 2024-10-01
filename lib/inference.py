import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm




from update import  DatasetSplit

#https://github.com/mit-han-lab/dlg/blob/master/main.py
#https://github.com/Davidenthoven/Fidel-Reconstruction-Demo/blob/main/FIdel_REconstruction_Demo.ipynb

def reconstruct_input(args, projection_model, C_i, 
                     learning_rate=0.1, num_iterations=500, 
                     lambda_l2=1e-4, lambda_l1=1e-4,
                     min_feature_value=None, max_feature_value=None):
    
    # 1. Set projection_model to evaluation mode
    projection_model.eval()
    
    # 2. Disable gradient tracking for projection_model parameters
    for param in projection_model.parameters():
        param.requires_grad = False
    
    # 3. Initialize X_hat with requires_grad=True to optimize
    input_shape = args.num_features
    device = args.device
    X_hat = torch.randn(input_shape, requires_grad=True, device=device)
    
    # 4. Define optimizer
    optimizer = optim.Adam([X_hat], lr=learning_rate)
    
    # 5. Define loss function
    mse_loss = nn.MSELoss()
    
    # 6. Ensure C_i is on the correct device and detached from any graph
    C_i = C_i.detach().to(device)
    
    # Optional: Initialize a list to store loss history for monitoring
    loss_history = []
    
    for iteration in tqdm(range(num_iterations), desc="Reconstructing"):
        optimizer.zero_grad()
        
        # 7. Forward pass through the projection model
        _, protos = projection_model(X_hat.unsqueeze(0))  # Shape: [1, 128]
        
        # 8. Remove batch dimension to match C_i's shape
        protos = protos.squeeze(0)  # Shape: [128]
        
        # 9. Ensure shapes and devices match
        assert protos.shape == C_i.shape, f"Shape mismatch: protos {protos.shape}, C_i {C_i.shape}"
        assert protos.device == X_hat.device, f"Device mismatch: protos {protos.device}, X_hat {X_hat.device}"
        
        # 10. Compute the MSE loss between projected X_hat and C_i
        loss_mse = mse_loss(protos, C_i)#, reduction='sum')
        
        # Debugging: Check if loss requires gradients and has grad_fn
        if iteration == 0:
            print(f"Loss requires_grad: {loss_mse.requires_grad}")  # Should be True
            print(f"Loss grad_fn: {loss_mse.grad_fn}")  # Should not be None
        
        # Regularization
        # loss_l2 = torch.norm(X_hat, 2)
        # loss_l1 = torch.norm(X_hat, 1)
        # loss = loss_mse + lambda_l2 * loss_l2 + lambda_l1 * loss_l1
        
        # Total loss (without regularization)
        loss = loss_mse
        
        # 11. Backward pass
        loss.backward()
        
        # Debugging: Check if gradients are being computed for X_hat
        if iteration == 0:
            print(f"Gradient for X_hat: {X_hat.grad}")  # Should not be None
        
        # 12. Optimizer step
        optimizer.step()
        
        # 13. Optionally clamp the values to valid range based on feature constraints
        if min_feature_value is not None and max_feature_value is not None:
            with torch.no_grad():
                X_hat.clamp_(min_feature_value, max_feature_value)
        
        # 14. Record the loss
        loss_history.append(loss_mse.item())
        
        # 15. Print loss at intervals
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Loss MSE={loss_mse.item():.4f}")
    
    # Optional: Plot loss history
    # import matplotlib.pyplot as plt
    # plt.plot(loss_history)
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE Loss')
    # plt.title('Reconstruction Loss Over Iterations')
    # plt.show()
    
    return X_hat.detach()



"""def get_proto_batch(C_i_list, Y_hats):

    protos = []
    for i in range(Y_hats.size(0)):
        proto = C_i_list[Y_hats[i]]
        protos.append(proto)
    return torch.stack(protos)


def reconstruct_input_batch_y(args,projection_model, C_i_list,   
                      learning_rate=0.1, num_iterations=500, 
                      lambda_l2=1e-4, lambda_l1=1e-4,
                      min_feature_value=None, max_feature_value=None):

    # Initialize X_hat with requires_grad=True to optimize
    batch_size = args.local_bs
    input_shape = args.num_features
    X_hats = torch.randn((local_bs,input_shape), requires_grad=True, device=args.device)
    Y_hats = torch.randn((batch_size, args.num_classes), requires_grad=True, device=args.device)  # One-hot labels
    

    
    # Define optimizer
    optimizer = optim.Adam([X_hats, Y_hats], lr=learning_rate)
    
    # Define loss functions
    mse_loss = nn.MSELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    for iteration in tqdm(range(num_iterations), desc="Reconstructing"):
        C_i = get_proto_batch(C_i_list, Y_hats)
        optimizer.zero_grad()
        
        # Forward pass through the projection model
        # Assuming the forward method returns log_probs and prototypes
        _, protos = projection_model(X_hats)  # Add batch dimension if necessary
        # Inside reconstruct_input in inference.py, before line 51
 
        # Compute the MSE loss between projected X_hat and C_i
        loss_mse = mse_loss(protos, C_i)
        
        # Compute L2 and L1 regularization
        loss_l2 = l2_loss(X_hats, torch.zeros_like(X_hats))
        loss_l1 = l1_loss(X_hats, torch.zeros_like(X_hats))
        
        # Total loss
        loss = loss_mse #+ lambda_l2 * loss_l2 + lambda_l1 * loss_l1
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Optionally clamp the values to valid range based on feature constraints
        if min_feature_value is not None and max_feature_value is not None:
            with torch.no_grad():
                X_hat.clamp_(min_feature_value, max_feature_value)
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Loss MSE={loss_mse.item():.4f}, "
                  f"Loss L2={loss_l2.item():.4f}, Loss L1={loss_l1.item():.4f}, Total Loss={loss.item():.4f}")
    
    return X_hats.detach(), Y_hats.detach()



import random

def reconstruct_input_batch(args,projection_model, C_i,   
                      learning_rate=0.1, num_iterations=500, 
                      lambda_l2=1e-4, lambda_l1=1e-4,
                      min_feature_value=None, max_feature_value=None):
  
    # Initialize X_hat with requires_grad=True to optimize
    batch_size = args.local_bs
    input_shape = args.num_features
    X_hats = torch.randn((local_bs,input_shape), requires_grad=True, device=args.device)
    

    
    # Define optimizer
    optimizer = optim.Adam([X_hats], lr=learning_rate)
    
    # Define loss functions
    mse_loss = nn.MSELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    for iteration in tqdm(range(num_iterations), desc="Reconstructing"):
        C_i = torch.stack([C_i] * batch_size, dim=0) 
        optimizer.zero_grad()
        
        # Forward pass through the projection model
        # Assuming the forward method returns log_probs and prototypes
        _, protos = projection_model(X_hats)  # Add batch dimension if necessary
        # Inside reconstruct_input in inference.py, before line 51
 
        # Compute the MSE loss between projected X_hat and C_i
        loss_mse = mse_loss(protos, C_i)
        
        # Compute L2 and L1 regularization
        loss_l2 = l2_loss(X_hats, torch.zeros_like(X_hats))
        loss_l1 = l1_loss(X_hats, torch.zeros_like(X_hats))
        
        # Total loss
        loss = loss_mse #+ lambda_l2 * loss_l2 + lambda_l1 * loss_l1
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Optionally clamp the values to valid range based on feature constraints
        if min_feature_value is not None and max_feature_value is not None:
            with torch.no_grad():
                X_hat.clamp_(min_feature_value, max_feature_value)
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Loss MSE={loss_mse.item():.4f}, "
                  f"Loss L2={loss_l2.item():.4f}, Loss L1={loss_l1.item():.4f}, Total Loss={loss.item():.4f}")
    
    return X_hats.detach()


import random

def sample_original_data(train_dataset, sample_size, label=None):

    sampled_data = {}
    if label is not None:
        train_dataset =  [(images, labels) for images, labels in train_dataset if (labels == label).any().item()]
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]
    
    for idx in sampled_indices:
        features, label = train_dataset[idx]
        if label.item() not in sampled_data:
            sampled_data[label.item()] = []
        sampled_data[label.item()].append(features.numpy())
    
    # Compute mean per class for sampled data
    sampled_class_means = {}
    for class_label, features in sampled_data.items():
        sampled_class_means[class_label] = np.mean(features, axis=0)
    
    return sampled_class_means"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import torch
from sklearn.metrics import mean_squared_error

import torch

def evaluate_reconstruction(args, reconstructed_inputs, train_dataset, label,idxs):
    
    # Efficient filtering
    
    train_dataset =  DatasetSplit(train_dataset, idxs)
    train_dataset_filtered = [images for images, labels in train_dataset if labels.item() == label]
    
    # Stack images into a single tensor and compute the average
    if len(train_dataset_filtered) > 0:
        X_original = torch.stack(train_dataset_filtered).mean(dim=0)
    else:
        raise ValueError(f"No images found for label {label}")
    
    # Ensure reconstructed_inputs has the same shape as X_original
    if reconstructed_inputs.shape != X_original.shape:
        raise ValueError("Shape mismatch between reconstructed inputs and original data.")
    
    # Compute the squared L2 distance
    squared_l2_distance = torch.norm(reconstructed_inputs - X_original) ** 2
    mse = mean_squared_error(X_original, reconstructed_inputs)

    print(f"Class {label}: Squared L2 Distance = {squared_l2_distance.item():.4f}", f"MSE = {mse:.4f}")
    return mse


def evaluate_reconstruction__(args, reconstructed_inputs, train_dataset, label):
    """
    Evaluate the similarity between reconstructed prototypes and sampled original class means.
    
    Args:
        reconstructed_protos (dict): Reconstructed feature vectors per class.
        sampled_original_data (dict): Sampled original feature vectors per class.
    
    Returns:
        dict: Dictionary mapping class labels to similarity scores.
    """
    similarity_scores = {}
    mse_scores = {}
    train_dataset_filtered =  [(images, labels) for images, labels in train_dataset if (labels == label).any().item()]
    for (images, labels) in train_dataset_filtered:
        X_original = images
        # Ensure both tensors are 1-D before computing similarity
        """if reconstructed_inputs.dim() > 1:
            reconstructed_inputs = reconstructed_inputs.view(-1)
        if X_original.dim() > 1:
            X_original = X_original.view(-1)"""
        # Compute Cosine Similarity
        #print("reconstructed_inputs shape", reconstructed_inputs.shape)
        #print("X_original shape", X_original.shape)
        cos_sim = cosine_similarity([reconstructed_inputs], [X_original])[0][0]
        similarity_scores[label] = cos_sim

        # Compute Mean Squared Error
        mse = mean_squared_error(X_original, reconstructed_inputs)
        mse_scores[label] = mse
    similarity_score = np.mean(list(similarity_scores.values()))
    mse_score = np.mean(list(mse_scores.values()))
    print(f"Class {label}: Cosine Similarity = {similarity_score:.4f}, MSE = {mse_score:.4f}")
    return similarity_scores,  mse_scores
    

def evaluate_reconstruction_(reconstructed_inputs, sampled_original_data):
    """
    Evaluate the similarity between reconstructed prototypes and sampled original class means.
    
    Args:
        reconstructed_protos (dict): Reconstructed feature vectors per class.
        sampled_original_data (dict): Sampled original feature vectors per class.
    
    Returns:
        dict: Dictionary mapping class labels to similarity scores.
    """
    similarity_scores = {}
    mse_scores = {}
    print(reconstructed_inputs)
    
    #for class_label, X_reconstructed in reconstructed_inputs.items():
    X_original = sampled_original_data[class_label]
    # Compute Cosine Similarity
    cos_sim = cosine_similarity([X_reconstructed], [X_original])[0][0]
    similarity_scores[class_label] = cos_sim
    
    # Compute Mean Squared Error
    mse = mean_squared_error(X_original, X_reconstructed)
    mse_scores[class_label] = mse
    
    print(f"Class {class_label}: Cosine Similarity = {cos_sim:.4f}, MSE = {mse:.4f}")
    
    
    
    return {"Cosine Similarity": similarity_scores, "MSE": mse_scores}

def get_feature_range(train_dataset, idxs):
    """
    Compute the min and max for each feature across the specified subset of the dataset.
    
    Args:
        train_dataset (Dataset): The entire training dataset.
        idxs (list or Tensor): Indices of the subset to consider.
        
    Returns:
        min_vals (Tensor): Minimum values per feature.
        max_vals (Tensor): Maximum values per feature.
    """
    subset = DatasetSplit(train_dataset, idxs)
    data = torch.stack([data for data, label in subset])
    min_vals = data.min(dim=0).values
    max_vals = data.max(dim=0).values
    return min_vals, max_vals



def generate_random_guess(min_vals, max_vals, device):
    """
    Generate a random guess within the feature range using numpy's uniform distribution.
    
    Args:
        min_vals (Tensor): Minimum values per feature.
        max_vals (Tensor): Maximum values per feature.
        device (torch.device): Device to place the tensor on.
        
    Returns:
        X_random (Tensor): Randomly generated input tensor.
    """
    # Ensure min_vals and max_vals are on CPU and convert to NumPy
    min_vals_np = min_vals.detach().cpu().numpy()
    max_vals_np = max_vals.detach().cpu().numpy()
    
    # Generate a single random sample using numpy.uniform for each feature
    # np.random.uniform can accept arrays for low and high to generate element-wise
    random_np = np.random.uniform(low=min_vals_np, high=max_vals_np)
    
    # Convert the NumPy array back to a PyTorch tensor
    X_random = torch.from_numpy(random_np).float().to(device)
    
    return X_random

def compute_baseline_mse(args, train_dataset, label, idxs):
    """
    Compute the baseline MSE using a random guess.
    
    Args:
        args: Arguments containing device information.
        train_dataset (Dataset): The entire training dataset.
        label (int): The class label to consider.
        idxs (list or Tensor): Indices of the subset to consider.
        
    Returns:
        baseline_mse (float): The MSE of the random guess.
    """
    train_subset = DatasetSplit(train_dataset, idxs)
    train_filtered = [data for data, lbl in train_subset if lbl.item() == label]
    
    if not train_filtered:
        print(f"No images found for label {label}")
        return None
    
    X_original = torch.stack(train_filtered).mean(dim=0)
    
    min_vals, max_vals = get_feature_range(train_dataset, idxs)
    X_random = generate_random_guess(min_vals, max_vals, args.device)
    
    mse = F.mse_loss(X_random, X_original).item()
    print(f"Baseline MSE (Random Guess) for Class {label}: {mse:.4f}")
    return mse

