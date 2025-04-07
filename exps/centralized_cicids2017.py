import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import gc  # Garbage collector


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Function to read data in chunks
def load_data_in_chunks(file_list, chunk_size):
    for file in file_list:
        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
            yield chunk

# Update this path to where your CICIDS2017 CSV files are located
path = '../dataset/cicids2017/*.csv'
file_list = glob.glob(path)

print(f'Found {len(file_list)} CSV files.')

if len(file_list) == 0:
    raise ValueError("No CSV files found. Please check the dataset path.")

# Define chunk size based on your system's capacity
chunk_size = 1000000

# **First Pass: Collect All Unique Labels**
print('Collecting all unique labels...')

unique_labels_set = set()

for file in file_list:
    print(f'Processing file: {file} for unique labels')
    for chunk in load_data_in_chunks([file], chunk_size):
        # Ensure 'Label' column exists
        if 'Label' not in chunk.columns:
            raise ValueError(f"'Label' column not found in {file}")
        
        # Extract unique labels
        unique_labels = chunk['Label'].astype(str).unique()
        unique_labels_set.update(unique_labels)
        
        # Optional: Clean up
        del chunk
        gc.collect()

# Convert the set to a sorted list
all_unique_labels = sorted(list(unique_labels_set))
print(f'All unique labels ({len(all_unique_labels)}): {all_unique_labels}')

# **Initialize and Fit LabelEncoder on All Unique Labels**
le_label = LabelEncoder()
le_label.fit(all_unique_labels)
print(f'Classes in label encoder: {le_label.classes_}')
print(f'Number of classes: {len(le_label.classes_)}')

# **Second Pass: Process Data and Encode Labels**
print('Processing data and encoding labels...')

# Initialize empty lists for features and labels
X_list = []
y_list = []

for file in file_list:
    print(f'Processing file: {file} for data and labels')
    for chunk in load_data_in_chunks([file], chunk_size):
        # Data Cleaning Steps
        # Drop columns with all NaNs
        chunk.dropna(axis=1, how='all', inplace=True)

        # Replace inf values with NaN and drop rows with NaN
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        # Remove irrelevant features
        irrelevant_features = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
        chunk.drop(irrelevant_features, axis=1, inplace=True, errors='ignore')

        # Encode categorical variables (excluding 'Label')
        categorical_cols = chunk.select_dtypes(include=['object']).columns.tolist()
        if 'Label' in categorical_cols:
            categorical_cols.remove('Label')

        # Initialize LabelEncoder for categorical features
        le_categorical = LabelEncoder()

        for col in categorical_cols:
            # Fit and transform each categorical column
            chunk[col] = le_categorical.fit_transform(chunk[col].astype(str))

        # Encode the 'Label' column using the fitted le_label
        chunk['Label'] = le_label.transform(chunk['Label'].astype(str))

        # Separate features and labels
        X_chunk = chunk.drop('Label', axis=1)
        y_chunk = chunk['Label']

        # Append to lists
        X_list.append(X_chunk)
        y_list.append(y_chunk)

        # Optional: Clean up
        del chunk, X_chunk, y_chunk
        gc.collect()

        # Optional: Break after processing a certain number of chunks to limit data size
        # if len(X_list) * chunk_size >= desired_size:
        #     break

# Concatenate all chunks
print('Concatenating all processed chunks...')
X = pd.concat(X_list, ignore_index=True)
y = pd.concat(y_list, ignore_index=True)
print(f'Dataset shape after loading and preprocessing: {X.shape}')

# Optional: Clean up lists to free memory
del X_list, y_list
gc.collect()

# Optimize data types
print('Optimizing data types...')
for col in X.select_dtypes(include=['float64']).columns:
    X[col] = pd.to_numeric(X[col], downcast='float')
for col in X.select_dtypes(include=['int64']).columns:
    X[col] = pd.to_numeric(X[col], downcast='integer')
print('Data types optimized.')

# Feature Scaling
print('Performing feature scaling...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Feature scaling completed.')

# Use a smaller subset for initial testing (e.g., 20% of the data)
print('Sampling a subset of the data for initial testing...')
X_sampled, _, y_sampled, _ = train_test_split(
    X_scaled, y, test_size=0.8, random_state=42, stratify=y
)
print(f'Sampled dataset shape: {X_sampled.shape}')
print(f'Class distribution after sampling: {Counter(y_sampled)}')

# Proceed with data splitting
print('Splitting data into training and testing sets...')
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled
    )
    print(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')
except ValueError as e:
    print(f'Error during train_test_split: {e}')
    print('Attempting to split without stratification...')
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42
    )
    print(f'Training set size: {X_train.shape}, Test set size: {X_test.shape}')
    print('Proceeding without stratification. Be cautious of class imbalance.')

# Convert data to PyTorch tensors
print('Converting data to PyTorch tensors...')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
print('Data conversion completed.')

# Create custom Dataset class
class CICIDS2017Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create DataLoaders
batch_size = 256  # Reduced batch size to lower memory usage
print('Creating DataLoaders...')
train_dataset = CICIDS2017Dataset(X_train_tensor, y_train_tensor)
test_dataset = CICIDS2017Dataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f'Number of training batches: {len(train_loader)}, Number of testing batches: {len(test_loader)}')

# Define the Improved Model
class ImprovedDenseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedDenseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, output_size)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
print(f'Number of features in X_train: {X_train.shape[1]}')
print(f'Number of features in X_test: {X_test.shape[1]}')

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
output_size = len(le_label.classes_)  # Correctly set to number of classes
model = ImprovedDenseModel(input_size, output_size).to(device)
print(f'Model initialized with input size {input_size} and output size {output_size}.')

# Define loss function with class weights to handle imbalance
print('Calculating class weights for loss function...')
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print('Loss function with class weights defined.')

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print('Optimizer initialized.')

# Training Loop
num_epochs = 10
print(f'Starting training for {num_epochs} epochs...')
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Statistics
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    epoch_acc = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_acc:.2f}%')

# Evaluation on Test Set
print('Evaluating model on test set...')
model.eval()
test_loss = 0
correct = 0
total = 0
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
test_acc = 100. * correct / total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%')

# Classification Metrics
print("\nClassification Report:")
try:
    print(classification_report(all_targets, all_preds, target_names=le_label.classes_))
except ValueError as e:
    print(f'Error in classification_report: {e}')
    print('Attempting to specify labels manually...')
    labels = np.unique(all_targets)
    print(classification_report(all_targets, all_preds, labels=labels, target_names=le_label.classes_[:len(labels)]))

# Confusion Matrix
print('Confusion Matrix:')
cm = confusion_matrix(all_targets, all_preds)
print(cm)

# Calculate Precision, Recall, and F1 Score
print("\nCalculating Precision, Recall, and F1 Score...")
precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Calculate Macro Accuracy


# Initialize list to store per-class accuracy
per_class_accuracy = []

# Total number of samples
total_samples = cm.sum()

# Number of classes
num_classes = cm.shape[0]

for i in range(num_classes):
    # True Positives for class i
    TP = cm[i, i]
    
    # False Positives for class i (sum of column i, excluding TP)
    FP = cm[:, i].sum() - TP
    
    # False Negatives for class i (sum of row i, excluding TP)
    FN = cm[i, :].sum() - TP
    
    # True Negatives for class i (total samples - TP - FP - FN)
    TN = total_samples - (TP + FP + FN)
    
    # Per-class accuracy
    accuracy = (TP + TN) / total_samples
    per_class_accuracy.append(accuracy)

# Calculate Macro Accuracy
macro_accuracy = np.mean(per_class_accuracy)
print(f'Macro Accuracy: {macro_accuracy:.4f}')
