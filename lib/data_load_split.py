import os
from functools import partial
import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from scipy.stats import dirichlet
from flwr.server.strategy.krum import Krum
from flwr.common import ndarrays_to_parameters
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import classification_report, confusion_matrix 

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.utils import resample
from collections import Counter
import gc  # Garbage collector
import glob


from poisoning import label_flipping_untargeted, flip_labels

class DataFrameDataset(Dataset):
    def __init__(self, features, targets):
        # Check if features and targets are pandas DataFrame/Series or numpy array and handle accordingly
        if isinstance(features, (pd.DataFrame, pd.Series)):
            self.features = features.values
        else:
            self.features = features
        
        if isinstance(targets, (pd.DataFrame, pd.Series)):
            self.targets = targets.values
            self.labels = targets.values
        else:
            self.targets = targets
            self.labels = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, target

"""Benign
0
SYNScan
3
TCPConnectScan
5
UDPScan
6
SYNFlood
2
HTTPFlood
1
SlowrateDoS
4
"""

def sample_within_range(group, min_samples, max_samples):
    # Determine the number of samples for the class
    n_samples = min(max(len(group), min_samples), max_samples)
    
    # Resample (if the group is larger than n_samples, it will undersample; otherwise, it oversamples)
    return resample(group, replace=len(group) < n_samples, n_samples=n_samples, random_state=42)

def sample_with_max(group, max_samples):
    return resample(group, replace=False, n_samples=min(len(group), max_samples), random_state=42)

def load_data_cicids2017(args):

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
    print('Converting data to PyTorch tensors...')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)
            
    # Create Dataset objects for training and testing data
    train_dataset = DataFrameDataset(X_train, Y_train)
    test_dataset = DataFrameDataset(X_test, Y_test)
    print("length of test dataset", test_dataset.__len__())
    return train_dataset, test_dataset


def load_data_cicids2017_old(args):
    folder = '../dataset/cicids2017'
    data = pd.DataFrame() # Initialize an empty DataFrame to store the dataset
    # Load the dataset
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            path = os.path.join(folder, file)
            print(path)
            data_file = pd.read_csv(path)
            data = pd.concat([data, data_file])
            
    
    # Drop unnecessary columns
    drop_columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
    data = data.drop(drop_columns, axis=1)
    #print number of na values
    print("Number of na values",data.isna().sum().sum())
    # Drop missing values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    # Drop duplicates
    data = data.drop_duplicates()
    # Separate features (X) and labels (Y)
    data = data.sample(frac=args.data_percent, random_state=42)
    class_counts = data['Label'].value_counts()

    # Print the result
    print("class count:", class_counts)
    min_samples = 100
    max_samples = 10000
    valid_classes = class_counts[class_counts >= min_samples].index
    df_filtered = data[data['Label'].isin(valid_classes)]
    df_balanced = df_filtered.groupby('Label', group_keys=False).apply(lambda x: sample_with_max(x, max_samples))

    #df_balanced = data.groupby('Label', group_keys=False).apply(lambda x: sample_within_range(x, min_samples, max_samples))
    #data = df_balanced
    class_counts = data['Label'].value_counts()

    # Print the result
    print("class count:", class_counts)
    print("Number of na values",data.isna().sum().sum())
    X = data.drop(['Label'], axis=1)
    #print number of columns
    print("Number of columns",len(data.columns))
    args.num_features = len(data.columns) - 1
    Y = data['Label']
    print(Y)
    #convert label to numerical values
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    #print number of unique labels
    print("labels", np.unique(Y))
    print("Number of unique labels", len(np.unique(Y)))
    args.num_classes = len(np.unique(Y))
    # Split the dataset 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
    print("Number of na values",X_train.isna().sum().sum())
    print("Number of infinite values in X_train:", np.isinf(X_train).sum().sum())

    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)

    



    # Create Dataset objects for training and testing data
    train_dataset = DataFrameDataset(X_train, Y_train)
    test_dataset = DataFrameDataset(X_test, Y_test)
    print("length of test dataset", test_dataset.__len__())
    return train_dataset, test_dataset


def load_data_5g_nidd(args):
    # Load the dataset
    path = '../dataset/5g-nidd/Combined.csv'
    data = pd.read_csv(path)
    # Drop unnecessary columns
    drop_columns = ['Unnamed: 0', 'RunTime', 'Min', 'Max', 'sTos', 'dTos', 'sDSb', 'dDSb', 'sHops', 'dHops', 'SrcWin', 'DstWin', 'sVid', 'dVid', 'SrcTCPBase', 'DstTCPBase', 'TcpRtt', 'SynAck', 'AckDat']
    data = data.drop(drop_columns, axis=1)
    # Drop missing values
    data = data.dropna()
    # Drop duplicates
    data = data.drop_duplicates()
    # Separate features (X) and labels (Y)
    X = data.drop(['Label', 'Attack Type', 'Attack Tool'], axis=1)
    Y = data['Attack Type']
    print(data.columns)
    # Convert categorical columns to numerical using one-hot encoding
    X = pd.get_dummies(X, columns=['Proto', 'Cause', 'State'])
    # Add columns 'State_FIN' and 'State_RST' if not existing and fill them with 0
    if 'State_FIN' not in X.columns:
        X['State_FIN'] = 0
    if 'State_RST' not in X.columns:
        X['State_RST'] = 0
    # Scale numerical features
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    """print ("----------************************----------------------***********************__________________________**********************------------")
    attack_cat_index = {}
    for i, attack in enumerate(y_save.unique()):
        print(attack)
        attack_cat_id = label_encoder.transform([attack])[0]
        print(attack_cat_id)"""
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(len(Y_test))
    if args.semi > 0:
        unlabeled_indices = np.random.choice(len(Y_train), size=int(len(Y_train) * args.semi), replace=False)
        Y_train[unlabeled_indices] = -1
    #return X_train, X_test, Y_train, Y_test, label_encoder
    train_dataset = DataFrameDataset(X_train, Y_train)
    test_dataset = DataFrameDataset(X_test, Y_test)
    print("length of test dataset", test_dataset.__len__())
    return train_dataset, test_dataset


def load_data_x_iiotid(args):   
    dtype_mapping = {
    
        'Protocol': 'object',
        'Service': 'object',
        'Duration': 'float64',
        'Scr_bytes': 'float64',
        'Des_bytes': 'float64',
        'Conn_state': 'int64',
        'missed_bytes': 'int64',
        'is_syn_only': 'bool',
        'Is_SYN_ACK': 'bool',
        'is_pure_ack': 'bool',
        'is_with_payload': 'bool',
        'FIN or RST': 'bool',
        'Bad_checksum': 'bool',
        'is_SYN_with_RST': 'bool',
        'Scr_pkts': 'int64',
        'Scr_ip_bytes': 'int64',
        'Des_pkts': 'int64',
        'Des_ip_bytes': 'int64',
        'total_bytes': 'float64',
        'total_packet': 'int64',
        'paket_rate': 'float64',
        'byte_rate': 'float64',
        'Scr_packts_ratio': 'float64',
        'Des_pkts_ratio': 'float64',
        'Scr_bytes_ratio': 'float64',
        'Des_bytes_ratio': 'float64',
        'Avg_user_time': 'float64',
        'Std_user_time': 'float64',
        'Avg_nice_time': 'float64',
        'Std_nice_time': 'float64',
        'Avg_system_time': 'float64',
        'Std_system_time': 'float64',
        'Avg_iowait_time': 'float64',
        'Std_iowait_time': 'float64',
        'Avg_ideal_time': 'float64',
        'Std_ideal_time': 'float64',
        'Avg_tps': 'float64',
        'Std_tps': 'float64',
        'Avg_rtps': 'float64',
        'Std_rtps': 'float64',
        'Avg_wtps': 'float64',
        'Std_wtps': 'float64',
        'Avg_ldavg_1': 'float64',
        'Std_ldavg_1': 'float64',
        'Avg_kbmemused': 'float64',
        'Std_kbmemused': 'float64',
        'Avg_num_Proc/s': 'float64',
        'Std_num_proc/s': 'float64',
        'Avg_num_cswch/s': 'float64',
        'std_num_cswch/s': 'float64',
        'Login_attempt': 'int64',
        'Succesful_login': 'int64',
        'File_activity': 'int64',
        'Process_activity': 'int64',
        'read_write_physical.process': 'int64',
        'is_privileged': 'bool',
        'class1': 'object',
        'class2': 'object',
        'class3': 'object'
    }

    # Load the dataset 
    na_values = ['-','?','aza','#DIV/0!']
    data = pd.read_csv('../dataset/x-iiotid/X-IIoTID dataset.csv', na_values=na_values) 
    """mask = data.replace(['-', '?', 'aza', '#DIV/0!'], pd.NA).isna().any(axis=1)
    na_rows = data[mask]

    # Save the filtered rows to a CSV file
    na_rows.to_csv('filtered_na_rows.csv', index=False)  # Set index=False if you do not want to save row indices in the file."""

    # Drop unnecessary columns 
    print(data.columns)
    df_head = data.head(10)
    df_head.to_csv('first_n_lines.csv', index=False)
    drop_columns = ['Date', 'Timestamp', 'Scr_IP', 'Scr_port', 'Des_IP', 'Des_port', 'anomaly_alert','OSSEC_alert', 'OSSEC_alert_level'] 
    data = data.drop(drop_columns, axis=1) 
    # Drop missing values 
    data = data.dropna() 

    # Replace special symbols like '?' with NaN
    data.replace('?', pd.NA, inplace=True)

    # Drop rows with NaN values
    data.dropna(inplace=True)
    # Drop duplicates 
    data = data.drop_duplicates() 
    """# Separate features (X) and labels (Y) 
    X = data.drop(['class1', 'class2', 'class3'], axis=1) 
    Y = data['class2'] 
    # Convert categorical columns to numerical using one-hot encoding 
    X = pd.get_dummies(X, columns=['Protocol', 'Service', 'is_syn_only', 'Is_SYN_ACK',
        'is_pure_ack', 'is_with_payload', 'FIN or RST', 'Bad_checksum',
        'is_SYN_with_RST']) """
    # Display information about missing values
    print("Missing values before imputation:\n", data.isnull().sum())
    # Convert columns to the correct dtype according to dtype_mapping
    for col, dtype in dtype_mapping.items():
        data[col] = data[col].astype(dtype)
    # Separating numerical and categorical columns
    # Identify columns by data type after explicit conversion
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    bool_cols = data.select_dtypes(include=['bool']).columns
    print("Numerical columns:\n", num_cols)
    print("Categorical columns:\n", cat_cols)
    print("Boolean columns:\n", bool_cols)
    """# Impute numerical columns with mean
    num_imputer = SimpleImputer(strategy='mean')0
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

    # Impute categorical columns with most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
    # Handle missing values for boolean columns"""

    # Convert boolean columns to 0 and 1
    data[bool_cols] = data[bool_cols].astype(int)
    """bool_imputer = SimpleImputer(strategy='most_frequent')  # or strategy='constant', fill_value=False
    data[bool_cols] = bool_imputer.fit_transform(data[bool_cols])
    print("Missing values after imputation:\n", data.isnull().sum().sum())"""
    data = data[data['class2'] != 'crypto-ransomware']

    #take only a percentage of the data
    data = data.sample(frac=args.data_percent, random_state=42)

    # Separate features (X) and labels (Y)
    X = data.drop(['class1', 'class2', 'class3'], axis=1) 
    print("(,(,(,(,,(number of columns",len(X.columns))
    Y = data['class2'] 
    print(len(Y))
    print("---------------------------------------------------------")
    print(Y.unique())
    
    # Convert categorical columns to numerical using one-hot encoding 
    #X = pd.get_dummies(X, columns=cat_cols)
    # Columns to exclude from categorical columns
    exclude_cols = ['class1', 'class2', 'class3']
    


    # Remove the excluded columns from the categorical columns list
    cat_cols = cat_cols.difference(exclude_cols)
    print("Categorical columns:\n", cat_cols)
    X = pd.get_dummies(X, columns=cat_cols)
    # Scale numerical features 
    scaler = StandardScaler() 
    X[X.columns] = scaler.fit_transform(X) 
    # Convert labels to numerical values 
    label_encoder = LabelEncoder() 
    Y = label_encoder.fit_transform(Y) 
    """if args.attack_type == 'label-flipping':
        Y = flip_labels(args, Y, args.flip_ratio)"""
        #Y = label_flipping_untargeted(Y, args.flip_ratio)
    # Split the dataset 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
    """if args.semi > 0:
        unlabeled_indices = np.random.choice(len(Y_train), size=int(len(Y_train) * args.semi), replace=False)
        Y_train[unlabeled_indices] = -1"""
    # Create Dataset objects for training and testing data
    print("(,(,(,(,,(number of columns",len(X_train.columns))
    train_dataset = DataFrameDataset(X_train, Y_train)
    test_dataset = DataFrameDataset(X_test, Y_test)
    print("---------------------------------------------------------")
    print(np.unique(Y_train))
    print("length of test dataset", test_dataset.__len__())
    return train_dataset, test_dataset


def change_label(df):
    df.label.replace(['DDoS-ICMP_Flood','DDoS-UDP_Flood','DDoS-TCP_Flood','DDoS-PSHACK_Flood','DDoS-SYN_Flood','DDoS-RSTFINFlood','DDoS-SynonymousIP_Flood','DDoS-ICMP_Fragmentation','DDoS-UDP_Fragmentation','DDoS-ACK_Fragmentation','DDoS-HTTP_Flood','DDoS-SlowLoris'],'DDos',inplace=True)
    df.label.replace(['DoS-UDP_Flood','DoS-TCP_Flood','DoS-SYN_Flood','DoS-HTTP_Flood'],'DoS',inplace=True)      
    df.label.replace(['Recon-HostDiscovery','Recon-OSScan','Recon-PortScan','Recon-PingSweep','VulnerabilityScan'],'Recon',inplace=True)
    df.label.replace(['MITM-ArpSpoofing','DNS_Spoofing'],'Spoofing',inplace=True)
    df.label.replace(['DictionaryBruteForce'],'BruteForce',inplace=True)
    df.label.replace(['BrowserHijacking','XSS','Uploading_Attack','SqlInjection','CommandInjection','Backdoor_Malware'],'Web-based',inplace=True)
    df.label.replace(['Mirai-greeth_flood','Mirai-udpplain','Mirai-greip_flood'],'Mirai',inplace=True)
    df.label.replace(['BenignTraffic'],'BENIGN',inplace=True)

def scaleStandardData(dataFrame, numeric_cols):
  scaler = preprocessing.StandardScaler()
  for col in numeric_cols:
    arr = dataFrame[col]
    arr = np.array(arr)
    dataFrame[col] = scaler.fit_transform(arr.reshape(len(arr),1))
  return dataFrame

def scaleMinMaxData(dataFrame, numeric_cols):
  scaler = preprocessing.MinMaxScaler()
  for col in numeric_cols:
    arr = dataFrame[col]
    arr = np.array(arr)
    dataFrame[col] = scaler.fit_transform(arr.reshape(len(arr),1))
  return dataFrame

def scaleData(dataFrame, numeric_cols):
  dataFrame = scaleStandardData(dataFrame, numeric_cols)
  dataFrame = scaleMinMaxData(dataFrame, numeric_cols)
  return dataFrame

def load_data_cic_iot(path='../dataset/cic-iot-2023/'):

    dfs = []
    dfs_test = []
    total_num = 10

    for i in range(0,total_num):
        filename = f"../dataset/cic-iot-2023/part-0000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
        if i >=10 :
            filename = f"../dataset/cic-iot-2023/part-000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
        #df_test = pd.read_csv("../dataset/cic-iot-2023/part-00005-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
        if (i< total_num*0.8):
            df = pd.read_csv(filename)
            dfs.append(df)
        else:
            df_test = pd.read_csv(filename)
            dfs_test.append(df_test)

    df = pd.concat(dfs)
    df_test = pd.concat(dfs_test)

    labels_to_remove = ['DictionaryBruteForce', 'BrowserHijacking', 'XSS', 'Uploading_Attack', 'SqlInjection', 'CommandInjection', 'Backdoor_Malware']
    change_label(df)
    change_label(df_test)

    x_label_distribute = np.array(
    df["label"].value_counts().index.tolist())
    y_label_distribute = np.array(
        df["label"].value_counts().values.tolist())

    # create dataframe labels (Dos,Probe,R2L,U2R,normal)
    label = pd.DataFrame(df.label)

    Y_TRAIN = df['label']
    Y_TEST = df_test['label']
    X_TRAIN = df.drop('label', axis=1).copy()
    X_TEST = df_test.drop('label', axis=1).copy()

    print(X_TRAIN.shape)
    print(X_TEST.shape)
    print(Y_TRAIN.shape)
    print(Y_TEST.shape)

    # Convert all float64 columns to float128
    float_cols = X_TRAIN.select_dtypes(include=['float64']).columns
    X_TRAIN[float_cols] = X_TRAIN[float_cols].astype(np.float128)
    float_cols = X_TEST.select_dtypes(include=['float64']).columns
    X_TEST[float_cols] = X_TEST[float_cols].astype(np.float128)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y_TRAIN)
    Y_TRAIN = Y_TRAIN.replace({'BENIGN': 0, 'DDos': 1,'DoS':2,'Mirai':3,'Spoofing':4,'Recon':5,'Web-based':6,'BruteForce':7})
    Y_TEST = Y_TEST.replace({'BENIGN': 0, 'DDos': 1,'DoS':2,'Mirai':3,'Spoofing':4,'Recon':5,'Web-based':6,'BruteForce':7})
    print("Y_TRAIN",Y_TRAIN.shape)
    print("Y_TEST",Y_TEST.shape)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    numeric_features = X_TRAIN.select_dtypes(include='number').columns
    # Scale for common data
    X_TRAIN = scaleData(X_TRAIN, numeric_features)


    # List of selected important features
    selected_features = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'urg_count', 'rst_count', 
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
        'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
        'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
        'Radius', 'Covariance', 'Variance', 'Weight', 
    ]#['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'syn_flag_number', 'psh_flag_number', 'ack_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'UDP', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']

    # Select important features from the training set
    X_rfeTrain = X_TRAIN.loc[:, selected_features]
    X_TEST = X_TEST.loc[:, selected_features]

    X_train, X_test, Y_train, Y_test = train_test_split(X_rfeTrain, Y_TRAIN, test_size=0.2, random_state=42)

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if args.semi > 0:
        unlabeled_indices = np.random.choice(len(Y_train), size=int(len(Y_train) * args.semi), replace=False)
        Y_train[unlabeled_indices] = -1
    #X_train, X_test, Y_train, Y_test = train_test_split(X_rfeTrain, Y_TRAIN, test_size=0.2, random_state=42)
    #return X_train, X_test, Y_train, Y_test, label_encoder
    train_dataset = DataFrameDataset(X_train, Y_train)
    test_dataset = DataFrameDataset(X_test, Y_test)
    print("length of test dataset", test_dataset.__len__())
    return train_dataset, test_dataset



def load_data_cic_iot_old(args):

    dfs = []
    for i in range(0,3):
        filename = f"../dataset/cic-iot-2023/part-0000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
        if i >=10 :
            filename = f"../dataset/cic-iot-2023/part-000{i}-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
        df_test = pd.read_csv("../dataset/cic-iot-2023/part-00005-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
        df = pd.read_csv(filename)
        dfs.append(df)

    labels_to_remove = ['DictionaryBruteForce', 'BrowserHijacking', 'XSS', 'Uploading_Attack', 'SqlInjection', 'CommandInjection', 'Backdoor_Malware']
    # class classification

    change_label(df)
    change_label(df_test)

    # change_label(test_data)

    #df_DDOS = df[df['label'].isin(['DDos', 'BENIGN'])]
    #df_DDOS_test = df_test[df_test['label'].isin(['DDos', 'BENIGN'])]

    x_label_distribute = np.array(
    df["label"].value_counts().index.tolist())
    y_label_distribute = np.array(
        df["label"].value_counts().values.tolist())
    
    # create dataframe labels (Dos,Probe,R2L,U2R,normal)
    label = pd.DataFrame(df.label)

    Y_TRAIN = df['label']
    Y_TEST = df_test['label']
    X_TRAIN = df.drop('label', axis=1).copy()
    X_TEST = df_test.drop('label', axis=1).copy()

    #import numpy as np

    # Convert all float64 columns to float128
    float_cols = X_TRAIN.select_dtypes(include=['float64']).columns
    X_TRAIN[float_cols] = X_TRAIN[float_cols].astype(np.float128)
    float_cols = X_TEST.select_dtypes(include=['float64']).columns
    X_TEST[float_cols] = X_TEST[float_cols].astype(np.float128)

    Y_TRAIN = Y_TRAIN.replace({'BENIGN': 0, 'DDos': 1,'DoS':2,'Mirai':3,'Spoofing':4,'Recon':5,'Web-based':6,'BruteForce':7})
    Y_TEST = Y_TEST.replace({'BENIGN': 0, 'DDos': 1,'DoS':2,'Mirai':3,'Spoofing':4,'Recon':5,'Web-based':6,'BruteForce':7})

    numeric_features = X_TRAIN.select_dtypes(include='number').columns
    # Scale for common data
    X_TRAIN = scaleData(X_TRAIN, numeric_features)
    clf = DecisionTreeClassifier(random_state=0)
    rfe = RFE(estimator=clf, n_features_to_select=30, step=1)
    rfe.fit(X_TRAIN, Y_TRAIN.astype(int))
    X_rfeTrain=rfe.transform(X_TRAIN)
    true=rfe.support_
    rfecolindex_train=[i for i, x in enumerate(true) if x]
    rfecolname_train=list(numeric_features[i] for i in rfecolindex_train)

    rfe.fit(X_TRAIN, Y_TRAIN.astype(int))
    X_rfeTrain=rfe.transform(X_TRAIN)
    true=rfe.support_
    rfecolindex_train=[i for i, x in enumerate(true) if x]
    rfecolname_train=list(numeric_features[i] for i in rfecolindex_train)

    print('Features selected for Train:',rfecolname_train)

    # List of selected important features
    selected_features = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'syn_flag_number', 'psh_flag_number', 'ack_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'UDP', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']

    # Select important features from the training set
    X_rfeTrain = X_TRAIN.loc[:, selected_features]
    X_TEST = X_TEST.loc[:, selected_features]

    #X_train, X_test, Y_train, Y_test = train_test_split(X_rfeTrain, Y_TRAIN, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test = X_rfeTrain, X_TEST, Y_TRAIN, Y_TEST
    # Data set size
    sizes = [X_train.shape[0], X_test.shape[0], Y_train.shape[0], Y_test.shape[0]]
    labels = ['X_train', 'X_test', 'Y_train', 'Y_test']


    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)


    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("-----------------load data-----------------  ")
    print("X_train shape: ", X_train_scaled.shape)
    print("Y_train shape: ", Y_TRAIN.shape)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y_TRAIN)
    return X_train, X_test, Y_train, Y_test, label_encoder

def split_data_random(x_train, y_train, x_test,y_test, num_clients, train_split=0.8):
    partition_size = x_train.shape[0] // num_clients
    all_indices = np.arange(x_train.shape[0])
    print(x_train.shape)
    print(y_train.shape)
    #client_id_to_indices = {}
    x_trains = {}
    y_trains = {}
    x_tests = {}
    y_tests = {}
    for i in range(num_clients):
        client_train_indices = np.random.choice(all_indices, int(partition_size*train_split), replace=False)
        #client_id_to_indices[i] = client_indices
        print(client_train_indices)
        x_trains[str(i)] = pd.DataFrame(x_train).iloc[client_train_indices]
        print(y_train.shape)
                
        if isinstance(y_train, pd.Series):
            y_trains[str(i)] = y_train.iloc[client_train_indices].values
        elif isinstance(y_train, np.ndarray):
            y_trains[str(i)] = y_train[client_train_indices]
        else:
            raise TypeError("Unsupported data type for y_train")
        all_indices = np.setdiff1d(all_indices, client_train_indices)
        client_test_indices = np.random.choice(all_indices, int(partition_size*(1-train_split)), replace=False)
        x_tests[str(i)] = pd.DataFrame(x_train).iloc[client_test_indices]
        if isinstance(y_train, pd.Series):
            y_tests[str(i)] = y_train.iloc[client_test_indices].values
        elif isinstance(y_train, np.ndarray):
            y_tests[str(i)] = y_train[client_test_indices]
        else:
            raise TypeError("Unsupported data type for y_train")
        all_indices = np.setdiff1d(all_indices, client_test_indices)
    return x_trains, y_trains, x_tests, y_tests

def split_data_noniid(x_train, y_train, x_test, y_test, num_clients,train_split=0.8):
    partition_size = x_train.shape[0] // num_clients
    all_indices = np.arange(x_train.shape[0])
    print(x_train.shape)
    #client_id_to_indices = {}
    indices = {}
    x_trains = {}
    y_trains = {}
    x_tests = {}
    y_tests = {}
    num_shards, num_instances = 200, y_train.shape[0] // 200    
    all_indices = np.arange(num_shards * num_instances)
    idx_shard = np.arange(num_shards)
    #dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_instances)
    labels = y_train[:num_shards * num_instances]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    num_shards_by_client = num_shards // num_clients
    # divide and assign 2 shards/client
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, num_shards_by_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            indices[i] = idxs[rand * num_instances:(rand + 1) * num_instances]
    for i in range(num_clients):
        partition_size = len(indices[i])
        client_train_indices = np.random.choice(indices[i], int(partition_size*train_split), replace=False)
        #client_id_to_indices[i] = client_indices
        x_trains[str(i)] = pd.DataFrame(x_train).iloc[client_train_indices]
        y_trains[str(i)] = y_train[client_train_indices]
        client_test_indices = np.setdiff1d(indices[i], client_train_indices)
        x_tests[str(i)] = pd.DataFrame(x_train).iloc[client_test_indices]
        y_tests[str(i)] = y_train[client_test_indices]
    return x_trains, y_trains, x_tests, y_tests

def split_data_noniid_unequal(x_train, y_train, x_test, y_test, num_clients,train_split=0.8):

    return 0

def split_data_noniid2(x_train, y_train, x_test, y_test, num_clients,train_split=0.8):
    return 0

def split_data_by_attack_type2(x_train, y_train, x_test, y_test, num_clients, clients_special_distribution, label_encoder, train_split=0.8):
    n = num_clients
    seed = 42

    x_trains = {}
    y_trains = {}
    x_tests = {}
    y_tests = {}
    
    special_distribution = {}
    attack_cat_index = {}
    for i, attack in enumerate(clients_special_distribution.keys()):
        print(attack)
        attack_cat_id = label_encoder.transform([attack])[0]
        print(attack_cat_id)
        special_distribution[attack_cat_id] = clients_special_distribution[attack]
        attack_cat_index[attack_cat_id] = np.array(np.where(y_train == attack_cat_id)[0])
    print(special_distribution)


    # split data by attack category
    """attack_cat_index = {}
    for i, attack_cat in enumerate(clients_special_distribution.keys()):
        attack_cat_index[attack_cat] = np.array(np.where(Y == attack_cat_id)[0])
        print(attack_cat_index[attack_cat])"""

    sorted_index_lists = [[] for _ in range(n)]

    for attack_cat in special_distribution.keys():
        if special_distribution[attack_cat] == "" or special_distribution[attack_cat] is None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(attack_cat_index[attack_cat]))
            partition_size = math.ceil(len(attack_cat_index[attack_cat]) / n)
            for i in range(n):
                start = i * partition_size
                end = (i + 1) * partition_size if i < n - 1 else len(attack_cat_index[attack_cat])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][shuffled_indices[start:end]])
        else:
            partition_sizes = [
                math.ceil(len(attack_cat_index[attack_cat]) * val)
                for val in special_distribution[attack_cat]
            ]
            np.random.seed(seed)
            np.random.shuffle(attack_cat_index[attack_cat])
            for i in range(n):
                start = sum(partition_sizes[:i])
                end = sum(partition_sizes[:i + 1])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][start:end])
    
    for i in range(n):
        sorted_index_lists[i] = np.array(sorted_index_lists[i])
    
    # check if all the data is taken into account
    total_index = np.concatenate(sorted_index_lists)
    try:
        assert len(np.unique(total_index)) == len(x_train)
    except AssertionError as e:
        print("WARNING : some data are not taken into account")

    for i in range(num_clients):
        partition_size = len(sorted_index_lists[i])
        client_train_indices = np.random.choice(sorted_index_lists[i], int(partition_size * train_split), replace=False)
        x_trains[str(i)] = pd.DataFrame(x_train).iloc[client_train_indices]
        y_trains[str(i)] = y_train[client_train_indices]
        client_test_indices = np.setdiff1d(sorted_index_lists[i], client_train_indices)
        x_tests[str(i)] = pd.DataFrame(x_train).iloc[client_test_indices]
        y_tests[str(i)] = y_train[client_test_indices]

    return x_trains, y_trains, x_tests, y_tests


def split_data_by_attack_type3(x_train, y_train, x_test, y_test, num_clients, clients_special_distribution, label_encoder, train_split=0.8):
    n = num_clients
    seed = 42

    x_trains = {}
    y_trains = {}
    x_tests = {}
    y_tests = {}

    
    special_distribution = {}
    attack_cat_index = {}
    for i, attack in enumerate(clients_special_distribution.keys()):
        print(attack)
        attack_cat_id = label_encoder.transform([attack])[0]
        print(attack_cat_id)
        special_distribution[attack_cat_id] = clients_special_distribution[attack]
        attack_cat_index[attack_cat_id] = np.array(np.where(y_train == attack_cat_id)[0])
    print(special_distribution)

    sorted_index_lists = [[] for _ in range(n)]
    all_test_indices = np.arange(x_test.shape[0])
    partition_size_ = x_test.shape[0] // num_clients
    for i in range(num_clients):
        client_test_indices = np.random.choice(all_test_indices, int(partition_size_*(1-train_split)), replace=False)
        x_tests[str(i)] = pd.DataFrame(x_test).iloc[client_test_indices]
        y_tests[str(i)] = y_test[client_test_indices]
     

    for attack_cat in special_distribution.keys():
        if special_distribution[attack_cat] == "" or special_distribution[attack_cat] is None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(attack_cat_index[attack_cat]))
            partition_size = math.ceil(len(attack_cat_index[attack_cat]) / n)
            for i in range(n):
                start = i * partition_size
                end = (i + 1) * partition_size if i < n - 1 else len(attack_cat_index[attack_cat])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][shuffled_indices[start:end]])
        else:
            partition_sizes = [
                math.ceil(len(attack_cat_index[attack_cat]) * val)
                for val in special_distribution[attack_cat]
            ]
            np.random.seed(seed)
            np.random.shuffle(attack_cat_index[attack_cat])
            for i in range(n):
                start = sum(partition_sizes[:i])
                end = sum(partition_sizes[:i + 1])
                sorted_index_lists[i].extend(attack_cat_index[attack_cat][start:end])
    
    for i in range(n):
        sorted_index_lists[i] = np.array(sorted_index_lists[i])
    
    # check if all the data is taken into account
    total_index = np.concatenate(sorted_index_lists)
    try:
        assert len(np.unique(total_index)) == len(x_train)
    except AssertionError as e:
        print("WARNING : some data are not taken into account")

    for i in range(num_clients):
        partition_size = len(sorted_index_lists[i])
        client_train_indices = np.random.choice(sorted_index_lists[i], int(partition_size * train_split), replace=False)
        x_trains[str(i)] = pd.DataFrame(x_train).iloc[client_train_indices]
        y_trains[str(i)] = y_train[client_train_indices]


    return x_trains, y_trains, x_tests, y_tests

def split_data_by_attack_type(x_train, y_train, x_test, y_test, num_clients, clients_special_distribution, train_split=0.8):
    n = num_clients
    seed = 42  # Set your desired seed value

    x_trains = {}
    y_trains = {}
    x_tests = {}
    y_tests = {}

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(y_train)
    classes = label_encoder.classes_

    attack_index = {}
    for i, attack in enumerate(classes):
        numerical_attack = label_encoder.transform([attack])[0]
        attack_index[attack] = np.where(y_train == numerical_attack)[0]

    sorted_index_lists = [[] for _ in range(n)]
    for attack in classes:
        print(clients_special_distribution[attack])
        if clients_special_distribution[attack] == "" or clients_special_distribution[attack] is None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(attack_index[attack]))
            partition_size = math.ceil(len(attack_index[attack]) / n)
            for i in range(n):
                start = i * partition_size
                end = (i + 1) * partition_size if i < n - 1 else len(attack_index[attack])
                sorted_index_lists[i].extend(attack_index[attack][shuffled_indices[start:end]])
        else:
            partition_size = len(attack_index[attack]) * clients_special_distribution[attack]
            np.random.seed(seed)
            np.random.shuffle(attack_index[attack])
            for i in range(n):
                start = int(sum(partition_size[:i]))
                end = int(sum(partition_size[:i + 1]))
                sorted_index_lists[i].extend(attack_index[attack][start:end])

    for i in range(n):
        sorted_index_lists[i] = np.array(sorted_index_lists[i])

    # Check if all the data is taken into account
    total_index = np.concatenate(sorted_index_lists)
    try:
        assert len(np.unique(total_index)) == len(x_train)
    except AssertionError as e:
        print("WARNING: Some data are not taken into account")

    for i in range(num_clients):
        partition_size = len(sorted_index_lists[i])
        client_train_indices = np.random.choice(sorted_index_lists[i], int(partition_size * train_split), replace=False)
        x_trains[str(i)] = pd.DataFrame(x_train).iloc[client_train_indices]
        y_trains[str(i)] = y_train[client_train_indices]
        client_test_indices = np.setdiff1d(sorted_index_lists[i], client_train_indices)
        x_tests[str(i)] = pd.DataFrame(x_train).iloc[client_test_indices]
        y_tests[str(i)] = y_train[client_test_indices]

    return x_trains, y_trains, x_tests, y_tests

      

def split_data_by_gnb(num_clients,train_split=0.8):
    x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
    x_trains[str(0)], x_tests[str(0)], y_trains[str(0)], y_tests[str(0)], label_encoder = load_data('../dataset/BTS_1.csv')
    x_trains[str(1)], x_tests[str(1)], y_trains[str(1)], y_tests[str(1)], label_encoder = load_data('../dataset/BTS_2.csv')
    return x_trains, y_trains, x_tests, y_tests

def split_data_by_client(num_clients=2,train_split=0.8):
    return 0

def split_data(x_train, y_train, x_test, y_test, num_clients):
    partition_size = x_train.shape[0] // num_clients
    client_id_to_indices = {}
    beg_ids = [i * partition_size for i in range(num_clients)]
    end_ids = [i * partition_size for i in range(1, num_clients + 1)]
    for client_id, (beg_id, end_id) in enumerate(zip(beg_ids, end_ids)):
        client_id_to_indices[client_id] = [beg_id, end_id]

    # Create a list of indices to split the data
    #split_indices = [i * partition_size for i in range(num_clients)] + [x_train.shape[0]]

    print("-------------------------------------------")
    print(x_train.shape[0])
    # Split the data
    #x_split = np.split(x_train, split_indices)
    #y_split = np.split(y_train, split_indices)
    # Check if the length of arrays is divisible by num_clients
    if len(x_train) % num_clients != 0 or len(y_train) % num_clients != 0:
        # Handle the case where the length is not divisible
        # For example, you can truncate the arrays to be divisible
        x_train = x_train[:len(x_train) // num_clients * num_clients]
        y_train = y_train[:len(y_train) // num_clients * num_clients]
    x_split = np.split(x_train, num_clients)
    y_split = np.split(y_train, num_clients)
    num_data_in_split = x_split[0].shape[0]
    print(num_data_in_split)
    print("-------------------------------------------")
    train_split = 0.8
    x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
    for idx, (client_x, client_y) in enumerate(zip(x_split, y_split)):
        train_end_idx = int(0.8 * num_data_in_split)
        x_trains[str(idx)] = client_x[:train_end_idx]
        y_trains[str(idx)] = client_y[:train_end_idx]
        x_tests[str(idx)] = client_x[train_end_idx:]
        y_tests[str(idx)] = client_y[train_end_idx:]
    return x_trains, y_trains, x_tests, y_tests

def split_data_by_slice(num_clients=2,train_split=0.8):
    x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
    x_trains[str(0)], x_tests[str(0)], y_trains[str(0)], y_tests[str(0)], label_encoder = load_data('../dataset/URLLC.csv')
    x_trains[str(1)], x_tests[str(1)], y_trains[str(1)], y_tests[str(1)], label_encoder = load_data('../dataset/mMTCcsv')
    x_trains[str(2)], x_tests[str(2)], y_trains[str(2)], y_tests[str(2)], label_encoder = load_data('../dataset/eMBB.csv')
    return x_trains, y_trains, x_tests, y_tests


def split_data_dirichlet_old(x_train, y_train, x_test, y_test, num_clients, alpha=0.1, train_split=0.8):
    # Number of data points in the training set
    num_data_points = x_train.shape[0]
    # Generate the Dirichlet distribution
    proportions = dirichlet.rvs(alpha=[alpha] * num_clients, size=1)[0]
    
    # Calculate the number of samples per client based on the proportion and training split
    client_train_sizes = (proportions * num_data_points * train_split).astype(int)
    client_test_sizes = (proportions * num_data_points * (1 - train_split)).astype(int)
    
    all_indices = np.arange(num_data_points)
    np.random.shuffle(all_indices)
    
    x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
    start_idx = 0

    # Assign train and test data for each client based on calculated sizes
    for i in range(num_clients):
        end_train_idx = start_idx + client_train_sizes[i]
        train_indices = all_indices[start_idx:end_train_idx]
        
        start_test_idx = end_train_idx
        end_test_idx = start_test_idx + client_test_sizes[i]
        test_indices = all_indices[start_test_idx:end_test_idx]
        
        # Creating training and testing datasets for each client
        x_trains[str(i)] = pd.DataFrame(x_train).iloc[train_indices]
        y_trains[str(i)] = y_train.iloc[train_indices]
        x_tests[str(i)] = pd.DataFrame(x_train).iloc[test_indices]
        y_tests[str(i)] = y_train.iloc[test_indices]
        
        # Update the start index for the next client
        start_idx = end_test_idx
    
    return x_trains, y_trains, x_tests, y_tests


#https://github.com/CasellaJr/Benchmarking-FedAvg-and-FedCurv-for-Image-Classification-Tasks/blob/main/non-iidness.py
def split_data_dirichlet_2(x_train, y_train, x_test, y_test, num_clients, alpha=0.1, train_split=0.8):
    """Split the data."""
    np.random.seed(self.seed)
    classes = len(np.unique(y_train))
    min_size = 0

    n = len(data)
    while min_size < self.min_samples_per_col:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(classes):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(self.alpha, num_clie,ts))
            proportions = [p * (len(idx_j) < n / num_clients)
                            for p, idx_j in zip(proportions, idx_batch)]
            proportions = np.array(proportions)
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_splitted = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_splitted)]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    return idx_batch


def split_data_dirichlet(x_train, y_train, x_test, y_test, num_clients, alpha=0.5, train_split=0.8):

    # Get unique classes and the number of data points
    unique_classes = np.unique(y_train)
    class_indices = {cls: np.where(y_train == cls)[0] for cls in unique_classes}
    
    # Dictionaries to hold the split data
    x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}

    all_test_indices = np.arange(x_test.shape[0])
    partition_size_ = x_test.shape[0] // num_clients
        
    
    # Initialize empty dataframes and arrays for each client
    for i in range(num_clients):
        x_trains[str(i)] = pd.DataFrame()
        y_trains[str(i)] = np.array([], dtype=y_train.dtype)
        #x_tests[str(i)] = pd.DataFrame()
        #y_tests[str(i)] = np.array([], dtype=y_train.dtype)
        client_test_indices = np.random.choice(all_test_indices, int(partition_size_), replace=False)
        x_tests[str(i)] = pd.DataFrame(x_test).iloc[client_test_indices]
        if isinstance(y_test, pd.Series):
            y_tests[str(i)] = y_test.iloc[client_test_indices].values
        elif isinstance(y_test, np.ndarray):
            y_tests[str(i)] = y_test[client_test_indices]

    # Loop over each client to distribute the data
    for i in range(num_clients):
        # For each class, distribute indices using a Dirichlet distribution
        for cls in unique_classes:
            # Get indices for this class
            indices = class_indices[cls]
            np.random.shuffle(indices)  # Shuffle indices to randomize data points
            
            # Generate proportions for this class
            proportions = dirichlet.rvs(alpha=[alpha] * num_clients, size=1)[0]
            num_train = int(round(len(indices) )) #* train_split))
            
            # Calculate number of training and testing indices for this class per client
            train_sizes = (proportions * num_train).astype(int)
            test_sizes = (proportions * (len(indices) - num_train)).astype(int)
            
            # Calculate the starting index for this client
            start_train_idx = sum(train_sizes[:i])
            start_test_idx = sum(test_sizes[:i])
            
            # Slice the indices array to get this client's training and testing indices
            client_train_indices = indices[start_train_idx:start_train_idx + train_sizes[i]]
            client_test_indices = indices[num_train + start_test_idx:num_train + start_test_idx + test_sizes[i]]
            
            # Append the data to the client's training and testing sets
            if len(client_train_indices) > 0:
                x_trains[str(i)] = pd.concat([x_trains[str(i)], x_train.iloc[client_train_indices]], ignore_index=True)
                if isinstance(y_train, pd.Series):
                    y_trains[str(i)] = np.append(y_trains[str(i)], y_train.iloc[client_train_indices].values)
                elif isinstance(y_train, np.ndarray):
                    y_trains[str(i)] = np.append(y_trains[str(i)], y_train[client_train_indices])
                #y_trains[str(i)] = np.append(y_trains[str(i)], y_train[client_train_indices])
            """if len(client_test_indices) > 0:
                x_tests[str(i)] = pd.concat([x_tests[str(i)], x_train.iloc[client_test_indices]], ignore_index=True)
                y_tests[str(i)] = np.append(y_tests[str(i)], y_train[client_test_indices])"""
    
    return x_trains, y_trains, x_tests, y_tests
    

def split_data(label_encoder, split_type= "random", x_train=None, y_train=None, x_test=None, y_test=None, num_clients=2, clients_special_distribution=None, alpha=0.1):
    if split_type == "random":
        return split_data_random(x_train, y_train, x_test, y_test, num_clients)
    elif split_type == "noniid":
        return split_data_noniid(x_train, y_train, x_test, y_test, num_clients)
    elif split_type == "by_attack_type":
        return split_data_by_attack_type3(x_train, y_train, x_test, y_test, num_clients, clients_special_distribution,label_encoder)
    elif split_type == "by_gnb":
        return split_data_by_gnb(num_clients)
    elif split_type == "by_client":
        return split_data_by_client(num_clients)
    elif split_type == "by_slice":
        return split_data_by_slice(num_clients)
    elif split_type == "dirichlet":
        return split_data_dirichlet(x_train, y_train, x_test, y_test, num_clients,alpha)
    else:
        return split_data_random(x_train, y_train, x_test, y_test, num_clients)