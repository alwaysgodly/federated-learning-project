"""
Data utilities for federated learning
Handles data partitioning and preprocessing
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_synthetic_dataset(n_samples: int = 1000, n_features: int = 20, 
                             n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        X, y arrays
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_classes=n_classes,
        random_state=42
    )
    return X, y


def load_digits_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load digits dataset (0-9 digit recognition)
    
    Returns:
        X, y arrays
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y


def preprocess_data(X: np.ndarray, y: np.ndarray, 
                   n_classes: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features and labels
    
    Args:
        X: Feature array
        y: Label array
        n_classes: Number of classes for one-hot encoding
        
    Returns:
        Preprocessed X, y
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode labels
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X_scaled, y_onehot


def create_non_iid_distribution(y: np.ndarray, n_clients: int, 
                                 concentration: float = 0.5) -> List[np.ndarray]:
    """
    Create non-IID (non-independent and identically distributed) data split
    Simulates realistic federated scenarios where clients have different data distributions
    
    Args:
        y: Labels array
        n_clients: Number of clients
        concentration: How concentrated data is (lower = more non-IID)
        
    Returns:
        List of indices for each client
    """
    n_classes = len(np.unique(y))
    class_indices = [np.where(y == i)[0] for i in range(n_classes)]
    
    client_indices = [[] for _ in range(n_clients)]
    
    # Distribute data with bias
    for class_idx, indices in enumerate(class_indices):
        # Create biased distribution using Dirichlet
        proportions = np.random.dirichlet([concentration] * n_clients)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        # Split indices according to proportions
        splits = np.split(indices, proportions)
        
        for client_idx, split in enumerate(splits):
            client_indices[client_idx].extend(split)
    
    # Shuffle each client's indices
    for idx in range(n_clients):
        np.random.shuffle(client_indices[idx])
    
    return client_indices


def create_iid_distribution(n_samples: int, n_clients: int) -> List[np.ndarray]:
    """
    Create IID (independent and identically distributed) data split
    
    Args:
        n_samples: Total number of samples
        n_clients: Number of clients
        
    Returns:
        List of indices for each client
    """
    indices = np.random.permutation(n_samples)
    split_size = n_samples // n_clients
    
    client_indices = []
    for i in range(n_clients):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_clients - 1 else n_samples
        client_indices.append(indices[start_idx:end_idx])
    
    return client_indices


def partition_data_federated(X: np.ndarray, y: np.ndarray, n_clients: int,
                             iid: bool = True, test_size: float = 0.2) -> Dict:
    """
    Partition data for federated learning simulation
    
    Args:
        X: Feature array
        y: Label array (one-hot encoded)
        n_clients: Number of clients
        iid: Whether to use IID distribution
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary with client data and global test set
    """
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    # Get label indices for distribution
    y_labels = np.argmax(y_train, axis=1)
    
    # Create distribution
    if iid:
        client_indices = create_iid_distribution(len(X_train), n_clients)
    else:
        client_indices = create_non_iid_distribution(y_labels, n_clients, concentration=0.5)
    
    # Create client datasets
    client_data = {}
    for i, indices in enumerate(client_indices):
        client_id = f"client_{i+1}"
        client_data[client_id] = {
            'X_train': X_train[indices],
            'y_train': y_train[indices],
            'data_size': len(indices)
        }
    
    # Add statistics
    data_sizes = [len(indices) for indices in client_indices]
    
    result = {
        'client_data': client_data,
        'X_test': X_test,
        'y_test': y_test,
        'distribution': 'IID' if iid else 'Non-IID',
        'n_clients': n_clients,
        'data_sizes': data_sizes,
        'total_train_samples': len(X_train),
        'total_test_samples': len(X_test)
    }
    
    return result


def print_data_statistics(partitioned_data: Dict):
    """Print statistics about the federated data partition"""
    print("\n" + "="*60)
    print("FEDERATED DATA STATISTICS")
    print("="*60)
    print(f"Distribution: {partitioned_data['distribution']}")
    print(f"Number of clients: {partitioned_data['n_clients']}")
    print(f"Total training samples: {partitioned_data['total_train_samples']}")
    print(f"Total test samples: {partitioned_data['total_test_samples']}")
    print("\nData per client:")
    
    for client_id, data in partitioned_data['client_data'].items():
        data_size = data['data_size']
        percentage = (data_size / partitioned_data['total_train_samples']) * 100
        print(f"  {client_id}: {data_size} samples ({percentage:.1f}%)")
    
    print("="*60 + "\n")


def simulate_bandwidth_constraints(data_size_mb: float, bandwidth_mbps: float = 10) -> float:
    """
    Simulate time taken to transfer data given bandwidth constraints
    
    Args:
        data_size_mb: Data size in megabytes
        bandwidth_mbps: Available bandwidth in Mbps
        
    Returns:
        Transfer time in seconds
    """
    transfer_time = (data_size_mb * 8) / bandwidth_mbps  # Convert MB to Mb
    return transfer_time


def estimate_model_size(weights: Dict[str, np.ndarray]) -> float:
    """
    Estimate model size in megabytes
    
    Args:
        weights: Model weight dictionary
        
    Returns:
        Model size in MB
    """
    total_bytes = 0
    for key, value in weights.items():
        total_bytes += value.nbytes
    
    size_mb = total_bytes / (1024 * 1024)
    return size_mb