"""
Federated Learning Server
Handles model aggregation and coordination of client training
"""

import numpy as np
import pickle
import hashlib
import time
from typing import List, Dict, Tuple
import json


class FederatedServer:
    """
    Central server for federated learning that coordinates training
    without accessing raw client data.
    """
    
    def __init__(self, model_architecture: Dict, security_enabled: bool = True):
        """
        Initialize federated server
        
        Args:
            model_architecture: Dictionary defining model structure
            security_enabled: Enable secure aggregation
        """
        self.model_architecture = model_architecture
        self.global_weights = self._initialize_weights()
        self.security_enabled = security_enabled
        self.round_number = 0
        self.training_history = []
        self.client_contributions = {}
        
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize random weights for the global model"""
        np.random.seed(42)
        weights = {}
        
        # Input to hidden layer
        input_size = self.model_architecture['input_size']
        hidden_size = self.model_architecture['hidden_size']
        weights['W1'] = np.random.randn(input_size, hidden_size) * 0.01
        weights['b1'] = np.zeros((1, hidden_size))
        
        # Hidden to output layer
        output_size = self.model_architecture['output_size']
        weights['W2'] = np.random.randn(hidden_size, output_size) * 0.01
        weights['b2'] = np.zeros((1, output_size))
        
        return weights
    
    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Return current global model weights"""
        return self.global_weights.copy()
    
    def aggregate_updates(self, client_updates: List[Dict], 
                         client_data_sizes: List[int]) -> Dict[str, np.ndarray]:
        """
        Federated Averaging (FedAvg) algorithm
        Aggregates client model updates weighted by their data sizes
        
        Args:
            client_updates: List of weight dictionaries from clients
            client_data_sizes: Number of samples each client trained on
            
        Returns:
            Aggregated global weights
        """
        if not client_updates:
            return self.global_weights
        
        total_samples = sum(client_data_sizes)
        aggregated_weights = {}
        
        # Initialize aggregated weights
        for key in client_updates[0].keys():
            aggregated_weights[key] = np.zeros_like(client_updates[0][key])
        
        # Weighted averaging
        for client_weights, data_size in zip(client_updates, client_data_sizes):
            weight = data_size / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight * client_weights[key]
        
        return aggregated_weights
    
    def secure_aggregation(self, client_updates: List[Dict], 
                          client_data_sizes: List[int]) -> Dict[str, np.ndarray]:
        """
        Simulate secure aggregation with noise addition for privacy
        In real implementation, this would use techniques like:
        - Secure multi-party computation
        - Homomorphic encryption
        - Differential privacy
        """
        # Standard aggregation
        aggregated = self.aggregate_updates(client_updates, client_data_sizes)
        
        # Add differential privacy noise (simplified)
        if self.security_enabled:
            noise_scale = 0.001  # Small noise for privacy
            for key in aggregated.keys():
                noise = np.random.normal(0, noise_scale, aggregated[key].shape)
                aggregated[key] += noise
        
        return aggregated
    
    def verify_client_update(self, client_id: str, weights: Dict) -> bool:
        """
        Verify integrity of client update using checksums
        Prevents malicious updates
        """
        # Create checksum of weights
        weights_bytes = pickle.dumps(weights)
        checksum = hashlib.sha256(weights_bytes).hexdigest()
        
        # In real system, compare with expected checksum
        # For demo, just verify weights are valid numpy arrays
        for key, value in weights.items():
            if not isinstance(value, np.ndarray):
                return False
            if np.isnan(value).any() or np.isinf(value).any():
                return False
        
        return True
    
    def federated_round(self, client_updates: List[Tuple[str, Dict, int]], 
                       round_metrics: Dict = None) -> Dict:
        """
        Execute one round of federated learning
        
        Args:
            client_updates: List of (client_id, weights, data_size) tuples
            round_metrics: Optional metrics from clients
            
        Returns:
            Round statistics
        """
        self.round_number += 1
        print(f"\n=== Federated Learning Round {self.round_number} ===")
        
        # Separate client information
        client_ids = [update[0] for update in client_updates]
        client_weights = [update[1] for update in client_updates]
        client_data_sizes = [update[2] for update in client_updates]
        
        # Verify all client updates
        valid_updates = []
        valid_sizes = []
        
        for client_id, weights, data_size in client_updates:
            if self.verify_client_update(client_id, weights):
                valid_updates.append(weights)
                valid_sizes.append(data_size)
                self.client_contributions[client_id] = \
                    self.client_contributions.get(client_id, 0) + 1
            else:
                print(f"Warning: Invalid update from {client_id}, skipping")
        
        print(f"Valid updates: {len(valid_updates)}/{len(client_updates)}")
        
        # Aggregate using secure aggregation
        start_time = time.time()
        self.global_weights = self.secure_aggregation(valid_updates, valid_sizes)
        aggregation_time = time.time() - start_time
        
        # Compile round statistics
        round_stats = {
            'round': self.round_number,
            'participating_clients': len(valid_updates),
            'total_samples': sum(valid_sizes),
            'aggregation_time': aggregation_time,
            'security_enabled': self.security_enabled
        }
        
        if round_metrics:
            round_stats.update(round_metrics)
        
        self.training_history.append(round_stats)
        
        print(f"Aggregation completed in {aggregation_time:.4f}s")
        
        return round_stats
    
    def get_training_history(self) -> List[Dict]:
        """Return complete training history"""
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save global model to file"""
        model_data = {
            'weights': self.global_weights,
            'architecture': self.model_architecture,
            'round': self.round_number,
            'history': self.training_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load global model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.global_weights = model_data['weights']
        self.model_architecture = model_data['architecture']
        self.round_number = model_data['round']
        self.training_history = model_data.get('history', [])
        print(f"Model loaded from {filepath}")