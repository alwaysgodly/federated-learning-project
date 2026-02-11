"""
Federated Learning Client
Trains models locally on private data without sharing raw data
"""

import numpy as np
from typing import Dict, Tuple
import time


class FederatedClient:
    """
    Client that trains models on local data while preserving privacy
    """
    
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray = None, y_test: np.ndarray = None):
        """
        Initialize federated client
        
        Args:
            client_id: Unique identifier for this client
            X_train: Training features (kept private)
            y_train: Training labels (kept private)
            X_test: Optional test features
            y_test: Optional test labels
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.local_weights = None
        self.data_size = len(X_train)
        
        print(f"Client {client_id} initialized with {self.data_size} samples")
    
    def receive_global_model(self, global_weights: Dict[str, np.ndarray]):
        """Receive and store global model from server"""
        self.local_weights = {k: v.copy() for k, v in global_weights.items()}
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward propagation through the network
        
        Returns:
            predictions, cache of intermediate values
        """
        # Layer 1
        Z1 = np.dot(X, self.local_weights['W1']) + self.local_weights['b1']
        A1 = self._sigmoid(Z1)
        
        # Layer 2
        Z2 = np.dot(A1, self.local_weights['W2']) + self.local_weights['b2']
        A2 = self._softmax(Z2)
        
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'X': X}
        return A2, cache
    
    def backward_pass(self, cache: Dict, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward propagation to compute gradients
        
        Returns:
            Dictionary of gradients
        """
        m = y_true.shape[0]
        
        # Output layer gradient
        dZ2 = cache['A2'] - y_true
        dW2 = (1/m) * np.dot(cache['A1'].T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.local_weights['W2'].T)
        dZ1 = dA1 * cache['A1'] * (1 - cache['A1'])
        dW1 = (1/m) * np.dot(cache['X'].T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return gradients
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-8)
        loss = np.sum(log_probs) / m
        return loss
    
    def local_training(self, epochs: int = 5, learning_rate: float = 0.01, 
                      batch_size: int = 32) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Train model locally on private data
        
        Args:
            epochs: Number of local training epochs
            learning_rate: Learning rate for gradient descent
            batch_size: Mini-batch size
            
        Returns:
            Updated weights, training metrics
        """
        print(f"\n{self.client_id} starting local training...")
        start_time = time.time()
        
        n_samples = len(self.X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        epoch_losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            batch_losses = []
            
            for batch_idx in range(n_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred, cache = self.forward_pass(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                batch_losses.append(loss)
                
                # Backward pass
                gradients = self.backward_pass(cache, y_batch)
                
                # Update weights
                self.local_weights['W1'] -= learning_rate * gradients['dW1']
                self.local_weights['b1'] -= learning_rate * gradients['db1']
                self.local_weights['W2'] -= learning_rate * gradients['dW2']
                self.local_weights['b2'] -= learning_rate * gradients['db2']
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluate on local test set if available
        metrics = {
            'training_time': training_time,
            'final_loss': epoch_losses[-1],
            'data_size': self.data_size
        }
        
        if self.X_test is not None and self.y_test is not None:
            test_acc = self.evaluate(self.X_test, self.y_test)
            metrics['test_accuracy'] = test_acc
            print(f"  Test Accuracy: {test_acc:.4f}")
        
        print(f"{self.client_id} completed training in {training_time:.2f}s")
        
        return self.local_weights, metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy"""
        y_pred, _ = self.forward_pass(X)
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
    
    def get_model_update(self) -> Tuple[str, Dict[str, np.ndarray], int]:
        """
        Return model update for server aggregation
        
        Returns:
            Tuple of (client_id, weights, data_size)
        """
        return (self.client_id, self.local_weights, self.data_size)
    
    def add_differential_privacy(self, epsilon: float = 1.0):
        """
        Add differential privacy noise to model weights
        
        Args:
            epsilon: Privacy budget (smaller = more privacy)
        """
        sensitivity = 0.01  # Maximum change in weights
        noise_scale = sensitivity / epsilon
        
        for key in self.local_weights.keys():
            noise = np.random.laplace(0, noise_scale, self.local_weights[key].shape)
            self.local_weights[key] += noise
        
        print(f"{self.client_id}: Added differential privacy (ε={epsilon})")