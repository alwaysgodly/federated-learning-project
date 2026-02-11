"""
Comparison Demo: Centralized ML vs Federated Learning
Shows the advantages of federated learning
"""

import sys
from pathlib import Path
import os
import numpy as np
import time


src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.fl_server import FederatedServer
from src.fl_clients import FederatedClient
from src.data_uitls import (
    load_digits_dataset, preprocess_data, partition_data_federated
)


def train_centralized(X_train, y_train, X_test, y_test, epochs=30):
    """
    Simulate centralized machine learning
    All data is collected in one place
    """
    print("\n" + "="*70)
    print("CENTRALIZED MACHINE LEARNING")
    print("="*70)
    
    print("\n[Privacy Impact] ALL data collected centrally")
    print("  ⚠️  Raw data transmitted from all clients")
    print("  ⚠️  Central server has access to sensitive data")
    print("  ⚠️  Single point of failure for data breaches")
    
    
    data_size_mb = (X_train.nbytes + y_train.nbytes) / (1024 * 1024)
    print(f"\n[Bandwidth Usage] {data_size_mb:.2f} MB of raw data transmitted")
    
    
    print("\n[Training] Centralized training on all data...")
    
    
    model_architecture = {
        'input_size': X_train.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    
    
    client = FederatedClient("central", X_train, y_train, X_test, y_test)
    server = FederatedServer(model_architecture)
    client.receive_global_model(server.get_global_model())
    
    start_time = time.time()
    _, metrics = client.local_training(epochs=epochs, learning_rate=0.1, batch_size=64)
    training_time = time.time() - start_time
    
    accuracy = client.evaluate(X_test, y_test)
    
    print(f"\n[Results]")
    print(f"  Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Training Time: {training_time:.2f}s")
    
    return {
        'accuracy': accuracy,
        'training_time': training_time,
        'bandwidth_mb': data_size_mb,
        'privacy_preserved': False,
        'data_localized': False
    }


def train_federated(X, y, n_clients=5, rounds=10):
    """
    Federated learning approach
    Data stays on client devices
    """
    print("\n" + "="*70)
    print("FEDERATED LEARNING")
    print("="*70)
    
    print("\n[Privacy Protection] Data remains on client devices")
    print("  ✓ Only model updates transmitted")
    print("  ✓ No access to raw data by central server")
    print("  ✓ Distributed security model")
    
    
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    partitioned_data = partition_data_federated(
        X_processed, y_processed, n_clients=n_clients, iid=False
    )
    
    
    model_architecture = {
        'input_size': X_processed.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    
    server = FederatedServer(model_architecture, security_enabled=True)
    
    
    from src.data_uitls import estimate_model_size
    model_size_mb = estimate_model_size(server.get_global_model())
    bandwidth_per_round = model_size_mb * n_clients * 2  # Upload + download
    
    print(f"\n[Bandwidth Usage] {model_size_mb:.4f} MB per client per round")
    print(f"  Total per round: {bandwidth_per_round:.4f} MB")
    print(f"  (vs {(X.nbytes + y.nbytes) / (1024 * 1024):.2f} MB for centralized)")
    
    
    clients = []
    for client_id, data in partitioned_data['client_data'].items():
        client = FederatedClient(
            client_id, data['X_train'], data['y_train'],
            partitioned_data['X_test'], partitioned_data['y_test']
        )
        clients.append(client)
    
    
    print(f"\n[Training] Federated training with {n_clients} clients...")
    
    start_time = time.time()
    
    for round_num in range(rounds):
        
        global_model = server.get_global_model()
        for client in clients:
            client.receive_global_model(global_model)
        
        
        client_updates = []
        for client in clients:
            local_weights, _ = client.local_training(epochs=3, learning_rate=0.1)
            client_updates.append(client.get_model_update())
        
        
        server.federated_round(client_updates)
    
    training_time = time.time() - start_time
    
    
    clients[0].receive_global_model(server.get_global_model())
    accuracy = clients[0].evaluate(
        partitioned_data['X_test'],
        partitioned_data['y_test']
    )
    
    total_bandwidth = bandwidth_per_round * rounds
    
    print(f"\n[Results]")
    print(f"  Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Total Bandwidth: {total_bandwidth:.2f} MB")
    
    return {
        'accuracy': accuracy,
        'training_time': training_time,
        'bandwidth_mb': total_bandwidth,
        'privacy_preserved': True,
        'data_localized': True
    }


def compare_approaches():
    """Compare centralized vs federated learning"""
    
    print("\n" + "="*70)
    print("COMPARISON: CENTRALIZED ML vs FEDERATED LEARNING")
    print("="*70)
    
    
    print("\nLoading dataset...")
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    
    
    centralized_results = train_centralized(X_train, y_train, X_test, y_test, epochs=30)
    federated_results = train_federated(X, y, n_clients=5, rounds=10)
    
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Centralized':<20} {'Federated':<20}")
    print("-" * 70)
    
    
    print(f"{'Accuracy':<30} {centralized_results['accuracy']:.4f} ({centralized_results['accuracy']*100:.1f}%)"
          f"{'':>5} {federated_results['accuracy']:.4f} ({federated_results['accuracy']*100:.1f}%)")
    
    
    print(f"{'Training Time (s)':<30} {centralized_results['training_time']:.2f}"
          f"{'':>14} {federated_results['training_time']:.2f}")
    
    
    print(f"{'Bandwidth (MB)':<30} {centralized_results['bandwidth_mb']:.2f}"
          f"{'':>14} {federated_results['bandwidth_mb']:.2f}")
    
    
    privacy_c = "❌ NO" if not centralized_results['privacy_preserved'] else "✓ YES"
    privacy_f = "✓ YES" if federated_results['privacy_preserved'] else "❌ NO"
    print(f"{'Privacy Preserved':<30} {privacy_c:<20} {privacy_f:<20}")
    
    
    local_c = "❌ NO" if not centralized_results['data_localized'] else "✓ YES"
    local_f = "✓ YES" if federated_results['data_localized'] else "❌ NO"
    print(f"{'Data Localized':<30} {local_c:<20} {local_f:<20}")
    
    
    print("\n" + "="*70)
    print("KEY ADVANTAGES OF FEDERATED LEARNING")
    print("="*70)
    
    print("\n1. PRIVACY & SECURITY")
    print("   ✓ Raw data never leaves client devices")
    print("   ✓ Reduces risk of centralized data breaches")
    print("   ✓ Complies with data protection regulations")
    
    print("\n2. BANDWIDTH EFFICIENCY")
    bandwidth_saved = centralized_results['bandwidth_mb'] - federated_results['bandwidth_mb']
    if bandwidth_saved > 0:
        savings_pct = (bandwidth_saved / centralized_results['bandwidth_mb']) * 100
        print(f"   ✓ Saves {bandwidth_saved:.2f} MB of bandwidth ({savings_pct:.1f}% reduction)")
    print("   ✓ Only model parameters transmitted, not raw data")
    
    print("\n3. DATA LOCALIZATION")
    print("   ✓ Meets regulatory requirements (GDPR, HIPAA, etc.)")
    print("   ✓ Users maintain control over their data")
    print("   ✓ No need for data transfer agreements")
    
    print("\n4. SCALABILITY")
    print("   ✓ Can train on distributed data sources")
    print("   ✓ No central storage bottleneck")
    print("   ✓ Easily scales to more clients")
    
    print("\n5. MODEL QUALITY")
    acc_diff = federated_results['accuracy'] - centralized_results['accuracy']
    if abs(acc_diff) < 0.05:
        print(f"   ✓ Comparable accuracy to centralized ({acc_diff:+.4f} difference)")
    elif acc_diff > 0:
        print(f"   ✓ Better accuracy than centralized (+{acc_diff:.4f})")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nFederated Learning provides:")
    print("  • Strong privacy guarantees")
    print("  • Efficient bandwidth usage")
    print("  • Regulatory compliance")
    print("  • Comparable model performance")
    print("\nMaking it ideal for privacy-sensitive applications!")
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_approaches()