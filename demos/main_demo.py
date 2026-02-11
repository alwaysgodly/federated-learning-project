"""
Main Federated Learning Demo
Demonstrates complete federated learning workflow with privacy and security
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.fl_server import FederatedServer
from src.fl_clients import FederatedClient
from src.data_uitls import (
    load_digits_dataset, preprocess_data, partition_data_federated,
    print_data_statistics, estimate_model_size, simulate_bandwidth_constraints
)


def run_federated_learning_demo(n_clients: int = 5, n_rounds: int = 10, 
                                iid: bool = True, security_enabled: bool = True):
    """
    Run complete federated learning demonstration
    
    Args:
        n_clients: Number of federated clients
        n_rounds: Number of federated learning rounds
        iid: Use IID or non-IID data distribution
        security_enabled: Enable security features
    """
    print("\n" + "="*80)
    print("FEDERATED LEARNING DEMONSTRATION")
    print("Addresses: Privacy, Security, Bandwidth, Data Localization, Coordination")
    print("="*80)
    
    
    print("\n[STEP 1] Loading Dataset...")
    X, y = load_digits_dataset()
    print(f"Dataset: Handwritten Digits (0-9)")
    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    
    
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    
    print("\n[STEP 2] Data Localization - Partitioning Data Across Clients...")
    partitioned_data = partition_data_federated(
        X_processed, y_processed, 
        n_clients=n_clients, 
        iid=iid
    )
    print_data_statistics(partitioned_data)
    
    
    print("[STEP 3] Initializing Federated Server...")
    model_architecture = {
        'input_size': X_processed.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    
    server = FederatedServer(
        model_architecture=model_architecture,
        security_enabled=security_enabled
    )
    print(f"Server initialized with security={'ENABLED' if security_enabled else 'DISABLED'}")
    
    
    model_size_mb = estimate_model_size(server.get_global_model())
    print(f"Global model size: {model_size_mb:.4f} MB")
    
    
    print("\n[STEP 4] Creating Federated Clients with Local Data...")
    clients = []
    for client_id, data in partitioned_data['client_data'].items():
        client = FederatedClient(
            client_id=client_id,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=partitioned_data['X_test'],
            y_test=partitioned_data['y_test']
        )
        clients.append(client)
    
    
    print("\n[STEP 5] Starting Federated Training Rounds...")
    print("Key Features:")
    print("  ✓ Privacy: Raw data never leaves client devices")
    print("  ✓ Security: Secure aggregation with verification")
    print("  ✓ Bandwidth: Only model updates transmitted (not raw data)")
    print("  ✓ Coordination: Cloud-based server orchestrates training")
    
    global_test_accuracies = []
    round_times = []
    total_bandwidth_used = 0
    
    for round_num in range(n_rounds):
        print(f"\n{'='*80}")
        print(f"FEDERATED ROUND {round_num + 1}/{n_rounds}")
        print('='*80)
        
        
        global_model = server.get_global_model()
        for client in clients:
            client.receive_global_model(global_model)
        
        
        print("\n[Privacy Protection] Clients training on local data...")
        client_updates = []
        client_metrics = []
        
        for client in clients:
            
            local_weights, metrics = client.local_training(
                epochs=3,
                learning_rate=0.1,
                batch_size=32
            )
            
            client_updates.append(client.get_model_update())
            client_metrics.append(metrics)
        
        
        print("\n[Bandwidth Analysis]")
        updates_size = model_size_mb * len(clients) * 2  
        total_bandwidth_used += updates_size
        transfer_time = simulate_bandwidth_constraints(updates_size, bandwidth_mbps=50)
        print(f"  Data transferred: {updates_size:.4f} MB")
        print(f"  Estimated transfer time: {transfer_time:.2f} seconds")
        print(f"  Raw data NOT transmitted (privacy preserved)")
        
        
        print("\n[Secure Aggregation & Coordination]")
        avg_client_accuracy = np.mean([m.get('test_accuracy', 0) for m in client_metrics])
        
        round_stats = server.federated_round(
            client_updates,
            round_metrics={
                'avg_client_test_accuracy': avg_client_accuracy,
                'bandwidth_mb': updates_size
            }
        )
        
        
        print("\n[Global Model Evaluation]")
        
        clients[0].receive_global_model(server.get_global_model())
        global_accuracy = clients[0].evaluate(
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        
        global_test_accuracies.append(global_accuracy)
        round_times.append(round_stats['aggregation_time'])
        
        print(f"Global Model Accuracy: {global_accuracy:.4f}")
        print(f"Improvement from clients: {global_accuracy - avg_client_accuracy:.4f}")
    
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING COMPLETED")
    print("="*80)
    
    print(f"\nFinal Global Model Accuracy: {global_test_accuracies[-1]:.4f}")
    print(f"Total Bandwidth Used: {total_bandwidth_used:.2f} MB")
    print(f"Average Round Time: {np.mean(round_times):.4f} seconds")
    print(f"Total Rounds: {n_rounds}")
    
    
    raw_data_size = (X_processed.nbytes + y_processed.nbytes) / (1024 * 1024)
    print(f"\nPrivacy Benefit:")
    print(f"  Raw data size: {raw_data_size:.2f} MB (NEVER transmitted)")
    print(f"  Model updates only: {total_bandwidth_used:.2f} MB")
    print(f"  Bandwidth savings: {((raw_data_size - total_bandwidth_used) / raw_data_size * 100):.1f}%")
    
    
    print("\n[STEP 6] Generating Visualizations...")
    create_visualizations(
        global_test_accuracies,
        partitioned_data,
        server.training_history
    )
    
    
    print("\n[STEP 7] Saving Final Model...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'final_model.pkl')
    server.save_model(model_path)
    
    return server, clients, global_test_accuracies


def create_visualizations(accuracies, partitioned_data, training_history):
    """Create visualizations of federated learning results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    ax1 = axes[0, 0]
    rounds = range(1, len(accuracies) + 1)
    ax1.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Global Test Accuracy', fontsize=12)
    ax1.set_title('Model Performance Over Federated Rounds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    
    ax2 = axes[0, 1]
    client_names = list(partitioned_data['client_data'].keys())
    data_sizes = [data['data_size'] for data in partitioned_data['client_data'].values()]
    colors = plt.cm.Set3(range(len(client_names)))
    ax2.bar(range(len(client_names)), data_sizes, color=colors)
    ax2.set_xlabel('Client', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title(f'Data Distribution ({partitioned_data["distribution"]})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(client_names)))
    ax2.set_xticklabels([f'C{i+1}' for i in range(len(client_names))])
    ax2.grid(True, alpha=0.3, axis='y')
    
    
    ax3 = axes[1, 0]
    bandwidth_per_round = [h.get('bandwidth_mb', 0) for h in training_history]
    ax3.plot(range(1, len(bandwidth_per_round) + 1), bandwidth_per_round, 
             marker='s', linewidth=2, color='coral', markersize=8)
    ax3.set_xlabel('Federated Round', fontsize=12)
    ax3.set_ylabel('Bandwidth (MB)', fontsize=12)
    ax3.set_title('Bandwidth Usage Per Round', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    
    ax4 = axes[1, 1]
    agg_times = [h.get('aggregation_time', 0) * 1000 for h in training_history]  # Convert to ms
    ax4.plot(range(1, len(agg_times) + 1), agg_times, 
             marker='^', linewidth=2, color='green', markersize=8)
    ax4.set_xlabel('Federated Round', fontsize=12)
    ax4.set_ylabel('Aggregation Time (ms)', fontsize=12)
    ax4.set_title('Server Aggregation Time', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'training_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {output_path}")
    
    plt.close()


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("FEDERATED LEARNING PROJECT")
    print("Solving Privacy, Security, Bandwidth, and Coordination Challenges")
    print("="*80)
    
    
    config = {
        'n_clients': 5,
        'n_rounds': 10,
        'iid': False,  
        'security_enabled': True
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
   
    server, clients, accuracies = run_federated_learning_demo(**config)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("✓ Privacy: Raw data never left client devices")
    print("✓ Security: Updates verified and securely aggregated")
    print("✓ Bandwidth: Only model updates transmitted (efficient)")
    print("✓ Data Localization: Each client maintains local data")
    print("✓ Coordination: Cloud server orchestrates distributed training")
    print("\nCheck the results folder for visualizations!")


if __name__ == "__main__":
    main()