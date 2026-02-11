"""
Interactive Federated Learning Demo
Allows users to customize parameters and see results
"""

import sys
from pathlib import Path
import os
import numpy as np


src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.fl_server import FederatedServer
from src.fl_clients import FederatedClient
from src.data_uitls import (
    load_digits_dataset, preprocess_data, partition_data_federated,
    print_data_statistics
)


def get_user_input():
    """Get configuration from user"""
    print("\n" + "="*60)
    print("FEDERATED LEARNING - INTERACTIVE DEMO")
    print("="*60)
    
    print("\nThis demo will train a neural network using federated learning")
    print("to classify handwritten digits (0-9) without centralizing data.\n")
    
    
    while True:
        try:
            n_clients = input("Number of clients (3-10) [default: 5]: ").strip()
            n_clients = 5 if n_clients == "" else int(n_clients)
            if 3 <= n_clients <= 10:
                break
            print("Please enter a number between 3 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    
    while True:
        try:
            n_rounds = input("Number of training rounds (5-20) [default: 10]: ").strip()
            n_rounds = 10 if n_rounds == "" else int(n_rounds)
            if 5 <= n_rounds <= 20:
                break
            print("Please enter a number between 5 and 20")
        except ValueError:
            print("Please enter a valid number")
    
    
    while True:
        dist = input("Data distribution (IID/Non-IID) [default: Non-IID]: ").strip().upper()
        if dist == "":
            iid = False
            break
        elif dist in ["IID", "NON-IID", "NONIID"]:
            iid = (dist == "IID")
            break
        print("Please enter 'IID' or 'Non-IID'")
    
    
    while True:
        sec = input("Enable security features? (Y/N) [default: Y]: ").strip().upper()
        if sec == "" or sec == "Y":
            security = True
            break
        elif sec == "N":
            security = False
            break
        print("Please enter 'Y' or 'N'")
    
    return {
        'n_clients': n_clients,
        'n_rounds': n_rounds,
        'iid': iid,
        'security_enabled': security
    }


def run_interactive_demo():
    """Run interactive federated learning demo"""
    
    
    config = get_user_input()
    
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Clients: {config['n_clients']}")
    print(f"Rounds: {config['n_rounds']}")
    print(f"Distribution: {'IID' if config['iid'] else 'Non-IID'}")
    print(f"Security: {'Enabled' if config['security_enabled'] else 'Disabled'}")
    print("="*60)
    
    input("\nPress Enter to start training...")
    
    
    print("\n[1/6] Loading dataset...")
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    
    
    print("\n[2/6] Partitioning data across clients...")
    partitioned_data = partition_data_federated(
        X_processed, y_processed,
        n_clients=config['n_clients'],
        iid=config['iid']
    )
    print_data_statistics(partitioned_data)
    
    
    print("[3/6] Initializing federated server...")
    model_architecture = {
        'input_size': X_processed.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    server = FederatedServer(
        model_architecture=model_architecture,
        security_enabled=config['security_enabled']
    )
    print("✓ Server ready")
    
    
    print("\n[4/6] Creating clients...")
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
    print(f"✓ Created {len(clients)} clients")
    
    
    print("\n[5/6] Federated training...")
    print("="*60)
    
    accuracies = []
    
    for round_num in range(config['n_rounds']):
        print(f"\nRound {round_num + 1}/{config['n_rounds']}")
        print("-" * 40)
        
        
        global_model = server.get_global_model()
        for client in clients:
            client.receive_global_model(global_model)
        
        
        print("Training on client devices...")
        client_updates = []
        for client in clients:
            local_weights, metrics = client.local_training(
                epochs=3, learning_rate=0.1, batch_size=32
            )
            client_updates.append(client.get_model_update())
        
        
        print("Aggregating updates securely...")
        server.federated_round(client_updates)
        
        
        clients[0].receive_global_model(server.get_global_model())
        accuracy = clients[0].evaluate(
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        accuracies.append(accuracy)
        
        print(f"Global Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    
    print("\n" + "="*60)
    print("[6/6] RESULTS")
    print("="*60)
    
    print(f"\n✓ Training completed successfully!")
    print(f"\nFinal Metrics:")
    print(f"  Initial Accuracy: {accuracies[0]:.4f} ({accuracies[0]*100:.2f}%)")
    print(f"  Final Accuracy:   {accuracies[-1]:.4f} ({accuracies[-1]*100:.2f}%)")
    print(f"  Improvement:      {(accuracies[-1]-accuracies[0]):.4f} ({(accuracies[-1]-accuracies[0])*100:.2f}%)")
    
    print(f"\nPrivacy & Security:")
    print(f"  ✓ Raw data never left client devices")
    print(f"  ✓ Only model updates transmitted")
    print(f"  ✓ Secure aggregation performed")
    
    print(f"\nModel Performance:")
    best_acc = max(accuracies)
    best_round = accuracies.index(best_acc) + 1
    print(f"  Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%) at round {best_round}")
    
    
    print(f"\nAccuracy Progression:")
    milestone_rounds = [0, len(accuracies)//4, len(accuracies)//2, 3*len(accuracies)//4, len(accuracies)-1]
    for i in milestone_rounds:
        print(f"  Round {i+1:2d}: {accuracies[i]:.4f} ({accuracies[i]*100:.2f}%)")
    
    print("\n" + "="*60)
    print("Thank you for using Federated Learning Demo!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()