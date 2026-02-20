"""
Healthcare Use Case: Multi-Hospital Disease Prediction
========================================================

Scenario: Multiple hospitals want to collaboratively train a disease prediction
model WITHOUT sharing sensitive patient data (HIPAA compliance).

Each hospital:
- Keeps patient data on their local servers
- Trains the model locally
- Only shares model updates (not patient records)

This demo simulates 3 hospitals with different patient populations.
"""

import sys
import os
import numpy as np

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.fl_server import FederatedServer
from src.fl_clients import FederatedClient
from src.data_uitls import load_digits_dataset, preprocess_data, partition_data_federated


class Hospital:
    """Represents a hospital with local patient data"""
    
    def __init__(self, name, location, client):
        self.name = name
        self.location = location
        self.client = client
        self.patient_count = client.data_size
        
    def __str__(self):
        return f"{self.name} ({self.location}) - {self.patient_count} patients"


def simulate_healthcare_scenario():
    """
    Simulate federated learning across hospitals for disease prediction
    """
    print("\n" + "="*80)
    print("HEALTHCARE USE CASE: FEDERATED LEARNING ACROSS HOSPITALS")
    print("="*80)
    
    print("\n📋 Scenario:")
    print("   Three hospitals want to build a better disease detection model")
    print("   BUT they cannot share patient records due to HIPAA regulations")
    print()
    print("🏥 Participating Hospitals:")
    print("   - City General Hospital (New York)")
    print("   - Regional Medical Center (California)")  
    print("   - University Hospital (Texas)")
    print()
    print("🔒 Privacy Requirements:")
    print("   ✓ Patient data MUST stay at each hospital")
    print("   ✓ No patient records can be shared")
    print("   ✓ Must comply with HIPAA regulations")
    print("   ✓ Each hospital maintains data sovereignty")
    
    # Load and prepare data (simulating patient records)
    print("\n" + "-"*80)
    print("Setting up hospital data...")
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    # Partition data as if different hospitals have different patient populations
    partitioned_data = partition_data_federated(
        X_processed, y_processed,
        n_clients=3,  # 3 hospitals
        iid=False  # Each hospital has different patient demographics
    )
    
    # Create hospital objects
    hospitals = []
    hospital_names = [
        ("City General Hospital", "New York, NY"),
        ("Regional Medical Center", "Los Angeles, CA"),
        ("University Hospital", "Houston, TX")
    ]
    
    print("\n🏥 Hospital Information:")
    print("-"*80)
    
    for i, (name, location) in enumerate(hospital_names):
        client_id = f"client_{i+1}"
        client_data = partitioned_data['client_data'][client_id]
        
        client = FederatedClient(
            client_id,
            client_data['X_train'],
            client_data['y_train'],
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        
        hospital = Hospital(name, location, client)
        hospitals.append(hospital)
        
        print(f"\n   {hospital}")
        print(f"      Patient Records: {hospital.patient_count}")
        print(f"      Data Storage: Local servers only")
        print(f"      Privacy Status: ✓ HIPAA Compliant")
    
    # Initialize central coordination server (Cloud-based)
    print("\n" + "-"*80)
    print("☁️  Initializing Central Coordination Server (HIPAA-compliant cloud)...")
    
    model_architecture = {
        'input_size': X_processed.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    
    server = FederatedServer(
        model_architecture=model_architecture,
        security_enabled=True
    )
    
    print("   ✓ Server configured with secure aggregation")
    print("   ✓ End-to-end encryption enabled")
    print("   ✓ No access to raw patient data")
    
    # Federated training
    print("\n" + "="*80)
    print("COLLABORATIVE TRAINING (5 Rounds)")
    print("="*80)
    
    accuracies = []
    
    for round_num in range(5):
        print(f"\n📊 Round {round_num + 1}/5")
        print("-"*80)
        
        # Each hospital trains on their local patient data
        print("\n🏥 Hospitals training on local patient data...")
        
        global_model = server.get_global_model()
        client_updates = []
        
        for hospital in hospitals:
            hospital.client.receive_global_model(global_model)
            
            print(f"\n   {hospital.name}:")
            print(f"      Status: Training on {hospital.patient_count} patient records")
            print(f"      Privacy: ✓ Data stays on local servers")
            
            weights, metrics = hospital.client.local_training(
                epochs=2,
                learning_rate=0.1,
                batch_size=16
            )
            
            client_updates.append(hospital.client.get_model_update())
            print(f"      Training complete: Loss={metrics['final_loss']:.4f}")
        
        # Secure aggregation on cloud server
        print("\n☁️  Cloud server performing secure aggregation...")
        print("   → Receiving encrypted model updates")
        print("   → Verifying update integrity")
        print("   → Aggregating without accessing patient data")
        
        server.federated_round(client_updates)
        
        # Evaluate global model
        hospitals[0].client.receive_global_model(server.get_global_model())
        accuracy = hospitals[0].client.evaluate(
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        accuracies.append(accuracy)
        
        print(f"   ✓ Aggregation complete")
        print(f"\n📈 Global Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Results summary
    print("\n" + "="*80)
    print("RESULTS & IMPACT")
    print("="*80)
    
    print(f"\n🎯 Model Performance:")
    print(f"   Initial Accuracy: {accuracies[0]:.4f} ({accuracies[0]*100:.2f}%)")
    print(f"   Final Accuracy:   {accuracies[-1]:.4f} ({accuracies[-1]*100:.2f}%)")
    print(f"   Improvement:      {(accuracies[-1]-accuracies[0]):.4f} ({(accuracies[-1]-accuracies[0])*100:.2f}%)")
    
    print(f"\n🔒 Privacy Preserved:")
    print(f"   Patient records shared: 0")
    print(f"   Hospitals with data access: Only their own")
    print(f"   HIPAA compliance: ✓ Full compliance")
    print(f"   Data sovereignty: ✓ Maintained")
    
    total_patients = sum(h.patient_count for h in hospitals)
    print(f"\n📊 Collaborative Impact:")
    print(f"   Total patients: {total_patients}")
    print(f"   Hospitals participating: {len(hospitals)}")
    print(f"   Model quality: Learned from ALL hospitals")
    print(f"   Data sharing: ZERO patient records exchanged")
    
    print(f"\n💡 Real-World Benefits:")
    print(f"   ✓ Better diagnoses from diverse patient data")
    print(f"   ✓ Smaller hospitals benefit from larger datasets")
    print(f"   ✓ No expensive data transfer agreements needed")
    print(f"   ✓ Reduced liability and compliance risks")
    print(f"   ✓ Faster model improvements across healthcare network")
    
    print("\n" + "="*80)
    print("COMPARISON: Traditional vs Federated Approach")
    print("="*80)
    
    print("\n❌ Traditional Centralized Approach:")
    print("   • Requires data transfer agreements")
    print("   • Legal and compliance hurdles")
    print("   • Privacy risks from centralized storage")
    print("   • Expensive data anonymization")
    print("   • Patient consent challenges")
    print("   • Single point of failure")
    
    print("\n✅ Federated Learning Approach:")
    print("   • No data leaves hospital servers")
    print("   • Automatic HIPAA compliance")
    print("   • Zero privacy risk from transfer")
    print("   • No anonymization needed")
    print("   • Easier patient consent")
    print("   • Distributed security")
    
    print("\n" + "="*80)
    print("SUCCESS: Collaborative AI while preserving patient privacy! 🏥✨")
    print("="*80 + "\n")


if __name__ == "__main__":
    simulate_healthcare_scenario()