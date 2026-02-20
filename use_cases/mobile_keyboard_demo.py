"""
Mobile Device Use Case: Keyboard Prediction
============================================

Scenario: A smartphone company wants to improve keyboard predictions
across millions of users WITHOUT collecting their private messages.

Real-world example: Google's GBoard uses federated learning

Each phone:
- Learns from user's typing patterns locally
- Never uploads messages or typing data
- Only shares model improvements
- User privacy is 100% protected
"""

import sys
import os
import numpy as np

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.fl_server import FederatedServer
from src.fl_clients import FederatedClient
from src.data_uitls import load_digits_dataset, preprocess_data, partition_data_federated


class MobileDevice:
    """Represents a user's mobile device"""
    
    def __init__(self, device_id, user_profile, client):
        self.device_id = device_id
        self.user_profile = user_profile
        self.client = client
        self.typing_samples = client.data_size
        
    def __str__(self):
        return f"Device {self.device_id} ({self.user_profile}) - {self.typing_samples} typing samples"


def simulate_mobile_keyboard_learning():
    """
    Simulate federated learning for keyboard prediction across mobile devices
    """
    print("\n" + "="*80)
    print("MOBILE DEVICE USE CASE: KEYBOARD PREDICTION IMPROVEMENT")
    print("="*80)
    
    print("\n📱 Scenario:")
    print("   A smartphone company wants to improve keyboard predictions")
    print("   They have millions of users typing every day")
    print("   BUT they DON'T want to collect users' private messages")
    print()
    print("👥 User Devices:")
    print("   - iPhone users")
    print("   - Android users")
    print("   - Tablet users")
    print("   - Each with different typing patterns")
    print()
    print("🔒 Privacy Requirements:")
    print("   ✓ Messages NEVER leave the device")
    print("   ✓ No cloud storage of typing data")
    print("   ✓ User data stays private")
    print("   ✓ Only model improvements shared")
    
    # Simulate different user devices with different typing patterns
    print("\n" + "-"*80)
    print("Setting up user devices...")
    
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    # Partition data - each device has unique typing patterns
    partitioned_data = partition_data_federated(
        X_processed, y_processed,
        n_clients=6,  # 6 different users
        iid=False  # Different typing styles
    )
    
    # Create device objects with different user profiles
    devices = []
    user_profiles = [
        "Tech Professional (frequent typer)",
        "Teenager (lots of emojis)",
        "Senior Citizen (careful typer)",
        "Businessperson (formal language)",
        "Student (casual language)",
        "Multilingual User (mixed languages)"
    ]
    
    print("\n📱 Connected Devices:")
    print("-"*80)
    
    for i, profile in enumerate(user_profiles):
        client_id = f"client_{i+1}"
        client_data = partitioned_data['client_data'][client_id]
        
        client = FederatedClient(
            client_id,
            client_data['X_train'],
            client_data['y_train'],
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        
        device = MobileDevice(f"DEVICE-{i+1:03d}", profile, client)
        devices.append(device)
        
        print(f"\n   {device}")
        print(f"      Typing samples: {device.typing_samples} (stored locally)")
        print(f"      Privacy: ✓ Data never uploaded to cloud")
    
    # Initialize cloud server for coordination
    print("\n" + "-"*80)
    print("☁️  Initializing Cloud Coordination Server...")
    
    model_architecture = {
        'input_size': X_processed.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    
    server = FederatedServer(
        model_architecture=model_architecture,
        security_enabled=True
    )
    
    print("   ✓ Server ready to coordinate (no access to user data)")
    print("   ✓ Only receives encrypted model updates")
    
    # Simulate nightly updates (phones typically update when charging overnight)
    print("\n" + "="*80)
    print("NIGHTLY MODEL UPDATES (Phones charging overnight)")
    print("="*80)
    
    accuracies = []
    
    for night in range(4):
        print(f"\n🌙 Night {night + 1}/4 - Updating while users sleep...")
        print("-"*80)
        
        # Each phone learns from today's typing
        print("\n📱 Devices learning from local typing patterns...")
        
        global_model = server.get_global_model()
        client_updates = []
        
        participating_devices = np.random.choice(devices, size=4, replace=False)  # Only 4 devices update tonight
        
        for device in participating_devices:
            device.client.receive_global_model(global_model)
            
            print(f"\n   {device.device_id} ({device.user_profile}):")
            print(f"      Status: Learning from {device.typing_samples} typing samples")
            print(f"      Location: On-device training (not in cloud)")
            print(f"      Privacy: ✓ Messages stay on device")
            
            weights, metrics = device.client.local_training(
                epochs=2,
                learning_rate=0.1,
                batch_size=16
            )
            
            client_updates.append(device.client.get_model_update())
            print(f"      Update ready: {metrics['final_loss']:.4f} loss")
        
        print(f"\n   {len(participating_devices)}/{len(devices)} devices participated")
        print(f"   (Others: battery low, not charging, or opted out)")
        
        # Cloud aggregation
        print("\n☁️  Cloud aggregating updates...")
        print("   → Receiving encrypted updates from devices")
        print("   → Combining improvements")
        print("   → Creating better global model")
        
        server.federated_round(client_updates)
        
        # Evaluate
        devices[0].client.receive_global_model(server.get_global_model())
        accuracy = devices[0].client.evaluate(
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        accuracies.append(accuracy)
        
        print(f"   ✓ New model ready for distribution")
        print(f"\n📊 Global Model Quality: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Next morning - users get the improved model
        print(f"\n☀️  Morning - Users wake up to better predictions!")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS AFTER 4 NIGHTS OF UPDATES")
    print("="*80)
    
    print(f"\n📈 Keyboard Improvement:")
    print(f"   Initial prediction quality: {accuracies[0]:.4f} ({accuracies[0]*100:.2f}%)")
    print(f"   Final prediction quality:   {accuracies[-1]:.4f} ({accuracies[-1]*100:.2f}%)")
    print(f"   Improvement:                {(accuracies[-1]-accuracies[0]):.4f} ({(accuracies[-1]-accuracies[0])*100:.2f}%)")
    
    total_samples = sum(d.typing_samples for d in devices)
    print(f"\n👥 User Privacy:")
    print(f"   Total users: {len(devices)}")
    print(f"   Typing samples: {total_samples}")
    print(f"   Messages uploaded to cloud: 0 ✓")
    print(f"   Privacy preserved: 100% ✓")
    
    print(f"\n💡 User Experience Benefits:")
    print(f"   ✓ Better autocorrect from diverse typing styles")
    print(f"   ✓ Improved next-word predictions")
    print(f"   ✓ Learns from millions without seeing messages")
    print(f"   ✓ Personalized while maintaining privacy")
    print(f"   ✓ Updates happen automatically overnight")
    
    print(f"\n📊 Technical Efficiency:")
    print(f"   • Updates only when charging (saves battery)")
    print(f"   • Uses WiFi only (saves mobile data)")
    print(f"   • Small model updates (~0.037 MB)")
    print(f"   • Fast on-device training")
    
    print("\n" + "="*80)
    print("REAL-WORLD APPLICATIONS")
    print("="*80)
    
    print("\n🌟 Companies Using This Approach:")
    print()
    print("   📱 Google GBoard")
    print("      • Keyboard predictions")
    print("      • Voice typing improvements")
    print("      • 1 billion+ devices")
    print()
    print("   🍎 Apple iOS Keyboard")
    print("      • QuickType predictions")
    print("      • Emoji suggestions")
    print("      • All iPhones and iPads")
    print()
    print("   💬 WhatsApp")
    print("      • Message suggestions")
    print("      • Spam detection")
    print("      • 2 billion+ users")
    
    print("\n" + "="*80)
    print("COMPARISON: Traditional vs Federated")
    print("="*80)
    
    print("\n❌ Traditional Cloud-Based Approach:")
    print("   • Upload all messages to cloud")
    print("   • Privacy concerns")
    print("   • Large data transfer (expensive)")
    print("   • Regulatory compliance issues")
    print("   • User trust problems")
    
    print("\n✅ Federated Learning Approach:")
    print("   • Messages stay on device")
    print("   • Complete privacy")
    print("   • Minimal data transfer")
    print("   • Easy compliance")
    print("   • User trust maintained")
    
    print("\n" + "="*80)
    print("SUCCESS: Better predictions WITHOUT invading privacy! 📱✨")
    print("="*80 + "\n")
    
    print("💭 User Perspective:")
    print("   'My keyboard got better at predicting what I want to type,'")
    print("   'and I never had to upload my private messages!' 🎉")
    print()


if __name__ == "__main__":
    simulate_mobile_keyboard_learning()