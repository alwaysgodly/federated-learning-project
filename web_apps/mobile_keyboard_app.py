"""
Mobile Keyboard Web Application - Federated Learning Demo
==========================================================
Interactive web interface for mobile keyboard prediction demo
"""

from flask import Flask, render_template, jsonify, request
import sys
import os
import numpy as np

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from src.fl_server import FederatedServer
from src.fl_clients import FederatedClient
from src.data_uitls import load_digits_dataset, preprocess_data, partition_data_federated

app = Flask(__name__)

# Global state
training_state = {
    'initialized': False,
    'current_night': 0,
    'max_nights': 4,
    'devices': [],
    'server': None,
    'accuracies': [],
    'training_active': False
}

def initialize_mobile_system():
    """Initialize the federated mobile keyboard system"""
    global training_state
    
    # Load data
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    # Partition for 6 devices
    partitioned_data = partition_data_federated(
        X_processed, y_processed,
        n_clients=6,
        iid=False
    )
    
    # Device information
    device_info = [
        {"profile": "Tech Professional", "icon": "💼", "color": "#3B82F6"},
        {"profile": "Teenager", "icon": "🎮", "color": "#10B981"},
        {"profile": "Senior Citizen", "icon": "👴", "color": "#F59E0B"},
        {"profile": "Businessperson", "icon": "👔", "color": "#8B5CF6"},
        {"profile": "Student", "icon": "🎓", "color": "#EC4899"},
        {"profile": "Multilingual User", "icon": "🌍", "color": "#06B6D4"}
    ]
    
    # Create clients
    devices = []
    for i, info in enumerate(device_info):
        client_id = f"client_{i+1}"
        client_data = partitioned_data['client_data'][client_id]
        
        client = FederatedClient(
            client_id,
            client_data['X_train'],
            client_data['y_train'],
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        
        devices.append({
            'id': client_id,
            'device_id': f"DEVICE-{i+1:03d}",
            'profile': info['profile'],
            'icon': info['icon'],
            'color': info['color'],
            'typing_samples': client.data_size,
            'client': client,
            'battery': 100,
            'charging': False
        })
    
    # Initialize server
    model_architecture = {
        'input_size': X_processed.shape[1],
        'hidden_size': 64,
        'output_size': 10
    }
    
    server = FederatedServer(
        model_architecture=model_architecture,
        security_enabled=True
    )
    
    training_state.update({
        'initialized': True,
        'devices': devices,
        'server': server,
        'test_data': (partitioned_data['X_test'], partitioned_data['y_test']),
        'current_night': 0,
        'accuracies': [],
        'training_active': False
    })
    
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('mobile.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the system"""
    try:
        initialize_mobile_system()
        
        # Return device info
        devices_info = [
            {
                'id': d['id'],
                'device_id': d['device_id'],
                'profile': d['profile'],
                'icon': d['icon'],
                'color': d['color'],
                'typing_samples': d['typing_samples'],
                'battery': d['battery']
            }
            for d in training_state['devices']
        ]
        
        return jsonify({
            'success': True,
            'devices': devices_info,
            'max_nights': training_state['max_nights']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_night', methods=['POST'])
def train_night():
    """Execute one night of training"""
    try:
        if not training_state['initialized']:
            return jsonify({'success': False, 'error': 'System not initialized'})
        
        if training_state['current_night'] >= training_state['max_nights']:
            return jsonify({'success': False, 'error': 'Training complete'})
        
        training_state['training_active'] = True
        
        # Randomly select 4 devices to update (some charging, some not)
        all_devices = training_state['devices']
        participating_devices = np.random.choice(all_devices, size=4, replace=False)
        
        # Get global model
        global_model = training_state['server'].get_global_model()
        
        # Each participating device trains
        client_updates = []
        device_results = []
        
        for device in all_devices:
            if device in participating_devices:
                # Device is charging and updating
                device['charging'] = True
                client = device['client']
                client.receive_global_model(global_model)
                
                weights, metrics = client.local_training(
                    epochs=2,
                    learning_rate=0.1,
                    batch_size=16
                )
                
                client_updates.append(client.get_model_update())
                
                device_results.append({
                    'id': device['id'],
                    'device_id': device['device_id'],
                    'profile': device['profile'],
                    'status': 'updated',
                    'loss': float(metrics['final_loss'])
                })
            else:
                # Device not charging or opted out
                device['charging'] = False
                device['battery'] = np.random.randint(20, 60)
                device_results.append({
                    'id': device['id'],
                    'device_id': device['device_id'],
                    'profile': device['profile'],
                    'status': 'skipped',
                    'reason': np.random.choice(['Battery low', 'Not charging', 'User opted out'])
                })
        
        # Server aggregates
        training_state['server'].federated_round(client_updates)
        
        # Evaluate global model
        test_client = training_state['devices'][0]['client']
        test_client.receive_global_model(training_state['server'].get_global_model())
        X_test, y_test = training_state['test_data']
        accuracy = test_client.evaluate(X_test, y_test)
        
        training_state['accuracies'].append(float(accuracy))
        training_state['current_night'] += 1
        training_state['training_active'] = False
        
        return jsonify({
            'success': True,
            'night': training_state['current_night'],
            'accuracy': float(accuracy),
            'device_results': device_results,
            'all_accuracies': training_state['accuracies']
        })
        
    except Exception as e:
        training_state['training_active'] = False
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current training status"""
    return jsonify({
        'initialized': training_state['initialized'],
        'current_night': training_state['current_night'],
        'max_nights': training_state['max_nights'],
        'accuracies': training_state['accuracies'],
        'training_active': training_state['training_active'],
        'completed': training_state['current_night'] >= training_state['max_nights']
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the training"""
    global training_state
    training_state = {
        'initialized': False,
        'current_night': 0,
        'max_nights': 4,
        'devices': [],
        'server': None,
        'accuracies': [],
        'training_active': False
    }
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("📱 MOBILE KEYBOARD FEDERATED LEARNING WEB APP")
    print("="*80)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5001")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5001)
