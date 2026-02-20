"""
Healthcare Web Application - Federated Learning Demo
=====================================================
Interactive web interface for hospital collaboration demo
"""

from flask import Flask, render_template, jsonify, request
import sys
import os
import numpy as np
import json
from datetime import datetime

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
    'current_round': 0,
    'max_rounds': 5,
    'hospitals': [],
    'server': None,
    'accuracies': [],
    'training_active': False
}

def initialize_healthcare_system():
    """Initialize the federated healthcare system"""
    global training_state
    
    # Load data
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    # Partition for 3 hospitals
    partitioned_data = partition_data_federated(
        X_processed, y_processed,
        n_clients=3,
        iid=False
    )
    
    # Hospital information
    hospital_info = [
        {"name": "City General Hospital", "location": "New York, NY", "color": "#3B82F6"},
        {"name": "Regional Medical Center", "location": "Los Angeles, CA", "color": "#10B981"},
        {"name": "University Hospital", "location": "Houston, TX", "color": "#F59E0B"}
    ]
    
    # Create clients
    hospitals = []
    for i, info in enumerate(hospital_info):
        client_id = f"client_{i+1}"
        client_data = partitioned_data['client_data'][client_id]
        
        client = FederatedClient(
            client_id,
            client_data['X_train'],
            client_data['y_train'],
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        
        hospitals.append({
            'id': client_id,
            'name': info['name'],
            'location': info['location'],
            'color': info['color'],
            'patient_count': client.data_size,
            'client': client
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
        'hospitals': hospitals,
        'server': server,
        'test_data': (partitioned_data['X_test'], partitioned_data['y_test']),
        'current_round': 0,
        'accuracies': [],
        'training_active': False
    })
    
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('healthcare.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the system"""
    try:
        initialize_healthcare_system()
        
        # Return hospital info
        hospitals_info = [
            {
                'id': h['id'],
                'name': h['name'],
                'location': h['location'],
                'color': h['color'],
                'patient_count': h['patient_count']
            }
            for h in training_state['hospitals']
        ]
        
        return jsonify({
            'success': True,
            'hospitals': hospitals_info,
            'max_rounds': training_state['max_rounds']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_round', methods=['POST'])
def train_round():
    """Execute one training round"""
    try:
        if not training_state['initialized']:
            return jsonify({'success': False, 'error': 'System not initialized'})
        
        if training_state['current_round'] >= training_state['max_rounds']:
            return jsonify({'success': False, 'error': 'Training complete'})
        
        training_state['training_active'] = True
        
        # Get global model
        global_model = training_state['server'].get_global_model()
        
        # Each hospital trains
        client_updates = []
        hospital_results = []
        
        for hospital in training_state['hospitals']:
            client = hospital['client']
            client.receive_global_model(global_model)
            
            weights, metrics = client.local_training(
                epochs=2,
                learning_rate=0.1,
                batch_size=16
            )
            
            client_updates.append(client.get_model_update())
            
            hospital_results.append({
                'id': hospital['id'],
                'name': hospital['name'],
                'loss': float(metrics['final_loss']),
                'status': 'completed'
            })
        
        # Server aggregates
        training_state['server'].federated_round(client_updates)
        
        # Evaluate global model
        test_client = training_state['hospitals'][0]['client']
        test_client.receive_global_model(training_state['server'].get_global_model())
        X_test, y_test = training_state['test_data']
        accuracy = test_client.evaluate(X_test, y_test)
        
        training_state['accuracies'].append(float(accuracy))
        training_state['current_round'] += 1
        training_state['training_active'] = False
        
        return jsonify({
            'success': True,
            'round': training_state['current_round'],
            'accuracy': float(accuracy),
            'hospital_results': hospital_results,
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
        'current_round': training_state['current_round'],
        'max_rounds': training_state['max_rounds'],
        'accuracies': training_state['accuracies'],
        'training_active': training_state['training_active'],
        'completed': training_state['current_round'] >= training_state['max_rounds']
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the training"""
    global training_state
    training_state = {
        'initialized': False,
        'current_round': 0,
        'max_rounds': 5,
        'hospitals': [],
        'server': None,
        'accuracies': [],
        'training_active': False
    }
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏥 HEALTHCARE FEDERATED LEARNING WEB APP")
    print("="*80)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000)