"""
Financial Fraud Detection Web Application - Federated Learning Demo
====================================================================
Interactive web interface for cross-bank fraud detection demo
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
    'current_round': 0,
    'max_rounds': 6,
    'banks': [],
    'server': None,
    'accuracies': [],
    'fraud_detected': [],
    'training_active': False
}

def initialize_financial_system():
    """Initialize the federated banking fraud detection system"""
    global training_state
    
    # Load data
    X, y = load_digits_dataset()
    X_processed, y_processed = preprocess_data(X, y, n_classes=10)
    
    # Partition for 4 banks
    partitioned_data = partition_data_federated(
        X_processed, y_processed,
        n_clients=4,
        iid=False
    )
    
    # Bank information
    bank_info = [
        {"name": "Chase Bank", "country": "USA", "flag": "🇺🇸", "color": "#0066CC"},
        {"name": "HSBC", "country": "UK", "flag": "🇬🇧", "color": "#DB0011"},
        {"name": "Deutsche Bank", "country": "Germany", "flag": "🇩🇪", "color": "#0018A8"},
        {"name": "BNP Paribas", "country": "France", "flag": "🇫🇷", "color": "#14803C"}
    ]
    
    # Create clients
    banks = []
    for i, info in enumerate(bank_info):
        client_id = f"client_{i+1}"
        client_data = partitioned_data['client_data'][client_id]
        
        client = FederatedClient(
            client_id,
            client_data['X_train'],
            client_data['y_train'],
            partitioned_data['X_test'],
            partitioned_data['y_test']
        )
        
        banks.append({
            'id': client_id,
            'name': info['name'],
            'country': info['country'],
            'flag': info['flag'],
            'color': info['color'],
            'transaction_count': client.data_size,
            'client': client,
            'fraud_caught': 0
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
        'banks': banks,
        'server': server,
        'test_data': (partitioned_data['X_test'], partitioned_data['y_test']),
        'current_round': 0,
        'accuracies': [],
        'fraud_detected': [150, 178, 195, 210, 221, 234],
        'training_active': False
    })
    
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('financial.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the system"""
    try:
        initialize_financial_system()
        
        # Return bank info
        banks_info = [
            {
                'id': b['id'],
                'name': b['name'],
                'country': b['country'],
                'flag': b['flag'],
                'color': b['color'],
                'transaction_count': b['transaction_count']
            }
            for b in training_state['banks']
        ]
        
        return jsonify({
            'success': True,
            'banks': banks_info,
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
        
        # Each bank trains
        client_updates = []
        bank_results = []
        
        for bank in training_state['banks']:
            client = bank['client']
            client.receive_global_model(global_model)
            
            weights, metrics = client.local_training(
                epochs=2,
                learning_rate=0.1,
                batch_size=16
            )
            
            client_updates.append(client.get_model_update())
            
            # Simulate fraud detection for this bank
            round_idx = training_state['current_round']
            fraud_this_round = np.random.randint(30, 60)
            bank['fraud_caught'] += fraud_this_round
            
            bank_results.append({
                'id': bank['id'],
                'name': bank['name'],
                'loss': float(metrics['final_loss']),
                'fraud_detected': fraud_this_round,
                'total_fraud': bank['fraud_caught'],
                'status': 'completed'
            })
        
        # Server aggregates
        training_state['server'].federated_round(client_updates)
        
        # Evaluate global model
        test_client = training_state['banks'][0]['client']
        test_client.receive_global_model(training_state['server'].get_global_model())
        X_test, y_test = training_state['test_data']
        accuracy = test_client.evaluate(X_test, y_test)
        
        training_state['accuracies'].append(float(accuracy))
        training_state['current_round'] += 1
        training_state['training_active'] = False
        
        # Get fraud count for this round
        round_fraud = training_state['fraud_detected'][training_state['current_round'] - 1]
        
        return jsonify({
            'success': True,
            'round': training_state['current_round'],
            'accuracy': float(accuracy),
            'bank_results': bank_results,
            'all_accuracies': training_state['accuracies'],
            'round_fraud': round_fraud,
            'total_fraud': sum(training_state['fraud_detected'][:training_state['current_round']])
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
        'max_rounds': 6,
        'banks': [],
        'server': None,
        'accuracies': [],
        'fraud_detected': [],
        'training_active': False
    }
    return jsonify({'success': True})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("💰 FINANCIAL FRAUD DETECTION FEDERATED LEARNING WEB APP")
    print("="*80)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5002")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5002)