# Quick Start Guide

## Get Started in 5 Minutes!

### Step 1: Install Dependencies

```bash
pip install numpy matplotlib scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Demo

```bash
cd federated_learning_project
python demos/main_demo.py
```

That's it! The demo will:
1. ✅ Load the handwritten digits dataset
2. ✅ Partition data across 5 clients (Non-IID)
3. ✅ Run 10 rounds of federated training
4. ✅ Show privacy, security, and bandwidth benefits
5. ✅ Generate visualizations
6. ✅ Save the trained model

### Step 3: View Results

After running, check:
- **Console output**: Training progress and metrics
- **results/training_results.png**: Visualization graphs
- **models/final_model.pkl**: Saved model

## What You'll See

### Console Output Example

```
================================================================================
FEDERATED LEARNING DEMONSTRATION
Addresses: Privacy, Security, Bandwidth, Data Localization, Coordination
================================================================================

[STEP 1] Loading Dataset...
Dataset: Handwritten Digits (0-9)
Total samples: 1797, Features: 64

[STEP 2] Data Localization - Partitioning Data Across Clients...
Distribution: Non-IID
Number of clients: 5
Total training samples: 1437

[STEP 5] Starting Federated Training Rounds...
Key Features:
  ✓ Privacy: Raw data never leaves client devices
  ✓ Security: Secure aggregation with verification
  ✓ Bandwidth: Only model updates transmitted

Round 1/10
  Global Model Accuracy: 0.1056 (10.56%)

...

Round 10/10
  Global Model Accuracy: 0.7694 (76.94%)

FEDERATED LEARNING COMPLETED
Final Global Model Accuracy: 0.7694
Total Bandwidth Used: 3.67 MB
Privacy Benefit:
  Raw data size: 1.01 MB (NEVER transmitted)
```

## Try Different Configurations

### Interactive Mode

```bash
python demos/interactive_demo.py
```

Customize:
- Number of clients (3-10)
- Training rounds (5-20)
- Data distribution (IID/Non-IID)
- Security features

### Comparison Mode

```bash
python demos/comparison_demo.py
```

See side-by-side comparison of:
- Centralized Machine Learning
- Federated Learning
- Privacy & bandwidth implications

## Understanding the Visualizations

The demo generates 4 plots:

1. **Global Model Accuracy** (top-left)
   - Shows model improvement over rounds
   - Should increase as training progresses

2. **Data Distribution** (top-right)
   - Shows how data is split across clients
   - Non-IID shows unequal distribution

3. **Bandwidth Usage** (bottom-left)
   - Constant ~0.37 MB per round
   - Much less than transmitting raw data

4. **Aggregation Time** (bottom-right)
   - Server processing time
   - Very fast (~0.2 milliseconds)

## Key Concepts

### What is Federated Learning?

Instead of:
```
All Clients → Send Data → Central Server → Train Model
```

We do:
```
Central Server → Send Model → Clients → Train Locally
Clients → Send Updates → Server → Aggregate Updates
```

### Why is it Better?

| Aspect | Traditional ML | Federated Learning |
|--------|---------------|-------------------|
| **Privacy** | ❌ Data exposed | ✅ Data stays local |
| **Security** | ❌ Single point of failure | ✅ Distributed |
| **Bandwidth** | ❌ High (all data) | ✅ Low (only updates) |
| **Compliance** | ❌ Data residency issues | ✅ Meets regulations |

## Code Examples

### Basic Usage

```python
from src.fl_server import FederatedServer
from src.fl_client import FederatedClient
from src.data_utils import load_digits_dataset, partition_data_federated

# Load data
X, y = load_digits_dataset()
data = partition_data_federated(X, y, n_clients=5)

# Create server
server = FederatedServer(model_architecture={...})

# Create clients
clients = [FederatedClient(id, X, y) for id, X, y in data]

# Federated training
for round in range(10):
    # Clients train locally
    updates = [client.train() for client in clients]
    # Server aggregates
    server.federated_round(updates)
```

### Add Privacy Protection

```python
# Enable differential privacy
client.add_differential_privacy(epsilon=1.0)
```

### Analyze Bandwidth

```python
from src.data_utils import estimate_model_size

model_size = estimate_model_size(server.get_global_model())
print(f"Model size: {model_size:.4f} MB")
```

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure you're in the project directory
```bash
cd federated_learning_project
python demos/main_demo.py
```

### Issue: Dependencies missing

**Solution**: Install all requirements
```bash
pip install numpy matplotlib scikit-learn
```

### Issue: Low accuracy

**Solution**: This is normal! The demo uses:
- Only 10 rounds (increase for better results)
- Non-IID data (more challenging)
- Simple neural network

Try:
```python
# In main_demo.py, change:
config = {
    'n_rounds': 20,  # More rounds
    'iid': True,     # Easier distribution
}
```

## Next Steps

1. **Understand the code**: Read through `src/fl_server.py` and `src/fl_client.py`

2. **Modify parameters**: Try different configurations in the demos

3. **Add features**: Implement advanced privacy techniques

4. **Use your data**: Replace the dataset with your own

5. **Read documentation**: Check `docs/IMPLEMENTATION.md` for details

## Common Questions

**Q: How does this ensure privacy?**
A: Raw data never leaves client devices. Only model updates (gradients) are sent to the server.

**Q: Is this production-ready?**
A: This is an educational implementation. For production, consider frameworks like TensorFlow Federated or PySyft.

**Q: Can I use my own dataset?**
A: Yes! Replace the `load_digits_dataset()` function with your data loading code.

**Q: How many clients do I need?**
A: The demo works with 3-10 clients. Real systems can have thousands.

**Q: Does this work with deep learning?**
A: Yes! The concepts apply to any neural network. This demo uses a simple network for clarity.

## Resources

- **Project README**: Complete documentation
- **Implementation Guide**: Technical details
- **Demo Scripts**: Working examples
- **Source Code**: Fully commented

## Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the demo scripts in `demos/`
3. Read the source code comments
4. Consult your course materials

---

**Ready to learn about federated learning? Start with `python demos/main_demo.py`!**