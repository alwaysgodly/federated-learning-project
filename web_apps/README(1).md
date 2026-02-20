# Federated Learning Web Applications

Interactive web demonstrations of federated learning use cases. Each app provides a visual, hands-on experience of privacy-preserving machine learning.

## 🌐 Available Web Apps

### 1. Healthcare Demo 🏥
**File:** `healthcare_app.py`  
**Port:** 5000

**Features:**
- Interactive 3-hospital network visualization
- Real-time training progress
- Privacy guarantees displayed
- Accuracy charts and metrics
- Training logs

**URL:** http://localhost:5000

---

### 2. Mobile Keyboard Demo 📱
**File:** `mobile_keyboard_app.py`  
**Port:** 5001

**Features:**
- 6 mobile devices with different user profiles
- Day/night cycle visualization
- Battery and charging indicators
- Real company examples (Google, Apple, WhatsApp)
- Privacy-preserving keyboard improvements

**URL:** http://localhost:5001

---

### 3. Financial Fraud Detection Demo 💰
**File:** `financial_fraud_app.py`  
**Port:** 5002

**Features:**
- 4 international banks (USA, UK, Germany, France)
- Real-time fraud detection metrics
- Money saved calculator
- Cross-border collaboration
- Regulatory compliance indicators

**URL:** http://localhost:5002

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to web_apps folder
cd web_apps

# Install dependencies
pip install -r requirements.txt
```

### Running the Apps

#### Option 1: Easy Launch (Windows)
```bash
launch.bat
# Then select 1, 2, or 3
```

#### Option 2: Run Individually

**Healthcare:**
```bash
python healthcare_app.py
# Open: http://localhost:5000
```

**Mobile Keyboard:**
```bash
python mobile_keyboard_app.py
# Open: http://localhost:5001
```

**Financial Fraud:**
```bash
python financial_fraud_app.py
# Open: http://localhost:5002
```

---

## 📱 App Comparison

| App | Industry | Clients | Training | Privacy Feature |
|-----|----------|---------|----------|-----------------|
| **Healthcare** | Medical | 3 hospitals | 5 rounds | HIPAA compliance |
| **Mobile** | Tech | 6 devices | 4 nights | Message privacy |
| **Financial** | Banking | 4 banks | 6 rounds | Transaction privacy |

## 📱 How to Use

### Healthcare App

1. **Initialize System**
   - Click "Initialize System" button
   - 3 hospitals will appear with their patient counts
   - System is ready for training

2. **Train**
   - Click "Train One Round"
   - Watch each hospital train on local data
   - See global model accuracy improve
   - Privacy maintained - no data shared!

3. **Monitor Progress**
   - View real-time accuracy chart
   - Check training logs
   - See improvement metrics
   - Progress bar shows completion

4. **Reset**
   - Click "Reset" to start over
   - Try different scenarios

---

## 🎨 Features

### Visual Elements

- **Hospital Cards**: Show each hospital's status
  - Patient count
  - Location
  - Training status
  - Real-time updates

- **Accuracy Chart**: Live graph showing:
  - Model improvement over rounds
  - Accuracy percentage
  - Training progression

- **Privacy Banner**: Highlights:
  - HIPAA compliance
  - Zero data sharing
  - Secure aggregation

- **Training Log**: Real-time updates:
  - System events
  - Training progress
  - Privacy confirmations

### Interactive Controls

- **Initialize**: Sets up the federated system
- **Train Round**: Executes one training round
- **Reset**: Clears all data and starts fresh

---

## 💻 Technical Details

### Architecture

```
Browser (Frontend)
    ↓ HTTP/JSON
Flask Server (Backend)
    ↓
Federated Learning System
    ├── Server (Aggregation)
    └── Clients (Hospitals)
```

### API Endpoints

#### POST `/api/initialize`
Initialize the federated system with hospitals

**Response:**
```json
{
  "success": true,
  "hospitals": [...],
  "max_rounds": 5
}
```

#### POST `/api/train_round`
Execute one training round

**Response:**
```json
{
  "success": true,
  "round": 1,
  "accuracy": 0.234,
  "hospital_results": [...],
  "all_accuracies": [...]
}
```

#### GET `/api/status`
Get current training status

**Response:**
```json
{
  "initialized": true,
  "current_round": 2,
  "accuracies": [0.15, 0.23],
  "completed": false
}
```

#### POST `/api/reset`
Reset the training system

---

## 🎯 Use Cases

### Educational
- Demonstrate federated learning concepts
- Show privacy preservation in action
- Visualize distributed training

### Presentations
- Live demos during talks
- Interactive workshops
- Client demonstrations

### Development
- Test federated learning algorithms
- Prototype new features
- Debug training issues

---

## 🔧 Customization

### Changing Number of Rounds

Edit `healthcare_app.py`:
```python
training_state = {
    'max_rounds': 10,  # Change from 5 to 10
    ...
}
```

### Adjusting Hospital Count

In `initialize_healthcare_system()`:
```python
partitioned_data = partition_data_federated(
    X_processed, y_processed,
    n_clients=5,  # Change from 3 to 5
    iid=False
)
```

### Styling

Edit `templates/healthcare.html`:
- Change colors in CSS
- Modify layout
- Add new visualizations

---

## 📊 Screenshots

### Main Interface
- Hospital cards with real-time status
- Training controls
- Accuracy metrics

### Training in Progress
- Animated hospital status
- Live accuracy updates
- Real-time logs

### Results
- Accuracy improvement chart
- Privacy guarantees
- Performance metrics

---

## 🚢 Deployment

### Local Development
```bash
python healthcare_app.py
# Visit: http://localhost:5000
```

### Production (Using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 healthcare_app:app
```

### Production (Using Waitress - Windows)
```bash
pip install waitress
waitress-serve --port=8000 healthcare_app:app
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "healthcare_app.py"]
```

Build and run:
```bash
docker build -t federated-healthcare .
docker run -p 5000:5000 federated-healthcare
```

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Change port in healthcare_app.py
app.run(debug=True, port=5001)  # Use different port
```

### Module Not Found

```bash
# Make sure you're in web_apps folder
cd web_apps

# Install dependencies
pip install -r requirements.txt
```

### Template Not Found

Make sure folder structure is:
```
web_apps/
├── healthcare_app.py
├── templates/
│   └── healthcare.html
└── requirements.txt
```

### Slow Training

Training happens on CPU. For faster results:
- Reduce dataset size
- Decrease epochs per round
- Use fewer clients

---

## 🔮 Future Enhancements

### Planned Features

- [ ] Mobile keyboard app
- [ ] Financial fraud detection app
- [ ] Real-time collaboration viewer
- [ ] Export training results
- [ ] Dark mode
- [ ] Multi-user support
- [ ] Save/load training state
- [ ] WebSocket for real-time updates
- [ ] 3D visualizations
- [ ] Mobile-responsive design

---

## 📖 Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Canvas API (custom drawing)
- **ML**: NumPy, scikit-learn
- **FL**: Custom implementation

---

## 🤝 Contributing

Want to add a new web app?

1. Create `your_app.py` in `web_apps/`
2. Create `templates/your_template.html`
3. Follow the healthcare app structure
4. Update this README

---

## 📝 License

Same as main project - MIT License

---

## 🎓 Learning Resources

### Understanding the Code

**Backend (healthcare_app.py):**
- Flask routes handle HTTP requests
- Training state stored in memory
- Federated learning logic in separate modules

**Frontend (healthcare.html):**
- Vanilla JavaScript (no frameworks)
- Fetch API for backend communication
- Canvas for custom charts
- CSS Grid for responsive layout

### Key Concepts

1. **State Management**: Global `training_state` dict
2. **Async Operations**: JavaScript async/await
3. **Real-time Updates**: Polling with fetch
4. **Visualization**: HTML Canvas API

---

## 💡 Tips

### For Presenters
- Initialize before presenting
- Explain each step as it happens
- Pause to highlight privacy features
- Show the training log

### For Developers
- Use browser DevTools for debugging
- Check Flask console for errors
- Monitor network tab for API calls
- Test on different browsers

### For Students
- Read the code comments
- Modify parameters to see effects
- Try breaking things to learn
- Build your own version

---

## 🌟 Showcase

Share your web app:
- Take screenshots
- Record a demo video
- Write a blog post
- Present at meetups

---

## 📞 Support

Issues with web apps?
1. Check the troubleshooting section
2. Review Flask documentation
3. Open GitHub issue
4. Check main project documentation

---

<div align="center">

**Making Federated Learning Visual and Interactive** 🚀✨

[Back to Main Project](../README.md)

</div>