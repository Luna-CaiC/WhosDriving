# Taxi Driver Identification using Deep Learning

A machine learning system that identifies individual taxi drivers based on their driving behavior patterns extracted from GPS trajectory data. This project demonstrates the application of neural networks for behavioral biometrics and driver profiling.

## 📋 Project Overview

This project implements a driver identification system that analyzes taxi GPS trajectories to classify and identify individual drivers. By extracting behavioral features from location data, timestamps, and taxi status information, the system learns unique driving patterns that distinguish one driver from another.

**Key Capabilities:**
- Multi-driver classification using trajectory-based features
- Automated feature extraction from raw GPS data
- Deep neural network with dropout regularization
- Comprehensive data validation and preprocessing
- Model persistence for deployment

## 🎯 Problem Statement

Given GPS trajectory data from multiple taxi drivers, the goal is to build a classification model that can accurately identify which driver is operating the vehicle based solely on their driving patterns. This has applications in:
- Driver authentication and security
- Fleet management and monitoring
- Behavioral analysis for insurance
- Fraud detection

## 📊 Dataset

The dataset consists of GPS trajectory data from **5 taxi drivers** collected over several months (July - December 2016) in Hong Kong.

**Data Structure:**
- **Format:** Daily CSV files (179 files total)
- **Time Period:** July 1, 2016 - December 26, 2016
- **Location:** Hong Kong area (Longitude: 113-115°, Latitude: 22-23°)

**Features per Record:**
| Column | Description | Type |
|--------|-------------|------|
| `plate` | Driver ID (0-4) | Integer |
| `longitude` | GPS longitude coordinate | Float |
| `latitude` | GPS latitude coordinate | Float |
| `time` | Timestamp of record | DateTime |
| `status` | Taxi status (0=vacant, 1=occupied) | Binary |

**Dataset Statistics:**
- Total data size: ~200MB
- Drivers: 5 unique drivers (labeled 0-4)
- Records per day: ~10,000-15,000 GPS points
- Temporal coverage: 6 months of continuous data

## 🔧 Technical Architecture

### Feature Engineering

The system extracts **20 comprehensive features** from each driver's trajectory:

**1. Trajectory Characteristics (1 feature)**
- Total trajectory length (number of GPS points)

**2. Spatial Features (6 features)**
- Longitude/Latitude mean (geographic center)
- Longitude/Latitude standard deviation (area coverage)
- Longitude/Latitude range (travel extent)

**3. Behavioral Features (2 features)**
- Occupied ratio (percentage of time with passengers)
- Status changes (frequency of pickup/dropoff events)

**4. Temporal Features (5 features)**
- Trip duration (total time in hours)
- Hour mean/std (time-of-day patterns)
- Peak morning ratio (7-9 AM activity)
- Peak evening ratio (5-7 PM activity)

**5. Movement Features (4 features)**
- Total distance traveled
- Average distance per segment
- Average speed
- Maximum speed

**6. Stopping Patterns (2 features)**
- Number of stop points
- Stop ratio (proportion of stationary time)

### Model Architecture

**Neural Network Design:**
```
Input Layer (20 features)
    ↓
Dense Layer (64 neurons, ReLU activation)
    ↓
Dropout (30% - prevents overfitting)
    ↓
Dense Layer (32 neurons, ReLU activation)
    ↓
Dropout (20% - regularization)
    ↓
Output Layer (5 neurons, Softmax activation)
```

**Training Configuration:**
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Batch Size:** 16 (adaptive based on dataset size)
- **Epochs:** Up to 50 with early stopping
- **Early Stopping:** Patience=15 epochs on validation accuracy
- **Data Split:** 70% training, 30% validation
- **Scaling:** StandardScaler normalization

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
TensorFlow/Keras 2.x or 3.x
pandas
numpy
scikit-learn
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn keras tensorflow
```

### Project Structure
```
Project3/
├── data_5drivers/          # GPS trajectory data (179 CSV files)
├── evaluation.py           # Feature extraction and prediction functions
├── simple_train.py         # Model training script
├── validate_data.py        # Data validation and quality checks
├── test_model.py          # Model testing utilities
├── taxi_driver_model.pkl  # Trained model (generated after training)
└── README.md              # This file
```

## 💻 Usage

### 1. Validate Data
Before training, verify data integrity:
```bash
python validate_data.py
```

**Output:**
- File validation results
- Data statistics and distributions
- Feature extraction test
- Recommendations for training

### 2. Train Model
Train the driver identification model:
```bash
python simple_train.py
```

**Training Process:**
1. Loads all CSV files from `data_5drivers/` directory
2. Extracts features from each driver's trajectories
3. Splits data into training/validation sets
4. Trains neural network with early stopping
5. Saves trained model to `taxi_driver_model.pkl`

**Expected Output:**
```
Total dataset shape: (X, 20)
Number of trajectories: X
Unique drivers: [0, 1, 2, 3, 4]
Training Accuracy: 0.XXXX
Validation Accuracy: 0.XXXX
```

### 3. Test Model
Test the trained model:
```bash
python test_model.py
```

Tests the model with sample trajectories or `test.pkl` if available.

### 4. Use for Prediction

```python
import pickle
from evaluation import process_data, run

# Load trained model
model = pickle.load(open('taxi_driver_model.pkl', 'rb'))

# Sample trajectory data
trajectory = [
    [114.10437, 22.573433, '2016-07-06 00:07:30', 1],
    [114.10500, 22.573500, '2016-07-06 00:08:00', 1],
    [114.10600, 22.573600, '2016-07-06 00:08:45', 0],
    # ... more GPS points
]

# Extract features
features = process_data(trajectory)

# Predict driver
driver_id = run(features, model)
print(f"Predicted Driver: {driver_id}")
```

## 📈 Performance

### Model Evaluation Metrics

**Performance Targets:**
| Validation Accuracy | Score | Status |
|---------------------|-------|--------|
| ≥ 60% | 50/50 | Excellent ✅ |
| 55-59% | 45/50 | Very Good 👍 |
| 50-54% | 40/50 | Good ✓ |
| 45-49% | 35/50 | Acceptable ⚠️ |
| 40-44% | 30/50 | Minimum ⚠️ |
| < 40% | < 30/50 | Needs Improvement ❌ |

**Expected Results:**
- Training accuracy: 70-90%
- Validation accuracy: 50-70%
- Inference time: < 10ms per trajectory

### Key Performance Factors

1. **Data Quality:** Clean GPS data with consistent sampling
2. **Feature Richness:** 20 diverse features capture driving behavior
3. **Regularization:** Dropout layers prevent overfitting
4. **Early Stopping:** Prevents overtraining on small datasets
5. **Stratified Split:** Ensures balanced class distribution

## 🔍 Key Features

### Data Processing
- ✅ Handles multiple CSV files automatically
- ✅ Robust time parsing with multiple format support
- ✅ Geographic coordinate validation
- ✅ Missing value handling
- ✅ Trajectory aggregation by driver

### Model Features
- ✅ Adaptive architecture based on dataset size
- ✅ Automatic feature scaling (StandardScaler)
- ✅ Label mapping for non-sequential driver IDs
- ✅ Early stopping to prevent overfitting
- ✅ Model persistence with metadata

### Validation & Testing
- ✅ Comprehensive data validation
- ✅ Feature extraction testing
- ✅ Sample prediction capabilities
- ✅ Performance interpretation

## 🛠️ Technical Implementation

### Core Functions

**`process_data(trajectory)`** (evaluation.py)
- Extracts 20 features from raw GPS trajectory
- Handles edge cases (single points, invalid times)
- Returns numpy array of normalized features

**`run(data, model)`** (evaluation.py)
- Performs driver prediction on processed features
- Handles both standalone models and pickled model dictionaries
- Returns integer driver ID (0-4)

**`train_simple_model()`** (simple_train.py)
- Complete training pipeline
- Automatic data loading and preprocessing
- Model architecture selection
- Training with validation
- Model persistence

**`validate_csv_file(file_path)`** (validate_data.py)
- Validates data format and structure
- Checks coordinate ranges
- Verifies time formats
- Analyzes data distributions

## 📝 Model Artifacts

The trained model is saved as `taxi_driver_model.pkl` containing:

```python
{
    'model': keras.Model,              # Trained neural network
    'scaler': StandardScaler,          # Feature scaler
    'label_mapping': dict,             # Original label → index
    'reverse_mapping': dict,           # Index → original label
    'num_features': int,               # Number of features (20)
    'num_classes': int,                # Number of drivers (5)
    'training_accuracy': float,        # Final training accuracy
    'validation_accuracy': float       # Final validation accuracy
}
```

## 🔬 Methodology

### Workflow
1. **Data Collection:** GPS trajectories from 5 taxi drivers
2. **Data Validation:** Quality checks and format verification
3. **Feature Engineering:** Extract 20 behavioral features
4. **Data Preprocessing:** Scaling and train/test split
5. **Model Training:** Neural network with dropout regularization
6. **Evaluation:** Validation accuracy assessment
7. **Deployment:** Model persistence for inference

### Design Decisions

**Why Neural Networks?**
- Captures complex non-linear relationships in driving behavior
- Handles high-dimensional feature space effectively
- Generalizes well with proper regularization

**Why These Features?**
- **Spatial features:** Capture preferred routes and areas
- **Temporal features:** Identify work schedule patterns
- **Behavioral features:** Distinguish passenger pickup strategies
- **Movement features:** Reveal driving style (speed, stops)

**Why Dropout?**
- Prevents overfitting on limited driver samples
- Improves generalization to unseen trajectories
- Reduces co-adaptation of neurons

## 🎓 Applications

- **Fleet Management:** Verify driver identity for vehicle assignment
- **Insurance:** Risk assessment based on driving patterns
- **Security:** Detect unauthorized vehicle usage
- **Research:** Study driver behavior and urban mobility patterns

## 📚 Technologies Used

- **Python 3.x** - Core programming language
- **Keras/TensorFlow** - Deep learning framework
- **Pandas** - Data manipulation and CSV processing
- **NumPy** - Numerical computations
- **Scikit-learn** - Preprocessing and model evaluation

