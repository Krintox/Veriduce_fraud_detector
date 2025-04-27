import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_SAMPLES = 100000
FRAUD_RATIO = 0.05  # 5% of samples will be fraudulent
NUM_FEATURES = 7  # Matching our input features

# Feature names matching our API input
FEATURE_NAMES = [
    'total_energy_use',
    'reported_reduction',
    'production_volume',
    'fuel_cost',
    'renewable_energy_pct',
    'historical_avg',
    'industry_avg'
]

def generate_synthetic_data(n_samples, fraud_ratio):
    """Generate synthetic carbon credit data with fraud cases."""
    # Generate normal data (95% of samples)
    n_normal = int(n_samples * (1 - fraud_ratio))
    n_fraud = n_samples - n_normal
    
    # Base distributions for normal data
    data = {
        'total_energy_use': np.random.lognormal(mean=12.5, sigma=0.4, size=n_samples),
        'production_volume': np.random.normal(loc=10000, scale=2000, size=n_samples),
        'fuel_cost': np.random.normal(loc=50000, scale=10000, size=n_samples),
        'renewable_energy_pct': np.random.beta(a=2, b=5, size=n_samples) * 100,
        'historical_avg': np.random.uniform(low=5, high=20, size=n_samples),
        'industry_avg': np.random.uniform(low=10, high=15, size=n_samples)
    }
    
    # Calculate reasonable reductions for normal data
    data['reported_reduction'] = np.clip(
        np.random.normal(
            loc=data['historical_avg'] * 1.1,  # Slightly better than historical
            scale=3,
            size=n_samples
        ),
        0, 100
    )
    
    # Create fraud indicators (all 0 initially)
    data['is_fraud'] = np.zeros(n_samples)
    
    # Select random samples to make fraudulent
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    data['is_fraud'][fraud_indices] = 1
    
    # Modify fraudulent samples to have anomalies
    for i in fraud_indices:
        # Randomly select which features to manipulate
        features_to_alter = random.sample(FEATURE_NAMES[:-2], k=random.randint(2, 4))
        
        for feature in features_to_alter:
            if feature == 'reported_reduction':
                # Make reductions unrealistically high
                data[feature][i] *= random.uniform(1.5, 4)
            elif feature == 'fuel_cost':
                # Make costs unrealistically low
                data[feature][i] *= random.uniform(0.3, 0.7)
            elif feature == 'renewable_energy_pct':
                # Make either too high or too low
                if random.random() > 0.5:
                    data[feature][i] *= random.uniform(1.5, 3)
                else:
                    data[feature][i] *= random.uniform(0.1, 0.5)
            else:
                # For other features, add significant noise
                data[feature][i] *= random.uniform(0.7, 1.5)
    
    # Ensure all values are within realistic bounds
    data['reported_reduction'] = np.clip(data['reported_reduction'], 0, 100)
    data['renewable_energy_pct'] = np.clip(data['renewable_energy_pct'], 0, 100)
    data['fuel_cost'] = np.abs(data['fuel_cost'])
    data['production_volume'] = np.abs(data['production_volume'])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match our expected structure
    df = df[FEATURE_NAMES + ['is_fraud']]
    
    return df

def train_fraud_detection_model():
    """Generate data and train the fraud detection model."""
    # Step 1: Generate synthetic dataset
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(NUM_SAMPLES, FRAUD_RATIO)
    df.to_csv('carbon_credits_100k.csv', index=False)
    print(f"Generated dataset with shape: {df.shape}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    
    # Step 2: Prepare data for training
    X = df.drop(columns=['is_fraud']).values
    y = df['is_fraud'].values
    
    # Split into train and validation (no test set needed for unsupervised)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    
    # Step 3: Initialize and train the model
    print("\nInitializing fraud detector...")
    fraud_detector = AdvancedFraudDetector(
        input_dim=X.shape[1],
        contamination=FRAUD_RATIO  # Expected fraud rate
    )
    
    print("Training model...")
    history = fraud_detector.train(
        X_train=X_train,
        validation_data=X_val,
        epochs=100,
        batch_size=256,
        patience=10,
        model_path='carbon_fraud_model'
    )
    
    # Step 4: Save the trained model
    fraud_detector.save_model('carbon_fraud_model_final')
    print("\nModel training complete and saved!")
    
    # Step 5: Validate on some samples
    print("\nSample validation:")
    sample_data = X_val[:5]
    results = fraud_detector.detect_fraud(sample_data)
    
    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"  Features: {dict(zip(FEATURE_NAMES, sample_data[i]))}")
        print(f"  Actual: {'Fraud' if y[split_idx+i] else 'Legitimate'}")
        print(f"  Predicted: {'Fraud' if results['fraud_flags'][i] else 'Legitimate'}")
        print(f"  Confidence: {results['confidence_scores'][i]:.2f}")
    
    return fraud_detector

# Run the training pipeline
if __name__ == "__main__":
    trained_model = train_fraud_detection_model()