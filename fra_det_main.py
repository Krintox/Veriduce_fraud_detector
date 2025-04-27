import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import logging
from sklearn.model_selection import train_test_split

class FraudDetector:
    """Simplified fraud detection model with autoencoder and anomaly detection."""
    
    def __init__(self, input_dim=7, contamination=0.05):  # 5% fraud rate
        self.input_dim = input_dim
        self.contamination = contamination
        self.scaler = RobustScaler()
        
        # Simplified autoencoder
        self.autoencoder = self._build_simple_autoencoder()
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.robust_cov = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('FraudDetector')
    
    def _build_simple_autoencoder(self):
        """Build a simpler autoencoder model."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.input_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, epochs=50, batch_size=256):
        """Train all components of the fraud detector."""
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train autoencoder
        self.logger.info("Training autoencoder...")
        self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Train anomaly detectors
        self.logger.info("Training isolation forest...")
        self.isolation_forest.fit(X_scaled)
        
        self.logger.info("Training robust covariance...")
        self.robust_cov.fit(X_scaled)
        
        self.logger.info("Training complete!")
    
    def detect_fraud(self, data):
        """Detect fraudulent data points."""
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        # Autoencoder reconstruction error
        reconstructions = self.autoencoder.predict(data_scaled)
        ae_errors = np.mean(np.square(data_scaled - reconstructions), axis=1)
        ae_scores = (ae_errors > np.quantile(ae_errors, 1-self.contamination)).astype(int)
        
        # Isolation forest predictions
        iso_scores = (self.isolation_forest.predict(data_scaled) == -1).astype(int)
        
        # Robust covariance predictions
        cov_scores = (self.robust_cov.predict(data_scaled) == -1).astype(int)
        
        # Combine scores
        combined_scores = 0.4*ae_scores + 0.3*iso_scores + 0.3*cov_scores
        predictions = (combined_scores >= 0.5).astype(int)
        
        return {
            'fraud_flags': predictions,
            'confidence': np.abs(combined_scores - 0.5) * 2,
            'autoencoder_errors': ae_errors,
            'isolation_scores': iso_scores,
            'covariance_scores': cov_scores
        }
    
    def save_model(self, path):
        """Save all model components."""
        self.autoencoder.save(f"{path}_autoencoder.h5")
        import joblib
        joblib.dump(self.isolation_forest, f"{path}_isolation.pkl")
        joblib.dump(self.robust_cov, f"{path}_covariance.pkl")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
    
    def load_model(self, path):
        """Load all model components."""
        self.autoencoder = tf.keras.models.load_model(f"{path}_autoencoder.h5")
        import joblib
        self.isolation_forest = joblib.load(f"{path}_isolation.pkl")
        self.robust_cov = joblib.load(f"{path}_covariance.pkl")
        self.scaler = joblib.load(f"{path}_scaler.pkl")

# Load and prepare the dataset
def load_and_prepare_data(filepath):
    """Load and prepare the carbon credits dataset."""
    df = pd.read_csv(filepath)
    
    # Check dataset structure
    required_columns = [
        'total_energy_use', 'reported_reduction', 'production_volume',
        'fuel_cost', 'renewable_energy_pct', 'historical_avg', 
        'industry_avg', 'is_fraud'
    ]
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Dataset missing required columns")
    
    # Separate features and labels
    X = df.drop(columns=['is_fraud']).values
    y = df['is_fraud'].values
    
    return X, y

# Main training function
def train_fraud_detector():
    """Main function to train the fraud detector."""
    # Load data
    try:
        X, y = load_and_prepare_data('E:/Bunny/Capstone_2025/machine_learning/fra_det_main/carbon_credits_100k.csv')
        print(f"Dataset loaded with {len(X)} samples")
        print(f"Fraud rate: {y.mean():.2%}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize and train model
    detector = FraudDetector(input_dim=X.shape[1], contamination=y.mean())
    
    # Train-test split (we'll use all data for training in this case)
    X_train = X  # Using all data for training
    
    # Train the model
    detector.train(X_train, epochs=50, batch_size=256)
    
    # Save the trained model
    detector.save_model('trained_fraud_detector')
    print("Model saved as 'trained_fraud_detector'")
    
    # Test detection on some samples
    test_samples = X[:5]
    results = detector.detect_fraud(test_samples)
    
    print("\nSample Detections:")
    for i, (sample, pred) in enumerate(zip(test_samples, results['fraud_flags'])):
        print(f"\nSample {i+1}:")
        print(f"Features: {sample}")
        print(f"Predicted: {'Fraud' if pred else 'Legitimate'}")
        print(f"Confidence: {results['confidence'][i]:.2f}")

# Run the training
if __name__ == "__main__":
    train_fraud_detector()
    
    
# total_energy_use:	Total energy consumption of the company
# reported_reduction:	Reduction in emissions as reported by the company
# production_volume:	Total production volume of the company
# fuel_cost:	Cost of fuel used
# renewable_energy_pct:	Percentage of renewable energy used
# historical_avg:	Historical average energy usage
# industry_avg:	Industry-wide average energy usage