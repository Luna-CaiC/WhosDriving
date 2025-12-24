import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from evaluation import process_data, run

def test_with_sample_data():
    """
    Test the model with sample data or test.pkl if available
    """
    
    # Try to load the trained model
    try:
        model = pickle.load(open('taxi_driver_model.pkl', 'rb'))
        print("Loaded trained model successfully")
    except FileNotFoundError:
        print("Trained model not found. Please run 'python simple_train.py' first to train the model.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Try to load test data
    try:
        test_data = pickle.load(open('test.pkl', 'rb'))
        print(f"Loaded test data with {len(test_data)} trajectories")
        
        # Test with each trajectory
        for i, trajectory in enumerate(test_data):
            print(f"\nTesting trajectory {i+1}:")
            print(f"Trajectory length: {len(trajectory)}")
            
            # Process the trajectory
            processed_data = process_data(trajectory)
            print(f"Extracted {len(processed_data)} features")
            
            # Make prediction
            prediction = run(processed_data, model)
            print(f"Predicted driver: {prediction}")
            
    except FileNotFoundError:
        print("test.pkl not found. Creating sample test data......")
        
        # Create sample trajectory data for testing
        sample_trajectory = [
            [114.10437, 22.573433, '2016-07-06 00:07:30', 1],
            [114.10500, 22.573500, '2016-07-06 00:08:00', 1],
            [114.10600, 22.573600, '2016-07-06 00:08:45', 0],
            [113.5, 22.3, '2016-07-06 00:09:00', 0],
            [113.6, 22.4, '2016-07-06 00:09:35', 1]
        ]
        
        print("Testing with sample trajectory:")
        print(f"Sample trajectory length: {len(sample_trajectory)}")
        
        # Process the sample trajectory
        processed_data = process_data(sample_trajectory)
        print(f"Extracted {len(processed_data)} features")
        print(f"Features: {processed_data}")
        
        # Make prediction
        prediction = run(processed_data, model)
        print(f"Predicted driver for sample data: {prediction}")
        
    except Exception as e:
        print(f"Error during testing: {e}")

def evaluate_model_performance():
    """
    Evaluate model performance if validation data is available
    """
    try:
        # This would require having a validation dataset with known labels
        print("\nModel evaluation would require labeled validation data")
        print("The current setup focuses on prediction rather than evaluation")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    print("Testing Taxi Driver Classification Model")
    print("=" * 50)
    
    test_with_sample_data()
    evaluate_model_performance()
    
    print("\nTesting completed!")