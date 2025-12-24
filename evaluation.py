import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

def process_data(traj):
    def extract_features(trajectory):
        """Extract comprehensive features from a trajectory"""
        if len(trajectory) == 0:
            return np.zeros(20)
            
        # Convert to numpy array for faster processing
        traj_array = np.array(trajectory, dtype=object)
        
        # basic features
        traj_length = len(trajectory)
        
        # Location features
        lons = traj_array[:, 0].astype(float)
        lats = traj_array[:, 1].astype(float)
        
        lon_mean = np.mean(lons)
        lat_mean = np.mean(lats)
        lon_std = np.std(lons) if len(lons) > 1 else 0
        lat_std = np.std(lats) if len(lats) > 1 else 0
        lon_range = np.max(lons) - np.min(lons)
        lat_range = np.max(lats) - np.min(lats)
        
        # Status features (occupied vs vacant)
        statuses = traj_array[:, 3].astype(int)
        occupied_ratio = np.mean(statuses)
        status_changes = np.sum(np.diff(statuses) != 0)
        
        # Time-based features
        try:
            # Handle different possible time formats
            times = []
            for time_str in traj_array[:, 2]:
                try:
                    if isinstance(time_str, str):
                        # Try different time formats
                        if 'T' in time_str:
                            time_obj = datetime.strptime(time_str[:19], '%Y-%m-%dT%H:%M:%S')
                        else:
                            time_obj = datetime.strptime(time_str[:19], '%Y-%m-%d %H:%M:%S')
                    else:
                        time_obj = datetime.strptime(str(time_str)[:19], '%Y-%m-%d %H:%M:%S')
                    times.append(time_obj)
                except:
                    # If parsing fails, use a default time
                    times.append(datetime(2016, 7, 1, 12, 0, 0))
            
            time_duration = (times[-1] - times[0]).total_seconds() / 3600  # hours
            
            hours = [t.hour for t in times]
            hour_mean = np.mean(hours)
            hour_std = np.std(hours) if len(hours) > 1 else 0
            
            # Peak hours (7-9 AM, 5-7 PM)
            peak_morning = np.mean([(7 <= h <= 9) for h in hours])
            peak_evening = np.mean([(17 <= h <= 19) for h in hours])
        except Exception as e:
            time_duration = 0
            hour_mean = 12
            hour_std = 0
            peak_morning = 0
            peak_evening = 0
            
        # Movement features (vectorized for better performance)
        if len(trajectory) > 1:
            lon_diffs = np.diff(lons) * 111000  # rough conversion to meters
            lat_diffs = np.diff(lats) * 111000
            distances = np.sqrt(lon_diffs**2 + lat_diffs**2)
            
            total_distance = np.sum(distances)
            avg_distance = np.mean(distances)
            
            # Speed calculation
            try:
                time_diffs = np.array([(times[i] - times[i-1]).total_seconds() 
                                     for i in range(1, len(times))])
                valid_times = time_diffs > 0
                if np.any(valid_times):
                    speeds = distances[valid_times] / time_diffs[valid_times]
                    avg_speed = np.mean(speeds)
                    max_speed = np.max(speeds)
                else:
                    avg_speed = 0
                    max_speed = 0
            except:
                avg_speed = 0
                max_speed = 0
        else:
            total_distance = 0
            avg_distance = 0
            avg_speed = 0
            max_speed = 0
            
        # Stopping patterns
        if len(trajectory) > 1:
            stop_threshold = 0.001  # degrees
            lon_stops = np.abs(np.diff(lons)) < stop_threshold
            lat_stops = np.abs(np.diff(lats)) < stop_threshold
            stop_points = np.sum(lon_stops & lat_stops)
            stop_ratio = stop_points / (traj_length - 1) if traj_length > 1 else 0
        else:
            stop_points = 0
            stop_ratio = 0
        
        features = [
            traj_length,
            lon_mean, lat_mean, lon_std, lat_std, lon_range, lat_range,
            occupied_ratio, status_changes,
            time_duration, hour_mean, hour_std, peak_morning, peak_evening,
            total_distance, avg_distance, avg_speed, max_speed,
            stop_points, stop_ratio
        ]
        
        # Ensure all features are finite numbers
        features = [float(f) if np.isfinite(f) else 0.0 for f in features]
        
        return np.array(features)
    
    # Extract features from the trajectory
    features = extract_features(traj)
    return features

def run(data, model):
    """
    Run prediction on processed data
    
    Input:
        data: the output of process_data function
        model: the trained model (either the model itself or loaded from pickle)
    Output:
        prediction: the predicted label(plate) of the data, an int value
    """
    
    # Handle case where model is loaded from pickle and contains both model and scaler
    if isinstance(model, dict) and 'model' in model and 'scaler' in model:
        actual_model = model['model']
        scaler = model['scaler']
        
        # Scale the data
        data_scaled = scaler.transform(data.reshape(1, -1))
        
        # Make prediction
        prediction_probs = actual_model.predict(data_scaled, verbose=0)
        prediction = np.argmax(prediction_probs[0])
        
    else:
        # Assume model is the actual Keras model (backward compatibility)
        # This might not work well without scaling if the model was trained on scaled data
        prediction_probs = model.predict(data.reshape(1, -1), verbose=0)
        
        # For binary classification (sigmoid output)
        if prediction_probs.shape[1] == 1:
            prediction = 1 if prediction_probs[0][0] > 0.5 else 0
        else:
            # For multi-class classification (softmax output)
            prediction = np.argmax(prediction_probs[0])
    
    return int(prediction)