import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

def validate_csv_file(file_path):
    """
    Validate a single CSV file
    """
    try:
        print(f"\n{'='*60}")
        print(f"Validating: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # load the CSV file
        data = pd.read_csv(file_path)
        
        print(f"File shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Check required columns
        required_columns = ['plate', 'longitude', 'latitude', 'time', 'status']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Check data types and ranges
        print(f"\nData types:")
        print(data.dtypes)
        
        # Check plates (drivers)
        unique_plates = sorted(data['plate'].unique())
        print(f"\nUnique plates (drivers): {unique_plates}")
        print(f"Number of drivers: {len(unique_plates)}")
        
        # Check trajectories per driver
        print(f"\nRecords per driver:")
        driver_counts = data['plate'].value_counts().sort_index()
        for plate in unique_plates:
            count = driver_counts.get(plate, 0)
            print(f"  Driver {plate}: {count} records")
        
        # Check coordinate ranges (should be reasonable for taxi data)
        lon_range = (data['longitude'].min(), data['longitude'].max())
        lat_range = (data['latitude'].min(), data['latitude'].max())
        
        print(f"\nCoordinate ranges:")
        print(f"  Longitude: {lon_range[0]:.6f} to {lon_range[1]:.6f}")
        print(f"  Latitude: {lat_range[0]:.6f} to {lat_range[1]:.6f}")
        
        # Check if coordinates look reasonable (Hong Kong area based on sample data)
        if 113.0 <= lon_range[0] and lon_range[1] <= 115.0 and 22.0 <= lat_range[0] and lat_range[1] <= 23.0:
            print("✅ Coordinates look reasonable for Hong Kong area")
        else:
            print("⚠️  Coordinates might be outside expected range")
        
        # Check time format
        print(f"\nTime data:")
        print(f"  Sample times: {data['time'].iloc[:3].tolist()}")
        
        try:
            # Try to parse a few time entries
            sample_times = data['time'].iloc[:5]
            for i, time_str in enumerate(sample_times):
                datetime.strptime(str(time_str), '%Y-%m-%d %H:%M:%S')
            print("✅ Time format is valid")
        except Exception as e:
            print(f"❌ Time format issue: {e}")
        
        # Check status values
        unique_status = sorted(data['status'].unique())
        print(f"\nStatus values: {unique_status}")
        
        if set(unique_status).issubset({0, 1}):
            print("✅ Status values are binary (0, 1) as expected")
        else:
            print("⚠️  Status values are not binary")
        
        status_counts = data['status'].value_counts()
        print(f"Status distribution:")
        print(f"  Status 0 (vacant): {status_counts.get(0, 0)} records")
        print(f"  Status 1 (occupied): {status_counts.get(1, 0)} records")
        
        # Check for missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() == 0:
            print("✅ No missing values found")
        else:
            print(f"⚠️  Missing values found:")
            for col, count in missing_data.items():
                if count > 0:
                    print(f"  {col}: {count} missing")
        
        # Sample trajectory for one driver
        sample_driver = unique_plates[0]
        sample_traj = data[data['plate'] == sample_driver].head(10)
        print(f"\nSample trajectory (Driver {sample_driver}, first 10 records):")
        print(sample_traj.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False

def validate_all_data():
    """
    Validate all CSV files in common data directories
    """
    print("Taxi Driver Dataset Validation")
    print("=" * 88)
    
    # Look for data in common locations
    data_paths = [
        'data/',
        'data_5drivers/',
        './',  # Current directory
    ]
    
    all_files = []
    
    for path in data_paths:
        if os.path.exists(path) and os.path.isdir(path):
            csv_files = glob.glob(os.path.join(path, '*.csv'))
            if csv_files:
                all_files.extend(csv_files)
                print(f"Found {len(csv_files)} CSV files in {path}")
                break
    
    if not all_files:
        print("❌ No CSV files found in any of the expected directories:")
        for path in data_paths:
            print(f"  - {path}")
        return
    
    # Sort files by name
    all_files.sort()
    
    print(f"\nFound {len(all_files)} CSV files total")
    print("Files:", [os.path.basename(f) for f in all_files])
    
    # Validate each file
    valid_files = 0
    total_trajectories = 0
    all_drivers = set()
    
    for file_path in all_files:
        if validate_csv_file(file_path):
            valid_files += 1
            
            # Count trajectories and drivers
            try:
                data = pd.read_csv(file_path)
                drivers_in_file = data['plate'].unique()
                trajectories_in_file = len(drivers_in_file)
                total_trajectories += trajectories_in_file
                all_drivers.update(drivers_in_file)
            except:
                pass
    
    # Summary
    print(f"\n{'='*66}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*66}")
    print(f"Total files processed: {len(all_files)}")
    print(f"Valid files: {valid_files}")
    print(f"Total trajectories across all files: {total_trajectories}")
    print(f"Unique drivers across all files: {sorted(all_drivers)}")
    print(f"Number of unique drivers: {len(all_drivers)}")
    
    if valid_files == len(all_files):
        print("✅ All files are valid and ready for training!")
    else:
        print(f"⚠️  {len(all_files) - valid_files} files have issues")
    
    # recommendations
    print(f"\nRECOMMENDATIONS:")
    if len(all_drivers) >= 3:
        print("✅ Good number of drivers for classification")
    else:
        print("⚠️  Very few drivers - classification might be difficult")
    
    if total_trajectories >= 50:
        print("✅ Good amount of data for training")
    elif total_trajectories >= 20:
        print("⚠️  Limited data - model performance might be affected")
    else:
        print("❌ Very little data - consider getting more data")

def test_feature_extraction():
    """
    Test the feature extraction on sample data
    """
    print(f"\n{'='*68}")
    print(f"TESTING FEATURE EXTRACTION")
    print(f"{'='*68}")
    
    # import the evaluation module
    try:
        from evaluation import process_data
        print("✅ Successfully imported process_data function")
    except ImportError as e:
        print(f"❌ Cannot import evaluation module: {e}")
        return
    
    # Create sample trajectory data
    sample_trajectory = [
        [114.10437, 22.573433, '2016-07-06 00:07:30', 1],
        [114.10500, 22.573500, '2016-07-06 00:08:00', 1],
        [114.10600, 22.573600, '2016-07-06 00:08:45', 0],
        [113.5, 22.3, '2016-07-06 00:09:00', 0],
        [113.6, 22.4, '2016-07-06 00:09:35', 1]
    ]
    
    print(f"Testing with sample trajectory of {len(sample_trajectory)} points...")
    
    try:
        features = process_data(sample_trajectory)
        print(f"✅ Feature extraction successful!")
        print(f"Number of features extracted: {len(features)}")
        print(f"Features: {features}")
        
        # Check if all features are finite numbers
        if np.all(np.isfinite(features)):
            print("✅ All features are valid numbers")
        else:
            print("⚠️  Some features are invalid (NaN or infinite)")
            
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")

if __name__ == "__main__":
    # Run validation
    validate_all_data()
    
    # Test feature extraction
    test_feature_extraction()
    
    print(f"\n{'='*88}")
    print("Validation completed!")
    print("If all checks passed, you can run: python simple_train.py")
    print(f"{'='*88}")