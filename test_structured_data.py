"""
Simple test script to verify the structured data classes work correctly.
Run this to test JSON, CSV, and Pickle datasets.
"""

import tempfile
import json
import pickle
import pandas as pd
import numpy as np
import torch
import os

# Test the new dataset classes
def test_structured_datasets():
    print("Testing structured data classes...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test JSONDataset
        print("\n1. Testing JSONDataset...")
        
        # Create sample JSON data
        json_data = {
            "single_array": [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]],
            "data_t": [[1.0, 2.0], [3.0, 4.0]],
            "data_t1": [[1.1, 2.1], [3.1, 4.1]]
        }
        
        json_path = os.path.join(temp_dir, "test.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        
        # Test single array format
        try:
            from openjepa.data.dataset import JSONDataset
            
            dataset1 = JSONDataset(
                json_path=json_path,
                data_key="single_array",
                time_offset=1
            )
            print(f"  Single array dataset length: {len(dataset1)}")
            data_t, data_t1 = dataset1[0]
            print(f"  Sample shapes: {data_t.shape}, {data_t1.shape}")
            
            # Test separate arrays format
            dataset2 = JSONDataset(
                json_path=json_path,
                data_t_key="data_t",
                data_t1_key="data_t1"
            )
            print(f"  Separate arrays dataset length: {len(dataset2)}")
            data_t, data_t1 = dataset2[0]
            print(f"  Sample shapes: {data_t.shape}, {data_t1.shape}")
            print("  ✓ JSONDataset working correctly")
            
        except Exception as e:
            print(f"  ✗ JSONDataset error: {e}")
        
        # Test CSVDataset
        print("\n2. Testing CSVDataset...")
        
        try:
            from openjepa.data.dataset import CSVDataset
            
            # Create sample CSV data
            df = pd.DataFrame({
                'feature1': [1.0, 1.1, 1.2, 1.3],
                'feature2': [2.0, 2.1, 2.2, 2.3],
                'data_t_1': [3.0, 3.1, 3.2, 3.3],
                'data_t_2': [4.0, 4.1, 4.2, 4.3],
                'data_t1_1': [3.1, 3.2, 3.3, 3.4],
                'data_t1_2': [4.1, 4.2, 4.3, 4.4]
            })
            
            csv_path = os.path.join(temp_dir, "test.csv")
            df.to_csv(csv_path, index=False)
            
            # Test single columns format
            dataset3 = CSVDataset(
                csv_path=csv_path,
                data_columns=['feature1', 'feature2'],
                time_offset=1
            )
            print(f"  Single columns dataset length: {len(dataset3)}")
            data_t, data_t1 = dataset3[0]
            print(f"  Sample shapes: {data_t.shape}, {data_t1.shape}")
            
            # Test separate columns format
            dataset4 = CSVDataset(
                csv_path=csv_path,
                data_t_columns=['data_t_1', 'data_t_2'],
                data_t1_columns=['data_t1_1', 'data_t1_2']
            )
            print(f"  Separate columns dataset length: {len(dataset4)}")
            data_t, data_t1 = dataset4[0]
            print(f"  Sample shapes: {data_t.shape}, {data_t1.shape}")
            print("  ✓ CSVDataset working correctly")
            
        except Exception as e:
            print(f"  ✗ CSVDataset error: {e}")
        
        # Test PickleDataset
        print("\n3. Testing PickleDataset...")
        
        try:
            from openjepa.data.dataset import PickleDataset
            
            # Create sample pickle data
            pickle_data = {
                "timeseries": [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]],
                "data_t": [[1.0, 2.0], [3.0, 4.0]],
                "data_t1": [[1.1, 2.1], [3.1, 4.1]]
            }
            
            pickle_path = os.path.join(temp_dir, "test.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(pickle_data, f)
            
            # Test single array format
            dataset5 = PickleDataset(
                pickle_path=pickle_path,
                data_key="timeseries",
                time_offset=1
            )
            print(f"  Single array dataset length: {len(dataset5)}")
            data_t, data_t1 = dataset5[0]
            print(f"  Sample shapes: {data_t.shape}, {data_t1.shape}")
            
            # Test separate arrays format
            dataset6 = PickleDataset(
                pickle_path=pickle_path,
                data_t_key="data_t",
                data_t1_key="data_t1"
            )
            print(f"  Separate arrays dataset length: {len(dataset6)}")
            data_t, data_t1 = dataset6[0]
            print(f"  Sample shapes: {data_t.shape}, {data_t1.shape}")
            print("  ✓ PickleDataset working correctly")
            
        except Exception as e:
            print(f"  ✗ PickleDataset error: {e}")
        
        # Test factory function
        print("\n4. Testing factory function...")
        
        try:
            from openjepa.data.dataset import create_dataset
            
            # Test JSON factory
            dataset_json = create_dataset(
                data_type="json",
                data_path=json_path,
                data_key="single_array",
                time_offset=1
            )
            print(f"  Factory JSON dataset length: {len(dataset_json)}")
            
            # Test CSV factory  
            dataset_csv = create_dataset(
                data_type="csv",
                data_path=csv_path,
                data_columns=['feature1', 'feature2'],
                time_offset=1
            )
            print(f"  Factory CSV dataset length: {len(dataset_csv)}")
            
            # Test Pickle factory
            dataset_pickle = create_dataset(
                data_type="pickle",
                data_path=pickle_path,
                data_key="timeseries",
                time_offset=1
            )
            print(f"  Factory Pickle dataset length: {len(dataset_pickle)}")
            print("  ✓ Factory function working correctly")
            
        except Exception as e:
            print(f"  ✗ Factory function error: {e}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_structured_datasets()
