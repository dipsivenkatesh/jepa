"""
Examples for using JSON, CSV, and Pickle datasets with JEPA.

This example shows how to create and use datasets from different file formats.
"""

import torch
import numpy as np
import json
import pickle
import csv
import os
import tempfile

from openjepa.data import JSONDataset, CSVDataset, PickleDataset, create_dataset


def create_sample_json_data():
    """Create sample JSON files for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Example 1: List format
    list_data = {
        "data_t": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        "data_t1": [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
    }
    list_path = os.path.join(temp_dir, "list_format.json")
    with open(list_path, 'w') as f:
        json.dump(list_data, f)
    
    # Example 2: Dict format
    dict_data = {
        "samples": [
            {"data": [1, 2, 3], "target": [2, 3, 4], "id": 0},
            {"data": [4, 5, 6], "target": [5, 6, 7], "id": 1},
            {"data": [7, 8, 9], "target": [8, 9, 10], "id": 2},
            {"data": [10, 11, 12], "target": [11, 12, 13], "id": 3}
        ]
    }
    dict_path = os.path.join(temp_dir, "dict_format.json")
    with open(dict_path, 'w') as f:
        json.dump(dict_data, f)
    
    # Example 3: Nested format
    nested_data = {
        "dataset": {
            "train": {
                "features": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "targets": [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
            }
        }
    }
    nested_path = os.path.join(temp_dir, "nested_format.json")
    with open(nested_path, 'w') as f:
        json.dump(nested_data, f)
    
    # Example 4: Temporal only (using temporal offset)
    temporal_data = {
        "time_series": [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [1.4, 2.4, 3.4]
        ]
    }
    temporal_path = os.path.join(temp_dir, "temporal_format.json")
    with open(temporal_path, 'w') as f:
        json.dump(temporal_data, f)
    
    return temp_dir, {
        "list": list_path,
        "dict": dict_path,
        "nested": nested_path,
        "temporal": temporal_path
    }


def create_sample_csv_data():
    """Create sample CSV files for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Example 1: Simple CSV with state_t and state_t1 columns
    simple_path = os.path.join(temp_dir, "simple.csv")
    with open(simple_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["state_t_1", "state_t_2", "state_t_3", "state_t1_1", "state_t1_2", "state_t1_3"])
        writer.writerow([1, 2, 3, 2, 3, 4])
        writer.writerow([4, 5, 6, 5, 6, 7])
        writer.writerow([7, 8, 9, 8, 9, 10])
        writer.writerow([10, 11, 12, 11, 12, 13])
    
    # Example 2: Time series CSV (using temporal offset)
    timeseries_path = os.path.join(temp_dir, "timeseries.csv")
    with open(timeseries_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value1", "value2", "value3"])
        for i in range(10):
            writer.writerow([i, i*0.1, i*0.2, i*0.3])
    
    return temp_dir, {
        "simple": simple_path,
        "timeseries": timeseries_path
    }


def create_sample_pickle_data():
    """Create sample pickle files for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Example 1: Dictionary format
    dict_data = {
        "data_t": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        "data_t1": np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
        "metadata": {"source": "synthetic", "version": 1.0}
    }
    dict_path = os.path.join(temp_dir, "dict_format.pkl")
    with open(dict_path, 'wb') as f:
        pickle.dump(dict_data, f)
    
    # Example 2: List format (temporal offset)
    list_data = np.random.randn(20, 5)  # 20 time steps, 5 features
    list_path = os.path.join(temp_dir, "list_format.pkl")
    with open(list_path, 'wb') as f:
        pickle.dump(list_data, f)
    
    return temp_dir, {
        "dict": dict_path,
        "list": list_path
    }


def example_json_datasets():
    """Example of using JSON datasets."""
    print("\n=== JSON Dataset Examples ===")
    
    temp_dir, json_files = create_sample_json_data()
    
    try:
        # Example 1: List format
        dataset1 = JSONDataset(
            data_path=json_files["list"],
            data_t_key="data_t",
            data_t1_key="data_t1",
            data_format="list"
        )
        print(f"List format dataset size: {len(dataset1)}")
        state_t, state_t1 = dataset1[0]
        print(f"Sample shapes: {state_t.shape}, {state_t1.shape}")
        print(f"Sample values: {state_t}, {state_t1}")
        
        # Example 2: Dict format
        dataset2 = JSONDataset(
            data_path=json_files["dict"],
            data_t_key="data",
            data_t1_key="target",
            data_format="dict"
        )
        print(f"\nDict format dataset size: {len(dataset2)}")
        state_t, state_t1 = dataset2[0]
        print(f"Sample shapes: {state_t.shape}, {state_t1.shape}")
        
        # Example 3: Nested format
        dataset3 = JSONDataset(
            data_path=json_files["nested"],
            data_t_key="dataset.train.features",
            data_t1_key="dataset.train.targets",
            data_format="nested"
        )
        print(f"\nNested format dataset size: {len(dataset3)}")
        
        # Example 4: Temporal offset (no data_t1_key)
        dataset4 = JSONDataset(
            data_path=json_files["temporal"],
            data_t_key="time_series",
            temporal_offset=1,
            data_format="list"
        )
        print(f"\nTemporal format dataset size: {len(dataset4)}")
        state_t, state_t1 = dataset4[0]
        print(f"Temporal sample: {state_t} -> {state_t1}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_csv_datasets():
    """Example of using CSV datasets."""
    print("\n=== CSV Dataset Examples ===")
    
    temp_dir, csv_files = create_sample_csv_data()
    
    try:
        # Example 1: Simple CSV with separate state_t and state_t1 columns
        dataset1 = CSVDataset(
            data_path=csv_files["simple"],
            data_t_columns=["state_t_1", "state_t_2", "state_t_3"],
            data_t1_columns=["state_t1_1", "state_t1_2", "state_t1_3"]
        )
        print(f"Simple CSV dataset size: {len(dataset1)}")
        state_t, state_t1 = dataset1[0]
        print(f"Sample shapes: {state_t.shape}, {state_t1.shape}")
        print(f"Sample values: {state_t}, {state_t1}")
        
        # Example 2: Time series CSV (using temporal offset)
        dataset2 = CSVDataset(
            data_path=csv_files["timeseries"],
            data_t_columns=["value1", "value2", "value3"],
            temporal_offset=1
        )
        print(f"\nTime series CSV dataset size: {len(dataset2)}")
        state_t, state_t1 = dataset2[0]
        print(f"Time series sample: {state_t} -> {state_t1}")
        
        # Example 3: Using column indices (fallback)
        dataset3 = CSVDataset(
            data_path=csv_files["timeseries"],
            data_t_columns=["1", "2", "3"],  # Column indices as strings
            temporal_offset=1
        )
        print(f"\nColumn indices CSV dataset size: {len(dataset3)}")
        
    except ImportError:
        print("Pandas not available, CSV examples skipped")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_pickle_datasets():
    """Example of using pickle datasets."""
    print("\n=== Pickle Dataset Examples ===")
    
    temp_dir, pickle_files = create_sample_pickle_data()
    
    try:
        # Example 1: Dictionary format
        dataset1 = PickleDataset(
            data_path=pickle_files["dict"],
            data_t_key="data_t",
            data_t1_key="data_t1"
        )
        print(f"Dict pickle dataset size: {len(dataset1)}")
        state_t, state_t1 = dataset1[0]
        print(f"Sample shapes: {state_t.shape}, {state_t1.shape}")
        
        # Example 2: Direct array format (temporal offset)
        dataset2 = PickleDataset(
            data_path=pickle_files["list"],
            temporal_offset=2
        )
        print(f"\nArray pickle dataset size: {len(dataset2)}")
        state_t, state_t1 = dataset2[0]
        print(f"Temporal sample shapes: {state_t.shape}, {state_t1.shape}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_factory_usage():
    """Example of using the factory function with new dataset types."""
    print("\n=== Factory Function Examples ===")
    
    # Create sample data
    temp_dir, json_files = create_sample_json_data()
    
    try:
        # Using factory function
        dataset = create_dataset(
            data_type="json",
            data_path=json_files["list"],
            data_t_key="data_t",
            data_t1_key="data_t1",
            data_format="list"
        )
        
        print(f"Factory created dataset type: {type(dataset).__name__}")
        print(f"Dataset size: {len(dataset)}")
        
        # Show available dataset types
        from data.dataset import create_dataset
        try:
            create_dataset("invalid_type", "")
        except ValueError as e:
            print(f"Available dataset types: {str(e).split(': ')[1]}")
            
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_with_transforms():
    """Example of using new datasets with transforms."""
    print("\n=== Datasets with Transforms Example ===")
    
    from data.transforms import get_tensor_training_transforms
    
    temp_dir, json_files = create_sample_json_data()
    
    try:
        # Create dataset with transforms
        transforms = get_tensor_training_transforms(add_noise=True, noise_std=0.01)
        
        dataset = JSONDataset(
            data_path=json_files["temporal"],
            data_t_key="time_series",
            temporal_offset=1,
            data_format="list",
            transform=transforms
        )
        
        print(f"Dataset with transforms size: {len(dataset)}")
        state_t, state_t1 = dataset[0]
        print(f"Transformed sample shapes: {state_t.shape}, {state_t1.shape}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run all examples for structured data formats."""
    print("JEPA Structured Data Format Examples")
    print("=" * 50)
    
    try:
        example_json_datasets()
        example_csv_datasets()
        example_pickle_datasets()
        example_factory_usage()
        example_with_transforms()
        
        print("\n" + "=" * 50)
        print("All structured data examples completed successfully!")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
