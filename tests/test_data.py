"""
Test suite for JEPA data handling components.

Tests the dataset classes, transforms, and data utilities.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import json
import pickle
import csv
from PIL import Image
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa.data.dataset import (
    JEPADataset, ImageSequenceDataset, CSVDataset, JSONDataset, 
    PickleDataset, StructuredDataset, get_dataset
)
from jepa.data.transforms import BaseTransform, ImageMaskingTransform, NormalizationTransform
from jepa.data.utils import create_dataloader, validate_data_format, create_temporal_pairs


class TestJEPADataset(unittest.TestCase):
    """Test cases for base JEPADataset class."""
    
    def test_initialization_requires_subclass(self):
        """Test that JEPADataset requires subclass implementation."""
        with self.assertRaises(TypeError):
            # Can't instantiate abstract base class
            dataset = JEPADataset("dummy_path")
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        class TestDataset(JEPADataset):
            def _load_data_files(self):
                return ["file1.txt", "file2.txt"]
            
            # Missing __getitem__ implementation
        
        dataset = TestDataset("dummy_path")
        
        with self.assertRaises(NotImplementedError):
            dataset[0]


class TestImageSequenceDataset(unittest.TestCase):
    """Test cases for ImageSequenceDataset."""
    
    def setUp(self):
        """Set up test fixtures with temporary image files."""
        self.test_dir = tempfile.mkdtemp()
        self.image_files = []
        
        # Create dummy image files
        for i in range(5):
            img = Image.new('RGB', (32, 32), color=(i*50, i*50, i*50))
            img_path = os.path.join(self.test_dir, f'image_{i:03d}.jpg')
            img.save(img_path)
            self.image_files.append(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        for img_path in self.image_files:
            if os.path.exists(img_path):
                os.unlink(img_path)
        os.rmdir(self.test_dir)
    
    def test_initialization(self):
        """Test ImageSequenceDataset initialization."""
        dataset = ImageSequenceDataset(self.test_dir)
        
        self.assertEqual(len(dataset), len(self.image_files))
        self.assertEqual(dataset.sequence_gap, 1)
        self.assertIsInstance(dataset.data_files, list)
    
    def test_invalid_directory(self):
        """Test initialization with invalid directory."""
        with self.assertRaises(ValueError):
            ImageSequenceDataset("/nonexistent/directory")
    
    def test_getitem(self):
        """Test getting items from dataset."""
        dataset = ImageSequenceDataset(self.test_dir)
        
        # Test valid indices
        for i in range(len(dataset) - 1):  # -1 because we need pairs
            img_t, img_t1 = dataset[i]
            self.assertIsInstance(img_t, Image.Image)
            self.assertIsInstance(img_t1, Image.Image)
    
    def test_sequence_gap(self):
        """Test different sequence gaps."""
        for gap in [1, 2, 3]:
            dataset = ImageSequenceDataset(self.test_dir, sequence_gap=gap)
            self.assertEqual(dataset.sequence_gap, gap)
            
            # Test that valid pairs can be created
            if len(dataset.data_files) > gap:
                img_t, img_t1 = dataset[0]
                self.assertIsInstance(img_t, Image.Image)
                self.assertIsInstance(img_t1, Image.Image)
    
    def test_transforms(self):
        """Test transforms are applied correctly."""
        transform = lambda x: x.resize((16, 16))
        dataset = ImageSequenceDataset(self.test_dir, transform=transform)
        
        img_t, img_t1 = dataset[0]
        self.assertEqual(img_t.size, (16, 16))
        self.assertEqual(img_t1.size, (16, 16))
    
    def test_return_indices(self):
        """Test returning indices along with data."""
        dataset = ImageSequenceDataset(self.test_dir, return_indices=True)
        result = dataset[0]
        
        self.assertEqual(len(result), 3)  # img_t, img_t1, index
        img_t, img_t1, idx = result
        self.assertIsInstance(img_t, Image.Image)
        self.assertIsInstance(img_t1, Image.Image)
        self.assertEqual(idx, 0)


class TestCSVDataset(unittest.TestCase):
    """Test cases for CSVDataset."""
    
    def setUp(self):
        """Set up test fixtures with temporary CSV file."""
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Create test CSV data
        data = [
            ['feature1', 'feature2', 'feature3'],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ]
        
        writer = csv.writer(self.test_file)
        writer.writerows(data)
        self.test_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.test_file.name)
    
    def test_initialization(self):
        """Test CSVDataset initialization."""
        dataset = CSVDataset(self.test_file.name)
        
        self.assertEqual(len(dataset), 4)  # 5 data rows - 1 for pairs
        self.assertIsInstance(dataset.data, pd.DataFrame)
    
    def test_getitem(self):
        """Test getting items from CSV dataset."""
        dataset = CSVDataset(self.test_file.name)
        
        data_t, data_t1 = dataset[0]
        self.assertIsInstance(data_t, torch.Tensor)
        self.assertIsInstance(data_t1, torch.Tensor)
        self.assertEqual(data_t.shape, (3,))  # 3 features
        self.assertEqual(data_t1.shape, (3,))
    
    def test_temporal_offset(self):
        """Test different temporal offsets."""
        for offset in [1, 2]:
            dataset = CSVDataset(self.test_file.name, temporal_offset=offset)
            if len(dataset) > 0:
                data_t, data_t1 = dataset[0]
                self.assertIsInstance(data_t, torch.Tensor)
                self.assertIsInstance(data_t1, torch.Tensor)
    
    def test_column_selection(self):
        """Test selecting specific columns."""
        columns = ['feature1', 'feature3']
        dataset = CSVDataset(self.test_file.name, data_columns=columns)
        
        data_t, data_t1 = dataset[0]
        self.assertEqual(data_t.shape, (2,))  # Only 2 selected features
        self.assertEqual(data_t1.shape, (2,))


class TestJSONDataset(unittest.TestCase):
    """Test cases for JSONDataset."""
    
    def setUp(self):
        """Set up test fixtures with temporary JSON file."""
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        
        # Create test JSON data
        data = [
            {'features': [1.0, 2.0, 3.0], 'metadata': {'id': 1}},
            {'features': [4.0, 5.0, 6.0], 'metadata': {'id': 2}},
            {'features': [7.0, 8.0, 9.0], 'metadata': {'id': 3}},
            {'features': [10.0, 11.0, 12.0], 'metadata': {'id': 4}},
            {'features': [13.0, 14.0, 15.0], 'metadata': {'id': 5}}
        ]
        
        json.dump(data, self.test_file)
        self.test_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.test_file.name)
    
    def test_initialization(self):
        """Test JSONDataset initialization."""
        dataset = JSONDataset(self.test_file.name, data_key='features')
        
        self.assertEqual(len(dataset), 4)  # 5 data items - 1 for pairs
        self.assertIsInstance(dataset.data, list)
    
    def test_getitem(self):
        """Test getting items from JSON dataset."""
        dataset = JSONDataset(self.test_file.name, data_key='features')
        
        data_t, data_t1 = dataset[0]
        self.assertIsInstance(data_t, torch.Tensor)
        self.assertIsInstance(data_t1, torch.Tensor)
        self.assertEqual(data_t.shape, (3,))  # 3 features
        self.assertEqual(data_t1.shape, (3,))


class TestPickleDataset(unittest.TestCase):
    """Test cases for PickleDataset."""
    
    def setUp(self):
        """Set up test fixtures with temporary pickle file."""
        self.test_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        
        # Create test pickle data
        data = np.random.randn(10, 5)  # 10 samples, 5 features
        pickle.dump(data, self.test_file)
        self.test_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.test_file.name)
    
    def test_initialization(self):
        """Test PickleDataset initialization."""
        dataset = PickleDataset(self.test_file.name)
        
        self.assertEqual(len(dataset), 9)  # 10 data items - 1 for pairs
        self.assertIsInstance(dataset.data, np.ndarray)
    
    def test_getitem(self):
        """Test getting items from pickle dataset."""
        dataset = PickleDataset(self.test_file.name)
        
        data_t, data_t1 = dataset[0]
        self.assertIsInstance(data_t, torch.Tensor)
        self.assertIsInstance(data_t1, torch.Tensor)
        self.assertEqual(data_t.shape, (5,))  # 5 features
        self.assertEqual(data_t1.shape, (5,))


class TestStructuredDataset(unittest.TestCase):
    """Test cases for StructuredDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files for different formats
        self.csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_data = [['f1', 'f2'], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        writer = csv.writer(self.csv_file)
        writer.writerows(csv_data)
        self.csv_file.close()
        
        self.json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json_data = [{'data': [1, 2]}, {'data': [3, 4]}, {'data': [5, 6]}]
        json.dump(json_data, self.json_file)
        self.json_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.csv_file.name)
        os.unlink(self.json_file.name)
    
    def test_csv_format(self):
        """Test StructuredDataset with CSV format."""
        dataset = StructuredDataset(self.csv_file.name, format='csv')
        
        self.assertGreater(len(dataset), 0)
        data_t, data_t1 = dataset[0]
        self.assertIsInstance(data_t, torch.Tensor)
        self.assertIsInstance(data_t1, torch.Tensor)
    
    def test_json_format(self):
        """Test StructuredDataset with JSON format."""
        dataset = StructuredDataset(
            self.json_file.name, 
            format='json', 
            data_key='data'
        )
        
        self.assertGreater(len(dataset), 0)
        data_t, data_t1 = dataset[0]
        self.assertIsInstance(data_t, torch.Tensor)
        self.assertIsInstance(data_t1, torch.Tensor)
    
    def test_unsupported_format(self):
        """Test StructuredDataset with unsupported format."""
        with self.assertRaises(ValueError):
            StructuredDataset("dummy.txt", format='unsupported')


class TestTransforms(unittest.TestCase):
    """Test cases for transform classes."""
    
    def test_base_transform(self):
        """Test BaseTransform abstract class."""
        transform = BaseTransform()
        
        with self.assertRaises(NotImplementedError):
            transform(torch.randn(10))
    
    def test_normalization_transform(self):
        """Test NormalizationTransform."""
        # Test data
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test standard normalization
        transform = NormalizationTransform(method='standard')
        normalized = transform(data)
        
        self.assertIsInstance(normalized, torch.Tensor)
        self.assertEqual(normalized.shape, data.shape)
        
        # Check that mean is approximately 0 and std is approximately 1
        self.assertAlmostEqual(normalized.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(normalized.std().item(), 1.0, places=5)
        
        # Test min-max normalization
        transform_minmax = NormalizationTransform(method='minmax')
        normalized_minmax = transform_minmax(data)
        
        self.assertAlmostEqual(normalized_minmax.min().item(), 0.0, places=5)
        self.assertAlmostEqual(normalized_minmax.max().item(), 1.0, places=5)


class TestDataUtils(unittest.TestCase):
    """Test cases for data utility functions."""
    
    def test_create_dataloader(self):
        """Test create_dataloader function."""
        # Create mock dataset
        class MockDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return torch.randn(3), torch.randn(3)
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
        
        self.assertEqual(dataloader.batch_size, 2)
        self.assertTrue(dataloader.shuffle)
        
        # Test that dataloader works
        for batch in dataloader:
            state_t, state_t1 = batch
            self.assertEqual(state_t.shape[0], 2)  # batch size
            self.assertEqual(state_t1.shape[0], 2)
            break  # Just test first batch
    
    def test_validate_data_format(self):
        """Test data format validation."""
        # Test valid formats
        self.assertTrue(validate_data_format("data.csv", "csv"))
        self.assertTrue(validate_data_format("data.json", "json"))
        self.assertTrue(validate_data_format("data.pkl", "pickle"))
        
        # Test invalid formats
        self.assertFalse(validate_data_format("data.txt", "csv"))
        self.assertFalse(validate_data_format("data.csv", "json"))
    
    def test_create_temporal_pairs(self):
        """Test temporal pair creation."""
        data = torch.randn(10, 5)  # 10 timesteps, 5 features
        
        pairs = create_temporal_pairs(data, offset=1)
        self.assertEqual(len(pairs), 9)  # 10 - 1
        
        for t, t1 in pairs:
            self.assertEqual(t.shape, (5,))
            self.assertEqual(t1.shape, (5,))
        
        # Test with different offset
        pairs_offset2 = create_temporal_pairs(data, offset=2)
        self.assertEqual(len(pairs_offset2), 8)  # 10 - 2


class TestGetDataset(unittest.TestCase):
    """Test cases for get_dataset factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test files
        self.csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_data = [['f1', 'f2'], [1.0, 2.0], [3.0, 4.0]]
        writer = csv.writer(self.csv_file)
        writer.writerows(csv_data)
        self.csv_file.close()
        
        self.json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json_data = [{'data': [1, 2]}, {'data': [3, 4]}]
        json.dump(json_data, self.json_file)
        self.json_file.close()
        
        self.img_dir = tempfile.mkdtemp()
        img = Image.new('RGB', (32, 32), color=(100, 100, 100))
        self.img_path = os.path.join(self.img_dir, 'test.jpg')
        img.save(self.img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.csv_file.name)
        os.unlink(self.json_file.name)
        os.unlink(self.img_path)
        os.rmdir(self.img_dir)
    
    def test_get_dataset_csv(self):
        """Test get_dataset with CSV file."""
        dataset = get_dataset(self.csv_file.name, dataset_type='csv')
        self.assertIsInstance(dataset, CSVDataset)
    
    def test_get_dataset_json(self):
        """Test get_dataset with JSON file."""
        dataset = get_dataset(
            self.json_file.name, 
            dataset_type='json',
            data_key='data'
        )
        self.assertIsInstance(dataset, JSONDataset)
    
    def test_get_dataset_images(self):
        """Test get_dataset with image directory."""
        dataset = get_dataset(self.img_dir, dataset_type='images')
        self.assertIsInstance(dataset, ImageSequenceDataset)
    
    def test_get_dataset_auto_detect(self):
        """Test get_dataset with automatic type detection."""
        # CSV file
        dataset_csv = get_dataset(self.csv_file.name)
        self.assertIsInstance(dataset_csv, CSVDataset)
        
        # JSON file
        dataset_json = get_dataset(self.json_file.name, data_key='data')
        self.assertIsInstance(dataset_json, JSONDataset)
        
        # Image directory
        dataset_img = get_dataset(self.img_dir)
        self.assertIsInstance(dataset_img, ImageSequenceDataset)
    
    def test_get_dataset_invalid_type(self):
        """Test get_dataset with invalid type."""
        with self.assertRaises(ValueError):
            get_dataset("dummy.txt", dataset_type='invalid')


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset components."""
    
    def test_dataset_with_dataloader(self):
        """Test dataset integration with PyTorch DataLoader."""
        # Create temporary CSV file
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_data = [
            ['f1', 'f2', 'f3'],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
        csv_file.close()
        
        try:
            # Create dataset and dataloader
            dataset = CSVDataset(csv_file.name)
            dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
            
            # Test iteration
            batches = list(dataloader)
            self.assertGreater(len(batches), 0)
            
            for state_t, state_t1 in batches:
                self.assertIsInstance(state_t, torch.Tensor)
                self.assertIsInstance(state_t1, torch.Tensor)
                self.assertEqual(state_t.shape[1], 3)  # 3 features
                self.assertEqual(state_t1.shape[1], 3)
                self.assertLessEqual(state_t.shape[0], 2)  # batch size
        
        finally:
            os.unlink(csv_file.name)


if __name__ == '__main__':
    unittest.main()
