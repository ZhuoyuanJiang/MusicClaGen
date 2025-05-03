"""
Basic tests for the project.
"""
import pytest
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import preprocess

def test_normalize_data():
    """Test the normalize_data function."""
    import numpy as np
    
    # Create test data
    test_data = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Normalize the data
    normalized = preprocess.normalize_data(test_data)
    
    # Check that the mean is approximately 0 and std is approximately 1
    assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)

def test_project_structure():
    """Test that the project structure is set up correctly."""
    # Check that key directories exist
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src/data',
        'src/models',
        'src/visualization',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        assert os.path.isdir(directory), f"Directory {directory} does not exist" 