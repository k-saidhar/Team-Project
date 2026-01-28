import unittest
from unittest.mock import patch, MagicMock
import os
import time
import random
import concurrent.futures

# Import the target module
import allforone

# Mock data
MOCK_FILES = [f"file_{i}.html" for i in range(100)]

def mock_load_file_bytes(folder, fname):
    return fname, f"content_of_{fname}".encode('utf-8')

def mock_ncd(x, y, x_key=None, y_key=None):
    # Simulate work
    time.sleep(0.005) 
    return random.random()

def mock_listdir(path):
    return MOCK_FILES

def mock_exists(path):
    if "cache" in path: return False
    return False

# Capture original print
original_print = print

class TestPipeline(unittest.TestCase):
    
    @patch('allforone.ProcessPoolExecutor', side_effect=concurrent.futures.ThreadPoolExecutor)
    @patch('allforone.load_file_bytes', side_effect=mock_load_file_bytes)
    @patch('allforone.ncd', side_effect=mock_ncd)
    @patch('os.listdir', side_effect=mock_listdir)
    @patch('os.path.exists', side_effect=mock_exists)
    @patch('builtins.print') 
    def test_pipeline_execution(self, mock_print, mock_exists, mock_listdir, mock_ncd, mock_load, mock_executor):
        
        # Use original print
        mock_print.side_effect = original_print 
        
        original_print(">>> STARTING PIPELINE TEST")
        
        # Run clustering with small batch size
        try:
            allforone.incremental_clustering(
                start_batch=1,
                batch_size=20,
                folder="mock_folder",
                dthreshold=0.3,
                max_workers=4,
                chunk_size=5,
                csv_file="test_stats.csv"
            )
        except Exception as e:
            print(f"!!! Exception: {e}")
            raise e
            
        print(">>> PIPELINE TEST COMPLETED")

if __name__ == '__main__':
    unittest.main()
