#!/usr/bin/env python3
"""
Test script to verify that the Trackio Space can properly read from the actual dataset
"""

import sys
import os
import json
import logging
from typing import Dict, Any

# Add the templates/spaces/trackio directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'templates', 'spaces', 'trackio'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """Test loading experiments from the actual dataset"""
    try:
        # Import the TrackioSpace class
        from app import TrackioSpace
        
        # Create a TrackioSpace instance pointing to the real dataset
        dataset_repo = "Tonic/trackio-experiments"
        hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            logger.warning("⚠️ No HF_TOKEN found in environment. Testing with public access.")
        
        logger.info(f"🔧 Testing dataset loading from {dataset_repo}")
        
        # Create TrackioSpace instance
        trackio_space = TrackioSpace(hf_token=hf_token, dataset_repo=dataset_repo)
        
        # Check how many experiments were loaded
        experiments_count = len(trackio_space.experiments)
        logger.info(f"📊 Loaded {experiments_count} experiments")
        
        if experiments_count == 0:
            logger.warning("⚠️ No experiments loaded - this might indicate a problem")
            return False
        
        # Test specific experiment IDs from the logs
        test_experiment_ids = [
            'exp_20250720_130853',
            'exp_20250720_134319',
            'exp_20250727_172507',
            'exp_20250727_172526'
        ]
        
        found_experiments = []
        for exp_id in test_experiment_ids:
            if exp_id in trackio_space.experiments:
                found_experiments.append(exp_id)
                experiment = trackio_space.experiments[exp_id]
                
                logger.info(f"✅ Found experiment: {exp_id}")
                logger.info(f"   Name: {experiment.get('name', 'N/A')}")
                logger.info(f"   Status: {experiment.get('status', 'N/A')}")
                logger.info(f"   Metrics count: {len(experiment.get('metrics', []))}")
                logger.info(f"   Parameters count: {len(experiment.get('parameters', {}))}")
                
                # Test metrics parsing specifically
                metrics = experiment.get('metrics', [])
                if metrics:
                    logger.info(f"   First metric entry: {metrics[0] if metrics else 'None'}")
                    
                    # Test if we can get a DataFrame for this experiment
                    from app import get_metrics_dataframe
                    df = get_metrics_dataframe(exp_id)
                    if not df.empty:
                        logger.info(f"   ✅ DataFrame created successfully: {len(df)} rows, {len(df.columns)} columns")
                        logger.info(f"   Available metrics: {list(df.columns)}")
                    else:
                        logger.warning(f"   ⚠️ DataFrame is empty for {exp_id}")
                else:
                    logger.warning(f"   ⚠️ No metrics found for {exp_id}")
        
        logger.info(f"📋 Found {len(found_experiments)} out of {len(test_experiment_ids)} test experiments")
        
        if found_experiments:
            logger.info("✅ Dataset loading appears to be working correctly!")
            return True
        else:
            logger.warning("⚠️ No test experiments found - dataset loading may have issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_parsing():
    """Test parsing metrics from the actual dataset format"""
    try:
        # Test with actual data structure from the dataset
        sample_metrics_json = '''[{"timestamp": "2025-07-20T11:20:01.780908", "step": 25, "metrics": {"loss": 1.1659, "grad_norm": 10.3125, "learning_rate": 7e-08, "num_tokens": 1642080.0, "mean_token_accuracy": 0.75923578992486, "epoch": 0.004851130919895701}}, {"timestamp": "2025-07-20T11:26:39.042155", "step": 50, "metrics": {"loss": 1.165, "grad_norm": 10.75, "learning_rate": 1.4291666666666667e-07, "num_tokens": 3324682.0, "mean_token_accuracy": 0.7577659255266189, "epoch": 0.009702261839791402}}]'''
        
        logger.info("🔧 Testing metrics parsing")
        
        # Parse the JSON
        metrics_list = json.loads(sample_metrics_json)
        logger.info(f"📊 Parsed {len(metrics_list)} metric entries")
        
        # Convert to DataFrame format (like the app does)
        import pandas as pd
        df_data = []
        for metric_entry in metrics_list:
            if isinstance(metric_entry, dict):
                step = metric_entry.get('step', 0)
                timestamp = metric_entry.get('timestamp', '')
                metrics = metric_entry.get('metrics', {})
                
                row = {'step': step, 'timestamp': timestamp}
                row.update(metrics)
                df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            logger.info(f"✅ DataFrame created: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"📋 Columns: {list(df.columns)}")
            logger.info(f"📊 Sample data:\n{df.head()}")
            return True
        else:
            logger.warning("⚠️ No data converted to DataFrame format")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing metrics parsing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting Trackio dataset fix verification")
    
    # Test metrics parsing first
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Metrics Parsing")
    logger.info("="*50)
    
    metrics_test_passed = test_metrics_parsing()
    
    # Test dataset loading
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Dataset Loading")
    logger.info("="*50)
    
    dataset_test_passed = test_dataset_loading()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    logger.info(f"Metrics Parsing: {'✅ PASSED' if metrics_test_passed else '❌ FAILED'}")
    logger.info(f"Dataset Loading: {'✅ PASSED' if dataset_test_passed else '❌ FAILED'}")
    
    if metrics_test_passed and dataset_test_passed:
        logger.info("🎉 All tests passed! The dataset fix should work correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
