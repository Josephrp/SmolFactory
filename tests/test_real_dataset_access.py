#!/usr/bin/env python3
"""
Test script to verify that the Trackio Space can read from the real Hugging Face dataset
This test requires an HF_TOKEN environment variable to access the dataset
"""

import sys
import os
import json
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_dataset_access():
    """Test direct access to the Hugging Face dataset"""
    try:
        hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            logger.warning("⚠️ No HF_TOKEN found. Skipping real dataset test.")
            logger.info("💡 Set HF_TOKEN environment variable to test with real dataset")
            return False
        
        from datasets import load_dataset
        
        dataset_repo = "Tonic/trackio-experiments"
        logger.info(f"🔧 Testing direct access to {dataset_repo}")
        
        # Load the dataset
        dataset = load_dataset(dataset_repo, token=hf_token)
        
        # Check structure
        experiment_count = len(dataset['train']) if 'train' in dataset else 0
        logger.info(f"📊 Dataset contains {experiment_count} experiments")
        
        if experiment_count == 0:
            logger.warning("⚠️ No experiments found in dataset")
            return False
        
        # Check columns
        columns = list(dataset['train'].column_names) if 'train' in dataset else []
        logger.info(f"📋 Dataset columns: {columns}")
        
        expected_columns = ['experiment_id', 'name', 'description', 'created_at', 'status', 'metrics', 'parameters', 'artifacts', 'logs', 'last_updated']
        missing_columns = [col for col in expected_columns if col not in columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Missing expected columns: {missing_columns}")
        else:
            logger.info("✅ All expected columns present")
        
        # Test parsing a few experiments
        successful_parses = 0
        for i, row in enumerate(dataset['train']):
            if i >= 3:  # Test first 3 experiments
                break
                
            exp_id = row.get('experiment_id', 'unknown')
            logger.info(f"\n🔬 Testing experiment: {exp_id}")
            
            # Test metrics parsing
            metrics_raw = row.get('metrics', '[]')
            try:
                if isinstance(metrics_raw, str):
                    metrics = json.loads(metrics_raw)
                    if isinstance(metrics, list):
                        logger.info(f"   ✅ Metrics parsed: {len(metrics)} entries")
                        if metrics:
                            first_metric = metrics[0]
                            if 'metrics' in first_metric:
                                metric_keys = list(first_metric['metrics'].keys())
                                logger.info(f"   📊 Sample metrics: {metric_keys[:5]}...")
                        successful_parses += 1
                    else:
                        logger.warning(f"   ⚠️ Metrics is not a list: {type(metrics)}")
                else:
                    logger.warning(f"   ⚠️ Metrics is not a string: {type(metrics_raw)}")
            except json.JSONDecodeError as e:
                logger.warning(f"   ❌ Failed to parse metrics JSON: {e}")
            
            # Test parameters parsing
            parameters_raw = row.get('parameters', '{}')
            try:
                if isinstance(parameters_raw, str):
                    parameters = json.loads(parameters_raw)
                    if isinstance(parameters, dict):
                        logger.info(f"   ✅ Parameters parsed: {len(parameters)} entries")
                    else:
                        logger.warning(f"   ⚠️ Parameters is not a dict: {type(parameters)}")
                else:
                    logger.warning(f"   ⚠️ Parameters is not a string: {type(parameters_raw)}")
            except json.JSONDecodeError as e:
                logger.warning(f"   ❌ Failed to parse parameters JSON: {e}")
        
        logger.info(f"\n📋 Successfully parsed {successful_parses} out of {min(3, experiment_count)} test experiments")
        
        return successful_parses > 0
        
    except Exception as e:
        logger.error(f"❌ Error testing direct dataset access: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trackio_space_with_real_dataset():
    """Test TrackioSpace class with real dataset"""
    try:
        hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            logger.warning("⚠️ No HF_TOKEN found. Skipping TrackioSpace test with real dataset.")
            return False
        
        # Add the templates/spaces/trackio directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'templates', 'spaces', 'trackio'))
        
        from app import TrackioSpace
        
        dataset_repo = "Tonic/trackio-experiments"
        logger.info(f"🔧 Testing TrackioSpace with {dataset_repo}")
        
        # Create TrackioSpace instance with real credentials
        trackio_space = TrackioSpace(hf_token=hf_token, dataset_repo=dataset_repo)
        
        # Check if it loaded experiments from the dataset (not backup)
        experiments_count = len(trackio_space.experiments)
        logger.info(f"📊 TrackioSpace loaded {experiments_count} experiments")
        
        if experiments_count == 0:
            logger.warning("⚠️ TrackioSpace loaded no experiments")
            return False
        
        # Check if the dataset manager is available
        if trackio_space.dataset_manager:
            logger.info("✅ Dataset manager is available - data preservation enabled")
        else:
            logger.warning("⚠️ Dataset manager not available - using legacy mode")
        
        # Test loading a specific experiment
        experiment_ids = list(trackio_space.experiments.keys())
        if experiment_ids:
            test_exp_id = experiment_ids[0]
            logger.info(f"🔬 Testing metrics loading for {test_exp_id}")
            
            from app import get_metrics_dataframe
            df = get_metrics_dataframe(test_exp_id)
            
            if not df.empty:
                logger.info(f"✅ Metrics DataFrame created: {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"📊 Available metrics: {list(df.columns)}")
                return True
            else:
                logger.warning(f"⚠️ Metrics DataFrame is empty for {test_exp_id}")
                return False
        else:
            logger.warning("⚠️ No experiments available for testing")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error testing TrackioSpace with real dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting real dataset access test")
    
    # Test direct dataset access
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Direct Dataset Access")
    logger.info("="*60)
    
    direct_test_passed = test_direct_dataset_access()
    
    # Test TrackioSpace with real dataset
    logger.info("\n" + "="*60)
    logger.info("TEST 2: TrackioSpace with Real Dataset")
    logger.info("="*60)
    
    trackio_test_passed = test_trackio_space_with_real_dataset()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Direct Dataset Access: {'✅ PASSED' if direct_test_passed else '❌ FAILED/SKIPPED'}")
    logger.info(f"TrackioSpace Integration: {'✅ PASSED' if trackio_test_passed else '❌ FAILED/SKIPPED'}")
    
    if direct_test_passed and trackio_test_passed:
        logger.info("🎉 All tests passed! The dataset integration is working correctly.")
        sys.exit(0)
    elif not os.environ.get('HF_TOKEN'):
        logger.info("ℹ️ Tests skipped due to missing HF_TOKEN. Set the token to test with real dataset.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
