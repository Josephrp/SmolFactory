#!/usr/bin/env python3
"""
Test script to validate data preservation in Trackio dataset operations
"""

import os
import sys
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_utils import TrackioDatasetManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_experiment(experiment_id: str, name: str, status: str = "running") -> Dict[str, Any]:
    """Create a sample experiment for testing"""
    return {
        'experiment_id': experiment_id,
        'name': name,
        'description': f"Test experiment {name}",
        'created_at': datetime.now().isoformat(),
        'status': status,
        'metrics': json.dumps([
            {
                'timestamp': datetime.now().isoformat(),
                'step': 100,
                'metrics': {
                    'loss': 1.5,
                    'accuracy': 0.85,
                    'learning_rate': 5e-6
                }
            }
        ]),
        'parameters': json.dumps({
            'model_name': 'HuggingFaceTB/SmolLM3-3B',
            'batch_size': 8,
            'learning_rate': 5e-6
        }),
        'artifacts': json.dumps([]),
        'logs': json.dumps([]),
        'last_updated': datetime.now().isoformat()
    }

def test_data_preservation():
    """Test data preservation functionality"""
    # Get HF token from environment
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if not hf_token:
        logger.error("❌ HF_TOKEN not found in environment variables")
        logger.info("Please set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable")
        return False
    
    # Use a test dataset repository
    test_dataset_repo = "tonic/trackio-test-preservation"
    
    try:
        logger.info("🧪 Starting data preservation test")
        logger.info(f"📊 Test dataset: {test_dataset_repo}")
        
        # Initialize dataset manager
        dataset_manager = TrackioDatasetManager(test_dataset_repo, hf_token)
        
        # Test 1: Check if dataset exists
        logger.info("\n📝 Test 1: Checking dataset existence...")
        exists = dataset_manager.check_dataset_exists()
        logger.info(f"Dataset exists: {exists}")
        
        # Test 2: Load existing experiments (should handle empty/non-existent gracefully)
        logger.info("\n📝 Test 2: Loading existing experiments...")
        existing_experiments = dataset_manager.load_existing_experiments()
        logger.info(f"Found {len(existing_experiments)} existing experiments")
        
        # Test 3: Add first experiment
        logger.info("\n📝 Test 3: Adding first experiment...")
        exp1 = create_sample_experiment("test_exp_001", "First Test Experiment")
        success = dataset_manager.upsert_experiment(exp1)
        logger.info(f"First experiment added: {success}")
        
        if not success:
            logger.error("❌ Failed to add first experiment")
            return False
        
        # Test 4: Add second experiment (should preserve first)
        logger.info("\n📝 Test 4: Adding second experiment...")
        exp2 = create_sample_experiment("test_exp_002", "Second Test Experiment")
        success = dataset_manager.upsert_experiment(exp2)
        logger.info(f"Second experiment added: {success}")
        
        if not success:
            logger.error("❌ Failed to add second experiment")
            return False
        
        # Test 5: Verify both experiments exist
        logger.info("\n📝 Test 5: Verifying both experiments exist...")
        all_experiments = dataset_manager.load_existing_experiments()
        logger.info(f"Total experiments after adding two: {len(all_experiments)}")
        
        exp_ids = [exp.get('experiment_id') for exp in all_experiments]
        if "test_exp_001" in exp_ids and "test_exp_002" in exp_ids:
            logger.info("✅ Both experiments preserved successfully")
        else:
            logger.error(f"❌ Experiments not preserved. Found IDs: {exp_ids}")
            return False
        
        # Test 6: Update existing experiment (should preserve others)
        logger.info("\n📝 Test 6: Updating first experiment...")
        exp1_updated = create_sample_experiment("test_exp_001", "Updated First Experiment", "completed")
        success = dataset_manager.upsert_experiment(exp1_updated)
        logger.info(f"First experiment updated: {success}")
        
        if not success:
            logger.error("❌ Failed to update first experiment")
            return False
        
        # Test 7: Verify update preserved other experiments
        logger.info("\n📝 Test 7: Verifying update preserved other experiments...")
        final_experiments = dataset_manager.load_existing_experiments()
        logger.info(f"Total experiments after update: {len(final_experiments)}")
        
        # Check that we still have both experiments
        if len(final_experiments) != 2:
            logger.error(f"❌ Wrong number of experiments after update: {len(final_experiments)}")
            return False
        
        # Check that first experiment was updated
        exp1_final = dataset_manager.get_experiment_by_id("test_exp_001")
        if exp1_final and exp1_final.get('status') == 'completed':
            logger.info("✅ First experiment successfully updated")
        else:
            logger.error("❌ First experiment update failed")
            return False
        
        # Check that second experiment was preserved
        exp2_final = dataset_manager.get_experiment_by_id("test_exp_002")
        if exp2_final and exp2_final.get('name') == "Second Test Experiment":
            logger.info("✅ Second experiment successfully preserved")
        else:
            logger.error("❌ Second experiment not preserved")
            return False
        
        # Test 8: Test filtering functionality
        logger.info("\n📝 Test 8: Testing filtering functionality...")
        running_experiments = dataset_manager.list_experiments(status_filter="running")
        completed_experiments = dataset_manager.list_experiments(status_filter="completed")
        
        logger.info(f"Running experiments: {len(running_experiments)}")
        logger.info(f"Completed experiments: {len(completed_experiments)}")
        
        if len(running_experiments) == 1 and len(completed_experiments) == 1:
            logger.info("✅ Filtering functionality works correctly")
        else:
            logger.error("❌ Filtering functionality failed")
            return False
        
        logger.info("\n🎉 All data preservation tests passed!")
        logger.info("✅ Data preservation functionality is working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Data Preservation Test Suite")
    logger.info("=" * 50)
    
    success = test_data_preservation()
    
    if success:
        logger.info("\n✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
