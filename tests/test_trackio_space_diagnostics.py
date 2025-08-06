#!/usr/bin/env python3
"""
Diagnostic script for Trackio Space issues
Helps debug dataset loading and API client issues
"""

import os
import sys
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'templates', 'spaces', 'trackio'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_manager():
    """Test dataset manager functionality"""
    try:
        from dataset_utils import TrackioDatasetManager
        
        # Test with environment variables
        hf_token = os.environ.get('HF_TOKEN')
        dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        
        if not hf_token:
            logger.warning("⚠️ HF_TOKEN not found in environment")
            return False
        
        logger.info(f"🔧 Testing dataset manager with repo: {dataset_repo}")
        
        # Initialize dataset manager
        manager = TrackioDatasetManager(dataset_repo, hf_token)
        
        # Test loading experiments
        experiments = manager.load_existing_experiments()
        logger.info(f"📊 Loaded {len(experiments)} experiments from dataset")
        
        # Test creating a sample experiment
        sample_experiment = {
            'experiment_id': f'test_diagnostic_{int(os.urandom(4).hex(), 16)}',
            'name': 'Diagnostic Test Experiment',
            'description': 'Test experiment created by diagnostic script',
            'created_at': '2025-01-27T12:00:00',
            'status': 'completed',
            'metrics': '[]',
            'parameters': '{"test": true}',
            'artifacts': '[]',
            'logs': '[]',
            'last_updated': '2025-01-27T12:00:00'
        }
        
        # Test upsert functionality
        logger.info("🧪 Testing experiment upsert...")
        success = manager.upsert_experiment(sample_experiment)
        
        if success:
            logger.info("✅ Dataset manager working correctly")
            
            # Verify the experiment was saved
            experiments_after = manager.load_existing_experiments()
            logger.info(f"📊 After upsert: {len(experiments_after)} experiments")
            
            return True
        else:
            logger.error("❌ Failed to upsert test experiment")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import dataset_utils: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Dataset manager test failed: {e}")
        return False

def test_trackio_space():
    """Test TrackioSpace initialization"""
    try:
        # Import the TrackioSpace class
        from app import TrackioSpace
        
        logger.info("🧪 Testing TrackioSpace initialization...")
        
        # Initialize TrackioSpace
        space = TrackioSpace()
        
        logger.info(f"📊 TrackioSpace initialized with {len(space.experiments)} experiments")
        logger.info(f"🛡️ Dataset manager available: {'Yes' if space.dataset_manager else 'No'}")
        logger.info(f"🔑 HF Token available: {'Yes' if space.hf_token else 'No'}")
        logger.info(f"📂 Dataset repo: {space.dataset_repo}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import TrackioSpace: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ TrackioSpace test failed: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    logger.info("🔍 Checking environment configuration...")
    
    # Check required environment variables
    env_vars = {
        'HF_TOKEN': os.environ.get('HF_TOKEN'),
        'TRACKIO_DATASET_REPO': os.environ.get('TRACKIO_DATASET_REPO'),
        'TRACKIO_URL': os.environ.get('TRACKIO_URL'),
        'SPACE_ID': os.environ.get('SPACE_ID')
    }
    
    for var, value in env_vars.items():
        if value:
            masked_value = value[:8] + '...' if len(value) > 8 and 'TOKEN' in var else value
            logger.info(f"✅ {var}: {masked_value}")
        else:
            logger.warning(f"⚠️ {var}: Not set")
    
    # Check if running on HF Spaces
    is_hf_spaces = bool(os.environ.get('SPACE_ID'))
    logger.info(f"🚀 Running on HF Spaces: {'Yes' if is_hf_spaces else 'No'}")
    
    return True

def fix_common_issues():
    """Suggest fixes for common issues"""
    logger.info("💡 Common issue fixes:")
    
    # Check dataset repository format
    dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
    if '/' not in dataset_repo:
        logger.warning(f"⚠️ Dataset repo format issue: {dataset_repo} should be 'username/dataset-name'")
    else:
        logger.info(f"✅ Dataset repo format looks good: {dataset_repo}")
    
    # Check for URL issues
    trackio_url = os.environ.get('TRACKIO_URL', 'https://tonic-test-trackio-test.hf.space')
    if trackio_url.startswith('https://https://') or trackio_url.startswith('http://http://'):
        logger.warning(f"⚠️ URL format issue detected: {trackio_url}")
        fixed_url = trackio_url.replace('https://https://', 'https://').replace('http://http://', 'http://')
        logger.info(f"💡 Fixed URL should be: {fixed_url}")
    else:
        logger.info(f"✅ Trackio URL format looks good: {trackio_url}")

def main():
    """Run all diagnostic tests"""
    logger.info("🔧 Starting Trackio Space diagnostics...")
    logger.info("=" * 60)
    
    try:
        # Test environment
        test_environment()
        logger.info("-" * 40)
        
        # Test dataset manager
        dataset_manager_ok = test_dataset_manager()
        logger.info("-" * 40)
        
        # Test TrackioSpace
        trackio_space_ok = test_trackio_space()
        logger.info("-" * 40)
        
        # Suggest fixes
        fix_common_issues()
        logger.info("-" * 40)
        
        # Summary
        logger.info("📋 DIAGNOSTIC SUMMARY:")
        logger.info(f"Dataset Manager: {'✅ OK' if dataset_manager_ok else '❌ Issues'}")
        logger.info(f"TrackioSpace: {'✅ OK' if trackio_space_ok else '❌ Issues'}")
        
        if dataset_manager_ok and trackio_space_ok:
            logger.info("🎉 All systems appear to be working correctly!")
            logger.info("💡 The issues in the logs might be related to:")
            logger.info("   - Empty dataset (expected for new setup)")
            logger.info("   - API client URL formatting (being auto-fixed)")
            logger.info("   - Remote data access (falling back to local data)")
        else:
            logger.warning("⚠️ Some issues detected. Check the logs above for details.")
        
    except Exception as e:
        logger.error(f"❌ Diagnostic script failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
