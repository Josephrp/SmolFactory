#!/usr/bin/env python3
"""
Test script for Hugging Face Datasets integration
"""

import os
import json
from datetime import datetime

def test_hf_datasets_integration():
    """Test the HF Datasets integration"""
    
    print("🧪 Testing Hugging Face Datasets Integration")
    print("=" * 50)
    
    # Check HF_TOKEN
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("✅ HF_TOKEN found")
    else:
        print("❌ HF_TOKEN not found")
        print("Please set HF_TOKEN environment variable")
        return False
    
    # Test dataset loading
    try:
        from datasets import load_dataset
        
        # Get dataset repository from environment variable
        dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        print(f"📊 Loading dataset: {dataset_repo}")
        
        dataset = load_dataset(dataset_repo, token=hf_token)
        print(f"✅ Dataset loaded successfully")
        
        # Check experiments
        if 'train' in dataset:
            experiments = {}
            for row in dataset['train']:
                exp_id = row.get('experiment_id')
                if exp_id:
                    experiments[exp_id] = {
                        'id': exp_id,
                        'name': row.get('name', ''),
                        'metrics': json.loads(row.get('metrics', '[]')),
                        'parameters': json.loads(row.get('parameters', '{}'))
                    }
            
            print(f"📈 Found {len(experiments)} experiments:")
            for exp_id, exp_data in experiments.items():
                metrics_count = len(exp_data['metrics'])
                print(f"   - {exp_id}: {exp_data['name']} ({metrics_count} metrics)")
                
                # Show sample metrics
                if exp_data['metrics']:
                    latest_metric = exp_data['metrics'][-1]
                    if 'metrics' in latest_metric:
                        sample_metrics = latest_metric['metrics']
                        print(f"     Latest: {list(sample_metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

def test_backup_fallback():
    """Test the backup fallback mechanism"""
    
    print("\n🔄 Testing Backup Fallback")
    print("=" * 30)
    
    # Simulate no HF_TOKEN
    original_token = os.environ.get('HF_TOKEN')
    os.environ['HF_TOKEN'] = ''
    
    try:
        # Import and test the TrackioSpace class
        from templates.spaces.trackio.app import TrackioSpace
        
        trackio = TrackioSpace()
        experiments = trackio.experiments
        
        print(f"✅ Backup fallback loaded {len(experiments)} experiments")
        
        for exp_id, exp_data in experiments.items():
            metrics_count = len(exp_data.get('metrics', []))
            print(f"   - {exp_id}: {exp_data.get('name', '')} ({metrics_count} metrics)")
        
        return True
        
    except Exception as e:
        print(f"❌ Backup fallback failed: {e}")
        return False
    
    finally:
        # Restore original token
        if original_token:
            os.environ['HF_TOKEN'] = original_token

def test_metrics_dataframe():
    """Test the metrics DataFrame conversion"""
    
    print("\n📊 Testing Metrics DataFrame Conversion")
    print("=" * 40)
    
    try:
        from templates.spaces.trackio.app import TrackioSpace
        
        trackio = TrackioSpace()
        
        # Test with a known experiment
        exp_id = 'exp_20250720_130853'
        df = trackio.get_metrics_dataframe(exp_id)
        
        if not df.empty:
            print(f"✅ DataFrame created for {exp_id}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head())
            
            # Test plotting
            if 'loss' in df.columns:
                print(f"   Loss range: {df['loss'].min():.4f} - {df['loss'].max():.4f}")
            
            return True
        else:
            print(f"❌ Empty DataFrame for {exp_id}")
            return False
            
    except Exception as e:
        print(f"❌ DataFrame conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Trackio HF Datasets Integration Test")
    print("=" * 50)
    
    # Run tests
    test1 = test_hf_datasets_integration()
    test2 = test_backup_fallback()
    test3 = test_metrics_dataframe()
    
    print("\n📋 Test Results")
    print("=" * 20)
    print(f"HF Datasets Loading: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Backup Fallback: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"DataFrame Conversion: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! Your HF Datasets integration is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the configuration and try again.") 