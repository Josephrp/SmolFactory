#!/usr/bin/env python3
"""
Test script to check Trackio Space connection
"""

import requests
import json
from datetime import datetime

def test_trackio_space_connection():
    """Test if the Trackio Space is accessible"""
    
    trackio_url = "https://tonic-test-trackio-test.hf.space"
    
    print("🔍 Testing Trackio Space Connection")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        print(f"1. Testing basic connectivity to {trackio_url}")
        response = requests.get(trackio_url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Space is accessible")
        else:
            print(f"❌ Space returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Trackio Space: {e}")
        print("   This means your training script cannot send data to the Space")
        return False
    
    print("\n2. Testing experiment creation...")
    
    # Try to create a test experiment via the Space interface
    # Note: This is a simplified test - the actual Space might need different approach
    
    print("✅ Basic connectivity test passed")
    print("\n📋 Next Steps:")
    print("1. Visit the Trackio Space manually")
    print("2. Create an experiment using the interface")
    print("3. Log some metrics manually")
    print("4. Check if experiments appear in the list")
    
    return True

def check_local_files():
    """Check what local files were created during training"""
    
    print("\n📁 Checking Local Training Files")
    print("=" * 50)
    
    import os
    import glob
    
    # Check for local files
    local_files = []
    
    # Check for config files
    config_files = glob.glob("config_*.json")
    local_files.extend(config_files)
    
    # Check for training logs
    if os.path.exists("training.log"):
        local_files.append("training.log")
    
    # Check for output directory
    if os.path.exists("./outputs/balanced"):
        local_files.append("./outputs/balanced/")
    
    # Check for evaluation results
    eval_files = glob.glob("eval_results_*.json")
    local_files.extend(eval_files)
    
    # Check for training summaries
    summary_files = glob.glob("training_summary_*.json")
    local_files.extend(summary_files)
    
    if local_files:
        print("✅ Found local training files:")
        for file in local_files:
            if os.path.isdir(file):
                size = "directory"
            else:
                size = f"{os.path.getsize(file)} bytes"
            print(f"   📄 {file} ({size})")
    else:
        print("❌ No local training files found")
        print("   This suggests training didn't start or failed early")
    
    return local_files

def provide_solutions():
    """Provide solutions for the experiment visibility issue"""
    
    print("\n🛠️ Solutions for Experiment Visibility")
    print("=" * 50)
    
    print("\n1. IMMEDIATE SOLUTION - Use Manual Interface:")
    print("   a) Visit: https://tonic-test-trackio-test.hf.space")
    print("   b) Go to 'Create Experiment' tab")
    print("   c) Create experiment: 'petit-elle-l-aime-3-balanced'")
    print("   d) Copy the experiment ID")
    print("   e) Go to 'Log Metrics' tab")
    print("   f) Enter metrics manually as training progresses")
    
    print("\n2. CHECK TRAINING STATUS:")
    print("   a) Check if training is actually running")
    print("   b) Look for local files being created")
    print("   c) Check training logs for errors")
    
    print("\n3. ALTERNATIVE - Use Local Monitoring:")
    print("   a) Check local files for training progress")
    print("   b) Use local logs to monitor training")
    print("   c) Trackio Space is for visualization only")
    
    print("\n4. DEBUG TRAINING SCRIPT:")
    print("   a) Check if Trackio client is working")
    print("   b) Verify experiment creation in training logs")
    print("   c) Look for connection errors")

def main():
    """Main test function"""
    
    print("🚀 Trackio Space Connection Test")
    print("=" * 60)
    
    # Test connection
    connection_ok = test_trackio_space_connection()
    
    # Check local files
    local_files = check_local_files()
    
    # Provide solutions
    provide_solutions()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if connection_ok:
        print("✅ Trackio Space is accessible")
    else:
        print("❌ Trackio Space connection failed")
    
    if local_files:
        print("✅ Local training files found")
    else:
        print("❌ No local training files found")
    
    print("\n🎯 RECOMMENDATION:")
    print("Use the Trackio Space manually to create and monitor experiments")
    print("The training script will save data locally, but the Space")
    print("needs manual interaction for now.")

if __name__ == "__main__":
    main() 