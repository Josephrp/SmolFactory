#!/usr/bin/env python3
"""
Debug script to test Trackio data structure and identify plotting issues
"""

import json
import os
from datetime import datetime
import pandas as pd

def debug_trackio_data():
    """Debug the Trackio data structure"""
    
    # Check if data file exists
    data_file = "trackio_experiments.json"
    print(f"🔍 Checking for data file: {data_file}")
    
    if os.path.exists(data_file):
        print("✅ Data file exists")
        with open(data_file, 'r') as f:
            data = json.load(f)
            print(f"📊 Data structure: {json.dumps(data, indent=2)}")
            
            experiments = data.get('experiments', {})
            print(f"📈 Found {len(experiments)} experiments")
            
            for exp_id, exp_data in experiments.items():
                print(f"\n🔬 Experiment: {exp_id}")
                print(f"   Name: {exp_data.get('name', 'N/A')}")
                print(f"   Status: {exp_data.get('status', 'N/A')}")
                print(f"   Metrics count: {len(exp_data.get('metrics', []))}")
                
                # Check metrics structure
                metrics = exp_data.get('metrics', [])
                if metrics:
                    print(f"   Latest metric entry: {json.dumps(metrics[-1], indent=2)}")
                    
                    # Test DataFrame conversion
                    data_list = []
                    for metric_entry in metrics:
                        step = metric_entry.get('step', 0)
                        timestamp = metric_entry.get('timestamp', '')
                        metrics_data = metric_entry.get('metrics', {})
                        
                        row = {'step': step, 'timestamp': timestamp}
                        row.update(metrics_data)
                        data_list.append(row)
                    
                    df = pd.DataFrame(data_list)
                    print(f"   DataFrame shape: {df.shape}")
                    print(f"   DataFrame columns: {list(df.columns)}")
                    if not df.empty:
                        print(f"   Sample data:\n{df.head()}")
                else:
                    print("   ❌ No metrics found")
    else:
        print("❌ Data file does not exist")
        
        # Create a test experiment to see if data persists
        print("\n🧪 Creating test experiment...")
        test_data = {
            'experiments': {
                'test_exp_001': {
                    'id': 'test_exp_001',
                    'name': 'Test Experiment',
                    'description': 'Debug test',
                    'created_at': datetime.now().isoformat(),
                    'status': 'running',
                    'metrics': [
                        {
                            'timestamp': datetime.now().isoformat(),
                            'step': 25,
                            'metrics': {
                                'loss': 1.165,
                                'accuracy': 0.75,
                                'learning_rate': 3.5e-6
                            }
                        }
                    ],
                    'parameters': {},
                    'artifacts': [],
                    'logs': []
                }
            },
            'current_experiment': 'test_exp_001',
            'last_updated': datetime.now().isoformat()
        }
        
        with open(data_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print("✅ Created test data file")

if __name__ == "__main__":
    debug_trackio_data() 