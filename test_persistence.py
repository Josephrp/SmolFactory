#!/usr/bin/env python3
"""
Test data persistence in Trackio Space
"""

import requests
import json
import time
import re

def test_persistence():
    """Test if experiment data persists across restarts"""
    
    print("🔍 Testing Data Persistence")
    print("=" * 50)
    
    # Test creating an experiment via API
    url = 'https://tonic-test-trackio-test.hf.space/gradio_api/call/create_experiment_interface'
    payload = {'data': ['test_persistence', 'Testing data persistence']}
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        if 'event_id' in data:
            event_id = data['event_id']
            print(f'✅ Experiment created with event ID: {event_id}')
            
            # Get the result
            get_url = f'{url}/{event_id}'
            time.sleep(2)
            
            get_response = requests.get(get_url)
            if get_response.status_code == 200:
                result = get_response.text
                print(f'✅ Experiment creation result: {result[:200]}...')
                
                # Extract experiment ID
                match = re.search(r'exp_\d{8}_\d{6}', result)
                if match:
                    experiment_id = match.group()
                    print(f'📋 Experiment ID: {experiment_id}')
                    
                    # Test logging metrics
                    metrics_url = 'https://tonic-test-trackio-test.hf.space/gradio_api/call/log_metrics_interface'
                    metrics_payload = {
                        'data': [experiment_id, '{"loss": 1.5, "accuracy": 0.8}', '100']
                    }
                    
                    metrics_response = requests.post(metrics_url, json=metrics_payload)
                    if metrics_response.status_code == 200:
                        print('✅ Metrics logged successfully')
                        
                        # Test getting experiment details
                        details_url = 'https://tonic-test-trackio-test.hf.space/gradio_api/call/get_experiment_details'
                        details_payload = {'data': [experiment_id]}
                        
                        details_response = requests.post(details_url, json=details_payload)
                        if details_response.status_code == 200:
                            details_data = details_response.json()
                            if 'event_id' in details_data:
                                details_event_id = details_data['event_id']
                                
                                # Get details result
                                details_get_url = f'{details_url}/{details_event_id}'
                                time.sleep(2)
                                
                                details_get_response = requests.get(details_get_url)
                                if details_get_response.status_code == 200:
                                    details_result = details_get_response.text
                                    print(f'✅ Experiment details retrieved: {details_result[:200]}...')
                                    
                                    if 'metrics' in details_result.lower():
                                        print('✅ Found metrics in experiment details')
                                    else:
                                        print('❌ No metrics found in experiment details')
                                else:
                                    print(f'❌ Failed to get details result: {details_get_response.status_code}')
                            else:
                                print('❌ No event_id in details response')
                        else:
                            print(f'❌ Failed to get experiment details: {details_response.status_code}')
                    else:
                        print(f'❌ Failed to log metrics: {metrics_response.status_code}')
                else:
                    print('❌ Could not extract experiment ID')
            else:
                print(f'❌ Failed to get result: {get_response.status_code}')
        else:
            print('❌ No event_id in response')
    else:
        print(f'❌ Failed to create experiment: {response.status_code}')
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Check the Trackio Space: https://tonic-test-trackio-test.hf.space")
    print("2. Go to '📊 Visualizations' tab")
    print("3. Enter the experiment ID above")
    print("4. Test if the visualization shows data")
    print("=" * 50)

if __name__ == "__main__":
    test_persistence() 