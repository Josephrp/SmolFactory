#!/usr/bin/env python3
"""
Simple test for TrackioConfig update method fix
"""

import trackio

print("Testing TrackioConfig update method...")

# Test that config exists and has update method
config = trackio.config
print(f"Config type: {type(config)}")
print(f"Has update method: {hasattr(config, 'update')}")

# Test update functionality
test_data = {
    'project_name': 'test_project',
    'experiment_name': 'test_experiment',
    'new_attribute': 'test_value'
}

print(f"Before update - project_name: {config.project_name}")
config.update(test_data)
print(f"After update - project_name: {config.project_name}")
print(f"New attribute: {config.new_attribute}")

print("✅ Update method works correctly!") 