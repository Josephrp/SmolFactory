#!/usr/bin/env python3
"""
Test script to verify TrackioConfig update method works with keyword arguments
"""

import trackio

print("Testing TrackioConfig update method with keyword arguments...")

# Test that config exists and has update method
config = trackio.config
print(f"Config type: {type(config)}")
print(f"Has update method: {hasattr(config, 'update')}")

# Test update with keyword arguments (like TRL does)
print(f"Before update - project_name: {config.project_name}")
config.update(allow_val_change=True, project_name="test_project")
print(f"After update - project_name: {config.project_name}")
print(f"New attribute allow_val_change: {config.allow_val_change}")

# Test update with dictionary
test_data = {
    'experiment_name': 'test_experiment',
    'new_attribute': 'test_value'
}
config.update(test_data)
print(f"After dict update - experiment_name: {config.experiment_name}")
print(f"New attribute: {config.new_attribute}")

# Test update with both dictionary and keyword arguments
config.update({'another_attr': 'dict_value'}, kwarg_attr='keyword_value')
print(f"Another attr: {config.another_attr}")
print(f"Kwarg attr: {config.kwarg_attr}")

print("✅ Update method works correctly with keyword arguments!") 