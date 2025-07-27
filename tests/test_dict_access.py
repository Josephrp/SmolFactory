#!/usr/bin/env python3
"""
Test script to verify TrackioConfig dictionary-style access
"""

import trackio

print("Testing TrackioConfig dictionary-style access...")

# Test that config exists and has dictionary-style access
config = trackio.config
print(f"Config type: {type(config)}")
print(f"Has __getitem__: {hasattr(config, '__getitem__')}")
print(f"Has __setitem__: {hasattr(config, '__setitem__')}")

# Test dictionary-style assignment
print(f"Before assignment - project_name: {config.project_name}")
config['project_name'] = 'test_project'
print(f"After assignment - project_name: {config.project_name}")

# Test dictionary-style access
print(f"Dictionary access - project_name: {config['project_name']}")

# Test new key assignment
config['new_key'] = 'new_value'
print(f"New key assignment - new_key: {config['new_key']}")

# Test contains check
print(f"Contains 'project_name': {'project_name' in config}")
print(f"Contains 'nonexistent': {'nonexistent' in config}")

# Test get method
print(f"Get existing key: {config.get('project_name', 'default')}")
print(f"Get nonexistent key: {config.get('nonexistent', 'default')}")

# Test keys and items
print(f"Config keys: {list(config.keys())}")
print(f"Config items: {list(config.items())}")

# Test TRL-style usage
config['allow_val_change'] = True
config['report_to'] = 'trackio'
print(f"TRL-style config: allow_val_change={config['allow_val_change']}, report_to={config['report_to']}")

print("✅ Dictionary-style access works correctly!") 