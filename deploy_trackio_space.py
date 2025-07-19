#!/usr/bin/env python3
"""
Deployment script for Trackio on Hugging Face Spaces
Automates the process of creating and configuring a Trackio Space
"""

import os
import json
import requests
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class TrackioSpaceDeployer:
    """Deployer for Trackio on Hugging Face Spaces"""
    
    def __init__(self, space_name: str, username: str, token: str):
        self.space_name = space_name
        self.username = username
        self.token = token
        self.space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
        
    def create_space(self) -> bool:
        """Create a new Hugging Face Space"""
        try:
            print(f"Creating Space: {self.space_name}")
            
            # Create space using Hugging Face CLI
            cmd = [
                "huggingface-cli", "repo", "create",
                f"{self.username}/{self.space_name}",
                "--type", "space",
                "--space-sdk", "gradio",
                "--space-hardware", "cpu-basic"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Space created successfully: {self.space_url}")
                return True
            else:
                print(f"❌ Failed to create space: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating space: {e}")
            return False
    
    def upload_files(self) -> bool:
        """Upload necessary files to the Space"""
        try:
            print("Uploading files to Space...")
            
            # Files to upload
            files_to_upload = [
                "app.py",
                "requirements_space.txt",
                "README.md"
            ]
            
            for file_path in files_to_upload:
                if os.path.exists(file_path):
                    # Use git to add and push files
                    subprocess.run(["git", "add", file_path], check=True)
                    subprocess.run(["git", "commit", "-m", f"Add {file_path}"], check=True)
                    subprocess.run(["git", "push"], check=True)
                    print(f"✅ Uploaded {file_path}")
                else:
                    print(f"⚠️  File not found: {file_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error uploading files: {e}")
            return False
    
    def configure_space(self) -> bool:
        """Configure the Space settings"""
        try:
            print("Configuring Space settings...")
            
            # Create space configuration
            space_config = {
                "title": "Trackio - Experiment Tracking",
                "emoji": "🚀",
                "colorFrom": "blue",
                "colorTo": "purple",
                "sdk": "gradio",
                "sdk_version": "4.0.0",
                "app_file": "app.py",
                "pinned": False
            }
            
            # Write README.md for the space
            space_readme = f"""---
title: Trackio for Petite Elle L'Aime
emoji: 🐠
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: true
license: mit
short_description: trackio for training monitoring
---

# Trackio Experiment Tracking

A Gradio interface for experiment tracking and monitoring.

## Features

- Create and manage experiments
- Log training metrics and parameters
- View experiment details and results
- Update experiment status

## Usage

1. Create a new experiment using the "Create Experiment" tab
2. Log metrics during training using the "Log Metrics" tab
3. View experiment details using the "View Experiments" tab
4. Update experiment status using the "Update Status" tab

## Integration

To connect your training script to this Trackio Space:

```python
from monitoring import SmolLM3Monitor

monitor = SmolLM3Monitor(
    experiment_name="my_experiment",
    trackio_url="{self.space_url}",
    enable_tracking=True
)
```

Visit: {self.space_url}
"""
            
            with open("README.md", "w") as f:
                f.write(space_readme)
            
            return True
            
        except Exception as e:
            print(f"❌ Error configuring space: {e}")
            return False
    
    def test_space(self) -> bool:
        """Test if the Space is working correctly"""
        try:
            print("Testing Space...")
            
            # Wait a bit for the space to build
            import time
            time.sleep(30)
            
            # Try to access the space
            response = requests.get(self.space_url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Space is accessible: {self.space_url}")
                return True
            else:
                print(f"⚠️  Space returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing space: {e}")
            return False
    
    def deploy(self) -> bool:
        """Complete deployment process"""
        print("🚀 Starting Trackio Space deployment...")
        
        # Step 1: Create space
        if not self.create_space():
            return False
        
        # Step 2: Configure space
        if not self.configure_space():
            return False
        
        # Step 3: Upload files
        if not self.upload_files():
            return False
        
        # Step 4: Test space
        if not self.test_space():
            print("⚠️  Space created but may need time to build")
        
        print(f"🎉 Deployment completed!")
        print(f"📊 Trackio Space URL: {self.space_url}")
        print(f"🔧 Space configuration: {self.space_url}/settings")
        
        return True

def main():
    """Main deployment function"""
    print("Trackio Space Deployment Script")
    print("=" * 40)
    
    # Get user input
    username = input("Enter your Hugging Face username: ").strip()
    space_name = input("Enter Space name (e.g., trackio-monitoring): ").strip()
    token = input("Enter your Hugging Face token (optional): ").strip()
    
    if not username or not space_name:
        print("❌ Username and Space name are required")
        sys.exit(1)
    
    # Create deployer
    deployer = TrackioSpaceDeployer(space_name, username, token)
    
    # Run deployment
    success = deployer.deploy()
    
    if success:
        print("\n✅ Deployment successful!")
        print(f"🌐 Your Trackio Space: {deployer.space_url}")
        print("\nNext steps:")
        print("1. Wait for the Space to build (usually 2-5 minutes)")
        print("2. Test the interface by visiting the Space URL")
        print("3. Use the Space URL in your training scripts")
    else:
        print("\n❌ Deployment failed!")
        print("Check the error messages above and try again.")

if __name__ == "__main__":
    main() 