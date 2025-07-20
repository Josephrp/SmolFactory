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
                "--type", "space"
            ]
            
            # Try to create the space first
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Try alternative approach without space-specific flags
                print("Retrying with basic space creation...")
                cmd = [
                    "huggingface-cli", "repo", "create",
                    f"{self.username}/{self.space_name}"
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
            
            # Get the project root directory (3 levels up from this script)
            project_root = Path(__file__).parent.parent.parent
            templates_dir = project_root / "templates" / "spaces"
            
            # Files to upload from templates/spaces
            files_to_upload = [
                "app.py",
                "requirements.txt"
            ]
            
            # README.md will be created by configure_space method
            
            # Copy files from templates/spaces to current directory
            copied_files = []
            for file_name in files_to_upload:
                source_path = templates_dir / file_name
                if source_path.exists():
                    import shutil
                    shutil.copy2(source_path, file_name)
                    copied_files.append(file_name)
                    print(f"✅ Copied {file_name} from templates")
                else:
                    print(f"⚠️  File not found: {source_path}")
            
            # Check if we're in a git repository
            try:
                subprocess.run(["git", "status"], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                print("⚠️  Not in a git repository, initializing...")
                subprocess.run(["git", "init"], check=True)
                subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/spaces/{self.username}/{self.space_name}"], check=True)
            
            # Add all files at once
            existing_files = [f for f in files_to_upload if os.path.exists(f)]
            if existing_files:
                subprocess.run(["git", "add"] + existing_files, check=True)
                subprocess.run(["git", "add", "README.md"], check=True)  # Add README.md that was created in configure_space
                subprocess.run(["git", "commit", "-m", "Initial Space setup"], check=True)
                
                # Push to the space
                try:
                    subprocess.run(["git", "push", "origin", "main"], check=True)
                    print(f"✅ Uploaded {len(existing_files)} files")
                except subprocess.CalledProcessError:
                    # Try pushing to master branch if main doesn't exist
                    subprocess.run(["git", "push", "origin", "master"], check=True)
                    print(f"✅ Uploaded {len(existing_files)} files")
            else:
                print("⚠️  No files found to upload")
            
            return True
            
        except Exception as e:
            print(f"❌ Error uploading files: {e}")
            return False
    
    def configure_space(self) -> bool:
        """Configure the Space settings"""
        try:
            print("Configuring Space settings...")
            
            # Get the project root directory (3 levels up from this script)
            project_root = Path(__file__).parent.parent.parent
            templates_dir = project_root / "templates" / "spaces"
            readme_template_path = templates_dir / "README.md"
            
            # Read README template if it exists
            if readme_template_path.exists():
                with open(readme_template_path, 'r', encoding='utf-8') as f:
                    readme_template = f.read()
                
                # Replace placeholder with actual space URL
                readme_content = readme_template.replace("{SPACE_URL}", self.space_url)
                
                # Write README.md for the space
                with open("README.md", "w", encoding='utf-8') as f:
                    f.write(readme_content)
                
                print(f"✅ Created README.md from template")
            else:
                print(f"⚠️  README template not found: {readme_template_path}")
                # Fallback to basic README
                basic_readme = f"""---
title: Trackio Tonic
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

Visit: {self.space_url}
"""
                with open("README.md", "w", encoding='utf-8') as f:
                    f.write(basic_readme)
                print(f"✅ Created basic README.md")
            
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