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
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Import Hugging Face Hub API
try:
    from huggingface_hub import HfApi, create_repo
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

class TrackioSpaceDeployer:
    """Deployer for Trackio on Hugging Face Spaces"""
    
    def __init__(self, space_name: str, username: str, token: str):
        self.space_name = space_name
        self.username = username
        self.token = token
        self.space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
        
        # Initialize HF API
        if HF_HUB_AVAILABLE:
            self.api = HfApi(token=self.token)
        else:
            self.api = None
        
    def create_space(self) -> bool:
        """Create a new Hugging Face Space using the latest API"""
        try:
            print(f"Creating Space: {self.space_name}")
            
            if not HF_HUB_AVAILABLE:
                print("❌ huggingface_hub not available, falling back to CLI")
                return self._create_space_cli()
            
            # Use the latest HF Hub API to create space
            repo_id = f"{self.username}/{self.space_name}"
            
            try:
                # Create the space using the API
                create_repo(
                    repo_id=repo_id,
                    token=self.token,
                    repo_type="space",
                    exist_ok=True,
                    private=False,  # Spaces are typically public
                    space_sdk="gradio",  # Specify Gradio SDK
                    space_hardware="cpu-basic"  # Use basic CPU
                )
                
                print(f"✅ Space created successfully: {self.space_url}")
                return True
                
            except Exception as api_error:
                print(f"API creation failed: {api_error}")
                print("Falling back to CLI method...")
                return self._create_space_cli()
                
        except Exception as e:
            print(f"❌ Error creating space: {e}")
            return False
    
    def _create_space_cli(self) -> bool:
        """Fallback method using CLI commands"""
        try:
            print("Using CLI fallback method...")
            
            # Set HF token for CLI
            os.environ['HF_TOKEN'] = self.token
            
            # Create space using Hugging Face CLI
            cmd = [
                "huggingface-cli", "repo", "create",
                f"{self.username}/{self.space_name}",
                "--type", "space"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"First attempt failed: {result.stderr}")
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
            print(f"❌ Error creating space with CLI: {e}")
            return False
    
    def prepare_space_files(self) -> str:
        """Prepare all necessary files for the Space in a temporary directory"""
        try:
            print("Preparing Space files...")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            print(f"Created temporary directory: {temp_dir}")
            
            # Get the project root directory (3 levels up from this script)
            project_root = Path(__file__).parent.parent.parent
            templates_dir = project_root / "templates" / "spaces"
            
            # Files to copy from templates/spaces
            files_to_copy = [
                "app.py",
                "requirements.txt",
                "README.md"
            ]
            
            # Copy files from templates/spaces to temp directory
            copied_files = []
            for file_name in files_to_copy:
                source_path = templates_dir / file_name
                dest_path = Path(temp_dir) / file_name
                
                if source_path.exists():
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(file_name)
                    print(f"✅ Copied {file_name} to temp directory")
                else:
                    print(f"⚠️  File not found: {source_path}")
            
            # Update README.md with actual space URL
            readme_path = Path(temp_dir) / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                # Replace placeholder with actual space URL
                readme_content = readme_content.replace("{SPACE_URL}", self.space_url)
                
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                print(f"✅ Updated README.md with space URL")
            
            print(f"✅ Prepared {len(copied_files)} files in temporary directory")
            return temp_dir
            
        except Exception as e:
            print(f"❌ Error preparing files: {e}")
            return None
    
    def upload_files_to_space(self, temp_dir: str) -> bool:
        """Upload files to the Space using git"""
        try:
            print("Uploading files to Space...")
            
            # Change to temp directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            # Initialize git repository
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", f"https://huggingface.co/spaces/{self.username}/{self.space_name}"], check=True, capture_output=True)
            
            # Add all files
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial Trackio Space setup"], check=True, capture_output=True)
            
            # Push to the space
            try:
                subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
                print("✅ Pushed to main branch")
            except subprocess.CalledProcessError:
                # Try pushing to master branch if main doesn't exist
                subprocess.run(["git", "push", "origin", "master"], check=True, capture_output=True)
                print("✅ Pushed to master branch")
            
            # Return to original directory
            os.chdir(original_dir)
            
            return True
            
        except Exception as e:
            print(f"❌ Error uploading files: {e}")
            # Return to original directory
            os.chdir(original_dir)
            return False
    
    def test_space(self) -> bool:
        """Test if the Space is working correctly"""
        try:
            print("Testing Space...")
            
            # Wait a bit for the space to build
            import time
            print("Waiting 180 seconds for Space to build...")
            time.sleep(180)
            
            # Try to access the space
            response = requests.get(self.space_url, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Space is accessible: {self.space_url}")
                return True
            else:
                print(f"⚠️  Space returned status code: {response.status_code}")
                print(f"Response: {response.text[:500]}...")
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
        
        # Step 2: Prepare files
        temp_dir = self.prepare_space_files()
        if not temp_dir:
            return False
        
        # Step 3: Upload files
        if not self.upload_files_to_space(temp_dir):
            return False
        
        # Step 4: Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            print("✅ Cleaned up temporary directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp directory: {e}")
        
        # Step 5: Test space
        if not self.test_space():
            print("⚠️  Space created but may need more time to build")
            print("Please check the Space manually in a few minutes")
        
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
    token = input("Enter your Hugging Face token: ").strip()
    
    if not username or not space_name or not token:
        print("❌ Username, Space name, and token are required")
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
        print("\nIf the Space doesn't work immediately, check:")
        print("- The Space logs at the Space URL")
        print("- That all files were uploaded correctly")
        print("- That the HF token has write permissions")
    else:
        print("\n❌ Deployment failed!")
        print("Check the error messages above and try again.")
        print("\nTroubleshooting:")
        print("1. Verify your HF token has write permissions")
        print("2. Check that the space name is available")
        print("3. Try creating the space manually on HF first")

if __name__ == "__main__":
    main() 