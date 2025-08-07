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
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

class TrackioSpaceDeployer:
    """Deployer for Trackio on Hugging Face Spaces"""
    
    def __init__(self, space_name: str, token: str, git_email: str = None, git_name: str = None, dataset_repo: str = None):
        self.space_name = space_name
        self.token = token
        self.dataset_repo = dataset_repo
        
        # Initialize HF API and get user info
        if HF_HUB_AVAILABLE:
            self.api = HfApi(token=self.token)
            # Get username from token
            try:
                user_info = self.api.whoami()
                # Handle different possible response formats
                if isinstance(user_info, dict):
                    # Try different possible keys for username
                    self.username = (
                        user_info.get('name') or 
                        user_info.get('username') or 
                        user_info.get('user') or 
                        'unknown'
                    )
                elif isinstance(user_info, str):
                    # If whoami returns just the username as string
                    self.username = user_info
                else:
                    # Fallback to CLI method
                    print("⚠️  Unexpected user_info format, trying CLI fallback...")
                    self.username = self._get_username_from_cli()
                
                if self.username and self.username != 'unknown':
                    print(f"✅ Authenticated as: {self.username}")
                else:
                    print("⚠️  Could not determine username from API, trying CLI...")
                    self.username = self._get_username_from_cli()
                    
            except Exception as e:
                print(f"❌ Failed to get user info from token: {e}")
                print("⚠️  Trying CLI fallback for username...")
                self.username = self._get_username_from_cli()
                if not self.username:
                    print("❌ Could not determine username. Please check your token.")
                    sys.exit(1)
        else:
            self.api = None
            self.username = self._get_username_from_cli()
        
        if not self.username:
            print("❌ Could not determine username. Please check your token.")
            sys.exit(1)
        
        self.space_url = f"https://huggingface.co/spaces/{self.username}/{self.space_name}"
        
        # Git configuration
        self.git_email = git_email or f"{self.username}@huggingface.co"
        self.git_name = git_name or self.username
    
    def _get_username_from_cli(self) -> str:
        """Fallback method to get username using CLI"""
        try:
            # Set HF token for CLI
            os.environ['HF_TOKEN'] = self.token
            
            # Get username using CLI
            result = subprocess.run(
                ["hf", "whoami"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                username = result.stdout.strip()
                if username:
                    print(f"✅ Got username from CLI: {username}")
                    return username
                else:
                    print("⚠️  CLI returned empty username")
                    return None
            else:
                print(f"⚠️  CLI whoami failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"⚠️  CLI fallback failed: {e}")
            return None
    
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
                "hf", "repo", "create",
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
                    "hf", "repo", "create",
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
            templates_dir = project_root / "templates" / "spaces" / "trackio"
            
            # Files to copy from templates/spaces/trackio
            files_to_copy = [
                "app.py",
                "requirements.txt",
                "README.md"
            ]
            
            # Copy files from templates/spaces/trackio to temp directory
            copied_files = []
            for file_name in files_to_copy:
                source_path = templates_dir / file_name
                dest_path = Path(temp_dir) / file_name
                
                if source_path.exists():
                    # For app.py, we need to customize it with user variables
                    if file_name == "app.py":
                        self._customize_app_py(source_path, dest_path)
                    else:
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
    
    def _customize_app_py(self, source_path: Path, dest_path: Path):
        """Customize app.py with user-specific variables"""
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace hardcoded values with user-specific ones
            replacements = {
                # Default dataset repository
                "'tonic/trackio-experiments'": f"'{self.dataset_repo or f'{self.username}/trackio-experiments'}'",
                "'trackio-experiments'": f"'{self.dataset_repo or f'{self.username}/trackio-experiments'}" if self.dataset_repo else "'trackio-experiments'",
                
                # Trackio URL
                "'https://tonic-test-trackio-test.hf.space'": f"'{self.space_url}'",
                "'https://your-trackio-space.hf.space'": f"'{self.space_url}'",
                
                # UI default values
                '"tonic/trackio-experiments"': f'"{self.dataset_repo or f"{self.username}/trackio-experiments"}"',
                '"trackio-experiments"': f'"{self.dataset_repo or f"{self.username}/trackio-experiments"}"' if self.dataset_repo else '"trackio-experiments"',
                
                # Examples in help text
                "'tonic/trackio-experiments'": f"'{self.username}/trackio-experiments'",
                "'your-username/trackio-experiments'": f"'{self.username}/trackio-experiments'",
                "'your-username/my-experiments'": f"'{self.username}/my-experiments'"
            }
            
            # Apply replacements
            for old, new in replacements.items():
                content = content.replace(old, new)
            
            # Write customized content
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Customized app.py with user variables")
            
        except Exception as e:
            print(f"❌ Error customizing app.py: {e}")
            # Fallback to copying original file
            shutil.copy2(source_path, dest_path)
    
    def upload_files_to_space(self, temp_dir: str) -> bool:
        """Upload files to the Space using HF Hub API directly"""
        try:
            print("Uploading files to Space using HF Hub API...")
            
            if not HF_HUB_AVAILABLE:
                print("❌ huggingface_hub not available for file upload")
                return False
            
            repo_id = f"{self.username}/{self.space_name}"
            
            # Upload each file using the HF Hub API
            temp_path = Path(temp_dir)
            uploaded_files = []
            
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    try:
                        # Upload file to the space
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=file_path.name,
                            repo_id=repo_id,
                            repo_type="space",
                            token=self.token
                        )
                        uploaded_files.append(file_path.name)
                        print(f"✅ Uploaded {file_path.name}")
                    except Exception as e:
                        print(f"❌ Failed to upload {file_path.name}: {e}")
                        return False
            
            print(f"✅ Successfully uploaded {len(uploaded_files)} files to Space")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading files: {e}")
            return False
    
    def set_space_secrets(self) -> bool:
        """Set environment variables/secrets for the Space using HF Hub API"""
        try:
            print("Setting Space secrets using HF Hub API...")
            
            if not HF_HUB_AVAILABLE:
                print("❌ huggingface_hub not available for setting secrets")
                return self._manual_secret_setup()
            
            repo_id = f"{self.username}/{self.space_name}"
            
            # Use the provided token as HF_TOKEN (starts as write token, will be switched to read token later)
            hf_token = self.token
            
            # Set the HF_TOKEN secret for the space using the API
            try:
                self.api.add_space_secret(
                    repo_id=repo_id,
                    key="HF_TOKEN",
                    value=hf_token,
                    description="Hugging Face token for dataset access (starts as write, switches to read)"
                )
                print("✅ Successfully set HF_TOKEN secret via API")
                
                # Set the TRACKIO_DATASET_REPO variable
                dataset_repo = self.dataset_repo or f"{self.username}/trackio-experiments"
                self.api.add_space_variable(
                    repo_id=repo_id,
                    key="TRACKIO_DATASET_REPO",
                    value=dataset_repo,
                    description="Dataset repository for Trackio experiments"
                )
                print(f"✅ Successfully set TRACKIO_DATASET_REPO variable: {dataset_repo}")
                
                # Set the TRACKIO_URL variable
                self.api.add_space_variable(
                    repo_id=repo_id,
                    key="TRACKIO_URL",
                    value=self.space_url,
                    description="Trackio Space URL for monitoring"
                )
                print(f"✅ Successfully set TRACKIO_URL variable: {self.space_url}")
                
                return True
                
            except Exception as api_error:
                print(f"❌ Failed to set secrets via API: {api_error}")
                print("Falling back to manual setup...")
                return self._manual_secret_setup()
            
        except Exception as e:
            print(f"❌ Error setting space secrets: {e}")
            return self._manual_secret_setup()
    
    def _manual_secret_setup(self) -> bool:
        """Fallback method for manual secret setup"""
        print("📝 Manual Space Secrets Configuration:")
        
        # Use the provided token as HF_TOKEN, but never display it
        hf_token = self.token
        print(f"   HF_TOKEN={'*' * 10}...hidden")
        
        dataset_repo = self.dataset_repo or f"{self.username}/trackio-experiments"
        print(f"   TRACKIO_DATASET_REPO={dataset_repo}")
        print(f"   TRACKIO_URL={self.space_url}")
        
        print("\n🔧 To set secrets in your Space:")
        print(f"1. Go to your Space settings: {self.space_url}/settings")
        print("2. Navigate to the 'Repository secrets' section")
        print("3. Add the following secrets:")
        print(f"   Name: HF_TOKEN")
        print(f"   Value: <your token>")
        print(f"   Name: TRACKIO_DATASET_REPO")
        print(f"   Value: {dataset_repo}")
        print(f"   Name: TRACKIO_URL")
        print(f"   Value: {self.space_url}")
        print("4. Save the secrets")
        print("\nNote: HF_TOKEN starts as write token and will be switched to read token after training")
        
        return True
    
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
        
        # Step 3: Upload files using HF Hub API
        if not self.upload_files_to_space(temp_dir):
            return False
        
        # Step 4: Set space secrets using API
        if not self.set_space_secrets():
            return False
        
        # Step 5: Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            print("✅ Cleaned up temporary directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp directory: {e}")
        
        # Step 6: Test space
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
    
    # Check if arguments are provided
    if len(sys.argv) >= 3:
        # Use command line arguments
        space_name = sys.argv[1]
        token = sys.argv[2]
        git_email = sys.argv[3] if len(sys.argv) > 3 else None
        git_name = sys.argv[4] if len(sys.argv) > 4 else None
        dataset_repo = sys.argv[5] if len(sys.argv) > 5 else None
        
        print(f"Using provided arguments:")
        print(f"  Space name: {space_name}")
        print(f"  Token: <hidden>")
        print(f"  Git email: {git_email or 'default'}")
        print(f"  Git name: {git_name or 'default'}")
        print(f"  Dataset repo: {dataset_repo or 'default'}")
    else:
        # Get user input (no username needed - will be extracted from token)
        space_name = input("Enter Space name (e.g., trackio-monitoring): ").strip()
        token = input("Enter your Hugging Face token: ").strip()
        
        # Get git configuration (optional)
        git_email = input("Enter your git email (optional, press Enter for default): ").strip()
        git_name = input("Enter your git name (optional, press Enter for default): ").strip()
        dataset_repo = input("Enter dataset repository (optional, press Enter for default): ").strip()
    
    if not space_name or not token:
        print("❌ Space name and token are required")
        sys.exit(1)
    
    # Use empty strings if not provided
    if not git_email:
        git_email = None
    if not git_name:
        git_name = None
    if not dataset_repo:
        dataset_repo = None
    
    # Create deployer (username will be extracted from token)
    deployer = TrackioSpaceDeployer(space_name, token, git_email, git_name, dataset_repo)
    
    # Run deployment
    success = deployer.deploy()
    
    if success:
        print("\n✅ Deployment successful!")
        print(f"🌐 Your Trackio Space: {deployer.space_url}")
        print(f"👤 Username: {deployer.username}")
        print(f"📊 Dataset Repository: {deployer.dataset_repo or f'{deployer.username}/trackio-experiments'}")
        print("\nNext steps:")
        print("1. Wait for the Space to build (usually 2-5 minutes)")
        print("2. Secrets have been automatically set via API")
        print("3. Test the interface by visiting the Space URL")
        print("4. Use the Space URL in your training scripts")
        print("\nIf the Space doesn't work immediately, check:")
        print("- The Space logs at the Space URL")
        print("- That all files were uploaded correctly")
        print("- That the HF token has write permissions")
        print("- That the secrets were set correctly in Space settings")
    else:
        print("\n❌ Deployment failed!")
        print("Check the error messages above and try again.")
        print("\nTroubleshooting:")
        print("1. Verify your HF token has write permissions")
        print("2. Check that the space name is available")
        print("3. Try creating the space manually on HF first")

if __name__ == "__main__":
    main() 