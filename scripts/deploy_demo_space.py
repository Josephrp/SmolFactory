#!/usr/bin/env python3
"""
Demo Space Deployment Script
Deploys a Gradio demo space to Hugging Face Spaces for testing the fine-tuned model.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Import Hugging Face Hub API
try:
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import SmolLM3Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoSpaceDeployer:
    """Deploy demo space to Hugging Face Spaces"""
    
    def __init__(self, hf_token: str, hf_username: str, model_id: str, 
                 subfolder: str = "int4", space_name: Optional[str] = None):
        self.hf_token = hf_token
        self.hf_username = hf_username
        self.model_id = model_id
        self.subfolder = subfolder
        self.space_name = space_name or f"{model_id.split('/')[-1]}-demo"
        self.space_id = f"{hf_username}/{self.space_name}"
        self.space_url = f"https://huggingface.co/spaces/{self.space_id}"
        
        # Template paths
        self.template_dir = Path(__file__).parent.parent / "templates" / "spaces" / "demo"
        self.workspace_dir = Path.cwd()
        
        # Initialize HF API
        if HF_HUB_AVAILABLE:
            self.api = HfApi(token=self.hf_token)
        else:
            self.api = None
            logger.warning("huggingface_hub not available, using CLI fallback")
    
    def validate_model_exists(self) -> bool:
        """Validate that the model exists on Hugging Face Hub"""
        try:
            logger.info(f"Validating model: {self.model_id}")
            
            if HF_HUB_AVAILABLE:
                # Use HF Hub API
                try:
                    model_info = self.api.model_info(self.model_id)
                    logger.info(f"✅ Model {self.model_id} exists and is accessible")
                    return True
                except Exception as e:
                    logger.error(f"❌ Model {self.model_id} not found via API: {e}")
                    return False
            else:
                # Fallback to requests
                url = f"https://huggingface.co/api/models/{self.model_id}"
                headers = {"Authorization": f"Bearer {self.hf_token}"}
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"✅ Model {self.model_id} exists and is accessible")
                    return True
                else:
                    logger.error(f"❌ Model {self.model_id} not found or not accessible")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Error validating model: {e}")
            return False
    
    def create_space_repository(self) -> bool:
        """Create the space repository on Hugging Face Hub"""
        try:
            logger.info(f"Creating Space: {self.space_name}")
            
            if not HF_HUB_AVAILABLE:
                logger.warning("huggingface_hub not available, falling back to CLI")
                return self._create_space_cli()
            
            # Use the latest HF Hub API to create space
            try:
                # Create the space using the API
                create_repo(
                    repo_id=self.space_id,
                    token=self.hf_token,
                    repo_type="space",
                    exist_ok=True,
                    private=False,  # Spaces are typically public
                    space_sdk="gradio",  # Specify Gradio SDK
                    space_hardware="cpu-basic"  # Use basic CPU
                )
                
                logger.info(f"✅ Space created successfully: {self.space_url}")
                return True
                
            except Exception as api_error:
                logger.error(f"API creation failed: {api_error}")
                logger.info("Falling back to CLI method...")
                return self._create_space_cli()
                
        except Exception as e:
            logger.error(f"❌ Error creating space: {e}")
            return False
    
    def _create_space_cli(self) -> bool:
        """Fallback method using CLI commands"""
        try:
            logger.info("Using CLI fallback method...")
            
            # Set HF token for CLI
            os.environ['HF_TOKEN'] = self.hf_token
            
            # Create space using Hugging Face CLI
            cmd = [
                "hf", "repo", "create",
                self.space_id,
                "--type", "space"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"First attempt failed: {result.stderr}")
                # Try alternative approach without space-specific flags
                logger.info("Retrying with basic space creation...")
                cmd = [
                    "hf", "repo", "create",
                    self.space_id
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Space created successfully: {self.space_url}")
                return True
            else:
                logger.error(f"❌ Failed to create space: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating space with CLI: {e}")
            return False
    
    def prepare_space_files(self) -> str:
        """Prepare all necessary files for the Space in a temporary directory"""
        try:
            logger.info("Preparing Space files...")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Copy template files
            copied_files = []
            for file_path in self.template_dir.iterdir():
                if file_path.is_file():
                    dest_path = Path(temp_dir) / file_path.name
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
                    logger.info(f"✅ Copied {file_path.name} to temp directory")
            
            # Update app.py with environment variables
            app_file = Path(temp_dir) / "app.py"
            if app_file.exists():
                with open(app_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add environment variable setup at the top
                env_setup = f"""
# Environment variables for model configuration
import os
os.environ['HF_MODEL_ID'] = '{self.model_id}'
os.environ['MODEL_SUBFOLDER'] = '{self.subfolder if self.subfolder else ""}'
os.environ['MODEL_NAME'] = '{self.model_id.split("/")[-1]}'

"""
                
                # Insert after imports
                lines = content.split('\n')
                import_end = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_end = i + 1
                    elif line.strip() == '' and import_end > 0:
                        break
                
                lines.insert(import_end, env_setup)
                content = '\n'.join(lines)
                
                with open(app_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("✅ Updated app.py with model configuration")
            
            # Create README.md for the space
            readme_content = f"""# Demo: {self.model_id}

This is an interactive demo for the fine-tuned model {self.model_id}.

## Features
- Interactive chat interface
- Customizable system prompts
- Advanced generation parameters
- Thinking mode support

## Model Information
- **Model ID**: {self.model_id}
- **Subfolder**: {self.subfolder if self.subfolder and self.subfolder.strip() else "main"}
- **Deployed by**: {self.hf_username}

## Usage
Simply start chatting with the model using the interface below!

---
*This demo was automatically deployed by the SmolLM3 Fine-tuning Pipeline*
"""
            
            with open(Path(temp_dir) / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info(f"✅ Prepared {len(copied_files)} files in temporary directory")
            return temp_dir
            
        except Exception as e:
            logger.error(f"❌ Error preparing files: {e}")
            return None
    
    def upload_files_to_space(self, temp_dir: str) -> bool:
        """Upload files to the Space using HF Hub API directly"""
        try:
            logger.info("Uploading files to Space using HF Hub API...")
            
            if not HF_HUB_AVAILABLE:
                logger.error("❌ huggingface_hub not available for file upload")
                return self._upload_files_cli(temp_dir)
            
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
                            repo_id=self.space_id,
                            repo_type="space",
                            token=self.hf_token
                        )
                        uploaded_files.append(file_path.name)
                        logger.info(f"✅ Uploaded {file_path.name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to upload {file_path.name}: {e}")
                        return False
            
            logger.info(f"✅ Successfully uploaded {len(uploaded_files)} files to Space")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading files: {e}")
            return self._upload_files_cli(temp_dir)
    
    def _upload_files_cli(self, temp_dir: str) -> bool:
        """Fallback method using CLI for file upload"""
        try:
            logger.info("Using CLI fallback for file upload...")
            
            # Set HF token for CLI
            os.environ['HF_TOKEN'] = self.hf_token
            
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(["git", "config", "user.name", "Demo Deployer"], cwd=temp_dir, check=True)
            subprocess.run(["git", "config", "user.email", "demo@example.com"], cwd=temp_dir, check=True)
            
            # Add files
            subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
            subprocess.run(["git", "commit", "-m", f"Deploy demo for {self.model_id}"], cwd=temp_dir, check=True)
            
            # Add remote and push
            remote_url = f"https://{self.hf_token}@huggingface.co/spaces/{self.space_id}"
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=temp_dir, check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=temp_dir, check=True)
            
            logger.info(f"✅ Successfully pushed files to space: {self.space_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git operation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error pushing to space: {e}")
            return False
    
    def set_space_secrets(self) -> bool:
        """Set environment variables/secrets for the Space using HF Hub API"""
        try:
            logger.info("Setting Space secrets using HF Hub API...")
            
            if not HF_HUB_AVAILABLE:
                logger.warning("❌ huggingface_hub not available for setting secrets")
                return self._manual_secret_setup()
            
            # Set the HF_TOKEN secret for the space using the API
            try:
                self.api.add_space_secret(
                    repo_id=self.space_id,
                    key="HF_TOKEN",
                    value=self.hf_token,
                    description="Hugging Face token for model access"
                )
                logger.info("✅ Successfully set HF_TOKEN secret via API")
                
                # Set model-specific environment variables
                self.api.add_space_variable(
                    repo_id=self.space_id,
                    key="HF_MODEL_ID",
                    value=self.model_id,
                    description="Model ID for the demo"
                )
                logger.info(f"✅ Successfully set HF_MODEL_ID variable: {self.model_id}")
                
                if self.subfolder and self.subfolder.strip():
                    self.api.add_space_variable(
                        repo_id=self.space_id,
                        key="MODEL_SUBFOLDER",
                        value=self.subfolder,
                        description="Model subfolder for the demo"
                    )
                    logger.info(f"✅ Successfully set MODEL_SUBFOLDER variable: {self.subfolder}")
                else:
                    logger.info("ℹ️ No subfolder specified, using main model")
                
                return True
                
            except Exception as api_error:
                logger.error(f"❌ Failed to set secrets via API: {api_error}")
                logger.info("Falling back to manual setup...")
                return self._manual_secret_setup()
            
        except Exception as e:
            logger.error(f"❌ Error setting space secrets: {e}")
            return self._manual_secret_setup()
    
    def _manual_secret_setup(self) -> bool:
        """Fallback method for manual secret setup"""
        logger.info("📝 Manual Space Secrets Configuration:")
        logger.info(f"   HF_TOKEN={self.hf_token}")
        logger.info(f"   HF_MODEL_ID={self.model_id}")
        if self.subfolder and self.subfolder.strip():
            logger.info(f"   MODEL_SUBFOLDER={self.subfolder}")
        else:
            logger.info("   MODEL_SUBFOLDER=(empty - using main model)")
        
        logger.info(f"\n🔧 To set secrets in your Space:")
        logger.info(f"1. Go to your Space settings: {self.space_url}/settings")
        logger.info("2. Navigate to the 'Repository secrets' section")
        logger.info("3. Add the following secrets:")
        logger.info(f"   Name: HF_TOKEN")
        logger.info(f"   Value: {self.hf_token}")
        logger.info(f"   Name: HF_MODEL_ID")
        logger.info(f"   Value: {self.model_id}")
        if self.subfolder and self.subfolder.strip():
            logger.info(f"   Name: MODEL_SUBFOLDER")
            logger.info(f"   Value: {self.subfolder}")
        else:
            logger.info("   Name: MODEL_SUBFOLDER")
            logger.info("   Value: (leave empty)")
        logger.info("4. Save the secrets")
        
        return True
    
    def test_space(self) -> bool:
        """Test if the Space is working correctly"""
        try:
            logger.info("Testing Space...")
            
            # Wait a bit for the space to build
            logger.info("Waiting 180 seconds for Space to build...")
            time.sleep(180)
            
            # Try to access the space
            response = requests.get(self.space_url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Space is accessible: {self.space_url}")
                return True
            else:
                logger.warning(f"⚠️  Space returned status code: {response.status_code}")
                logger.warning(f"Response: {response.text[:500]}...")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing space: {e}")
            return False
    
    def deploy(self) -> bool:
        """Main deployment method"""
        logger.info(f"🚀 Starting demo space deployment for {self.model_id}")
        
        # Step 1: Validate model exists
        if not self.validate_model_exists():
            return False
        
        # Step 2: Create space repository
        if not self.create_space_repository():
            return False
        
        # Step 3: Prepare files
        temp_dir = self.prepare_space_files()
        if not temp_dir:
            return False
        
        # Step 4: Upload files
        if not self.upload_files_to_space(temp_dir):
            return False
        
        # Step 5: Set space secrets
        if not self.set_space_secrets():
            return False
        
        # Step 6: Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            logger.info("✅ Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not clean up temp directory: {e}")
        
        # Step 7: Test space
        if not self.test_space():
            logger.warning("⚠️  Space created but may need more time to build")
            logger.info("Please check the Space manually in a few minutes")
        
        logger.info(f"🎉 Demo space deployment completed!")
        logger.info(f"📊 Space URL: {self.space_url}")
        logger.info(f"🔧 Space configuration: {self.space_url}/settings")
        
        return True

def main():
    """Main function for command line usage"""
    print("Demo Space Deployment Script")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description="Deploy demo space to Hugging Face Spaces")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token")
    parser.add_argument("--hf-username", required=True, help="Hugging Face username")
    parser.add_argument("--model-id", required=True, help="Model ID to deploy demo for")
    parser.add_argument("--subfolder", default="int4", help="Model subfolder (default: int4)")
    parser.add_argument("--space-name", help="Custom space name (optional)")
    
    args = parser.parse_args()
    
    deployer = DemoSpaceDeployer(
        hf_token=args.hf_token,
        hf_username=args.hf_username,
        model_id=args.model_id,
        subfolder=args.subfolder,
        space_name=args.space_name
    )
    
    success = deployer.deploy()
    
    if success:
        print("\n✅ Deployment successful!")
        print(f"🌐 Your Demo Space: {deployer.space_url}")
        print(f"👤 Username: {deployer.hf_username}")
        print(f"🤖 Model: {deployer.model_id}")
        print("\nNext steps:")
        print("1. Wait for the Space to build (usually 2-5 minutes)")
        print("2. Secrets have been automatically set via API")
        print("3. Test the interface by visiting the Space URL")
        print("4. Share your demo with others!")
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
        print("3. Verify the model exists and is accessible")
        print("4. Try creating the space manually on HF first")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 