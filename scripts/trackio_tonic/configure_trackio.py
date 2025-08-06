#!/usr/bin/env python3
"""
Configuration script for Trackio environment variables
"""

import os
import json
import subprocess
from datetime import datetime

def get_username_from_token(token: str) -> str:
    """Get username from HF token with fallback to CLI"""
    try:
        # Try API first
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        
        # Handle different possible response formats
        if isinstance(user_info, dict):
            # Try different possible keys for username
            username = (
                user_info.get('name') or 
                user_info.get('username') or 
                user_info.get('user') or 
                None
            )
        elif isinstance(user_info, str):
            # If whoami returns just the username as string
            username = user_info
        else:
            username = None
            
        if username:
            print(f"✅ Got username from API: {username}")
            return username
        else:
            print("⚠️  Could not get username from API, trying CLI...")
            return get_username_from_cli(token)
            
    except Exception as e:
        print(f"⚠️  API whoami failed: {e}")
        print("⚠️  Trying CLI fallback...")
        return get_username_from_cli(token)

def get_username_from_cli(token: str) -> str:
    """Fallback method to get username using CLI"""
    try:
        # Set HF token for CLI
        os.environ['HF_TOKEN'] = token
        
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

def configure_trackio():
    """Configure Trackio environment variables"""
    
    print("🔧 Trackio Configuration")
    print("=" * 40)
    
    # Get HF token (single token approach)
    hf_token = os.environ.get('HF_TOKEN')
    
    # Use the single HF_TOKEN
    active_token = hf_token
    
    if active_token:
        username = get_username_from_token(active_token)
        if username:
            print(f"✅ Authenticated as: {username}")
        else:
            print("⚠️  Could not determine username from token")
            username = 'unknown'
    else:
        username = 'unknown'
    
    # Use username in dataset repository if not specified
    dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', f'{username}/trackio-experiments')
    
    # Current configuration
    current_config = {
        'HF_TOKEN': hf_token or 'Not set',
        'TRACKIO_DATASET_REPO': dataset_repo,
        'SPACE_ID': os.environ.get('SPACE_ID', 'Not set'),
        'TRACKIO_URL': os.environ.get('TRACKIO_URL', 'Not set')
    }
    
    print("📋 Current Configuration:")
    for key, value in current_config.items():
        status = "✅" if value != "Not set" else "❌"
        print(f"   {status} {key}: {value}")
    
    print("\n🎯 Configuration Options:")
    print("1. Set HF_TOKEN - Main token (starts as write, switches to read after training)")
    print("2. Set TRACKIO_DATASET_REPO - Dataset repository (optional)")
    print("3. Set SPACE_ID - HF Space ID (auto-detected)")
    print("4. Set TRACKIO_URL - Trackio Space URL (auto-detected)")
    
    # Check if running on HF Spaces
    if os.environ.get('SPACE_ID'):
        print("\n🚀 Running on Hugging Face Spaces")
        print(f"   Space ID: {os.environ.get('SPACE_ID')}")
    
    # Validate configuration
    print("\n🔍 Configuration Validation:")
    
    # Check HF_TOKEN
    if current_config['HF_TOKEN'] != 'Not set':
        print("✅ HF_TOKEN is set")
        print("   This allows training operations and dataset access")
        print("   Note: Token will be automatically switched from write to read after training")
    else:
        print("❌ HF_TOKEN is not set")
        print("   Please set HF_TOKEN for training operations")
        print("   Get your token from: https://huggingface.co/settings/tokens")
    
    # Check dataset repository
    print(f"📊 Dataset Repository: {dataset_repo}")
    
    # Test dataset access if token is available
    test_token = current_config['HF_TOKEN']
    if test_token != 'Not set':
        print("\n🧪 Testing Dataset Access...")
        try:
            from datasets import load_dataset
            from huggingface_hub import HfApi
            
            # First check if the dataset repository exists
            api = HfApi(token=test_token)
            
            try:
                # Try to get repository info
                repo_info = api.repo_info(repo_id=dataset_repo, repo_type="dataset")
                print(f"✅ Dataset repository exists: {dataset_repo}")
                
                # Try to load the dataset
                dataset = load_dataset(dataset_repo, token=test_token)
                print(f"✅ Successfully loaded dataset: {dataset_repo}")
                
                # Show experiment count
                if 'train' in dataset:
                    experiment_count = len(dataset['train'])
                    print(f"📈 Found {experiment_count} experiments in dataset")
                    
                    # Show sample experiments
                    if experiment_count > 0:
                        print("🔬 Sample experiments:")
                        for i, row in enumerate(dataset['train'][:3]):  # Show first 3
                            exp_id = row.get('experiment_id', 'Unknown')
                            name = row.get('name', 'Unnamed')
                            print(f"   {i+1}. {exp_id}: {name}")
                
            except Exception as repo_error:
                if "404" in str(repo_error) or "not found" in str(repo_error).lower():
                    print(f"⚠️  Dataset repository '{dataset_repo}' doesn't exist yet")
                    print("   This is normal if you haven't created the dataset yet")
                    print("   Run setup_hf_dataset.py to create the dataset")
                else:
                    print(f"❌ Error accessing dataset repository: {repo_error}")
                    print("   Check that your token has read permissions")
                
        except ImportError:
            print("❌ Required packages not available")
            print("   Install with: pip install datasets huggingface_hub")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            print("   This might be normal if the dataset doesn't exist yet")
            print("   Run setup_hf_dataset.py to create the dataset")
    else:
        print("\n🧪 Dataset Access Test:")
        print("❌ Cannot test dataset access - no valid token set")
    
    # Generate configuration file
    config_file = "trackio_config.json"
    config_data = {
        'hf_token': current_config['HF_TOKEN'],
        'dataset_repo': current_config['TRACKIO_DATASET_REPO'],
        'space_id': current_config['SPACE_ID'],
        'trackio_url': current_config['TRACKIO_URL'],
        'username': username,
        'last_updated': datetime.now().isoformat(),
        'notes': 'Trackio configuration - HF_TOKEN starts as write token, switches to read token after training'
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n💾 Configuration saved to: {config_file}")
    
    # Show environment variable commands
    print("\n📝 Environment Variables for HF Space:")
    print("=" * 50)
    print(f"HF_TOKEN={current_config['HF_TOKEN']}")
    print(f"TRACKIO_DATASET_REPO={current_config['TRACKIO_DATASET_REPO']}")
    if current_config['TRACKIO_URL'] != 'Not set':
        print(f"TRACKIO_URL={current_config['TRACKIO_URL']}")
    
    print("\n🎯 Next Steps:")
    print("1. HF_TOKEN will be automatically set during deployment (starts as write token)")
    print("2. HF_TOKEN will be automatically switched to read token after training")
    print("3. Optionally set TRACKIO_DATASET_REPO to use a different dataset")
    print("4. Deploy your updated app.py to the Space")
    print("5. Run setup_hf_dataset.py if you haven't created the dataset yet")
    
    print("\n📚 Usage Examples")
    print("=" * 30)
    print("1. Default Dataset")
    print(f"   Repository: {username}/trackio-experiments")
    print("   Description: Default dataset for your experiments")
    print(f"   Set with: TRACKIO_DATASET_REPO={username}/trackio-experiments")
    print()
    print("2. Personal Dataset")
    print(f"   Repository: {username}/trackio-experiments")
    print("   Description: Your personal experiment dataset")
    print(f"   Set with: TRACKIO_DATASET_REPO={username}/trackio-experiments")
    print()
    print("3. Team Dataset")
    print("   Repository: your-org/team-experiments")
    print("   Description: Shared dataset for team experiments")
    print("   Set with: TRACKIO_DATASET_REPO=your-org/team-experiments")
    print()
    print("4. Project Dataset")
    print(f"   Repository: {username}/smollm3-experiments")
    print("   Description: Dataset specific to SmolLM3 experiments")
    print(f"   Set with: TRACKIO_DATASET_REPO={username}/smollm3-experiments")

def show_usage_examples():
    """Show usage examples for different dataset configurations"""
    examples = [
        {
            'name': 'Default Dataset',
            'repo': 'your-username/trackio-experiments',
            'description': 'Default dataset for your experiments',
            'env_var': 'TRACKIO_DATASET_REPO=your-username/trackio-experiments'
        },
        {
            'name': 'Personal Dataset',
            'repo': 'your-username/trackio-experiments',
            'description': 'Your personal experiment dataset',
            'env_var': 'TRACKIO_DATASET_REPO=your-username/trackio-experiments'
        },
        {
            'name': 'Team Dataset',
            'repo': 'your-org/team-experiments',
            'description': 'Shared dataset for team experiments',
            'env_var': 'TRACKIO_DATASET_REPO=your-org/team-experiments'
        },
        {
            'name': 'Project Dataset',
            'repo': 'your-username/smollm3-experiments',
            'description': 'Dataset specific to SmolLM3 experiments',
            'env_var': 'TRACKIO_DATASET_REPO=your-username/smollm3-experiments'
        }
    ]
    
    print("\n📚 Usage Examples")
    print("=" * 30)
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Repository: {example['repo']}")
        print(f"   Description: {example['description']}")
        print(f"   Set with: {example['env_var']}")
        print()

if __name__ == "__main__":
    configure_trackio() 