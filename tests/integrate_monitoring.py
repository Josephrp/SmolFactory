#!/usr/bin/env python3
"""
Script to integrate improved monitoring with HF Datasets into training scripts
"""

import os
import sys
import re
from pathlib import Path

def update_training_script(script_path: str):
    """Update a training script to include improved monitoring"""
    
    print(f"🔧 Updating {script_path}...")
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if monitoring is already imported
    if 'from monitoring import' in content:
        print(f"  ⚠️  Monitoring already imported in {script_path}")
        return False
    
    # Add monitoring import
    import_pattern = r'(from \w+ import.*?)(\n\n|\n$)'
    match = re.search(import_pattern, content, re.MULTILINE | re.DOTALL)
    
    if match:
        # Add monitoring import after existing imports
        new_import = match.group(1) + '\nfrom monitoring import create_monitor_from_config\n' + match.group(2)
        content = content.replace(match.group(0), new_import)
    else:
        # Add at the beginning if no imports found
        content = 'from monitoring import create_monitor_from_config\n\n' + content
    
    # Find the main training function and add monitoring
    # Look for patterns like "def main():" or "def train():"
    main_patterns = [
        r'def main\(\):',
        r'def train\(\):',
        r'def run_training\(\):'
    ]
    
    monitoring_added = False
    for pattern in main_patterns:
        if re.search(pattern, content):
            # Add monitoring initialization after config loading
            config_pattern = r'(config\s*=\s*get_config\([^)]+\))'
            config_match = re.search(config_pattern, content)
            
            if config_match:
                monitoring_code = '''
    # Initialize monitoring
    monitor = None
    if config.enable_tracking:
        try:
            monitor = create_monitor_from_config(config, getattr(config, 'experiment_name', None))
            logger.info(f"✅ Monitoring initialized for experiment: {monitor.experiment_name}")
            logger.info(f"📊 Dataset repository: {monitor.dataset_repo}")
            
            # Log configuration
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
            monitor.log_configuration(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            logger.warning("Continuing without monitoring...")
'''
                
                # Insert monitoring code after config loading
                insert_point = config_match.end()
                content = content[:insert_point] + monitoring_code + content[insert_point:]
                
                # Add monitoring callback to trainer
                trainer_pattern = r'(trainer\s*=\s*[^)]+\))'
                trainer_match = re.search(trainer_pattern, content)
                
                if trainer_match:
                    callback_code = '''
    # Add monitoring callback if available
    if monitor:
        try:
            callback = monitor.create_monitoring_callback()
            trainer.add_callback(callback)
            logger.info("✅ Monitoring callback added to trainer")
        except Exception as e:
            logger.error(f"Failed to add monitoring callback: {e}")
'''
                    
                    insert_point = trainer_match.end()
                    content = content[:insert_point] + callback_code + content[insert_point:]
                
                # Add training summary logging
                train_pattern = r'(trainer\.train\(\))'
                train_match = re.search(train_pattern, content)
                
                if train_match:
                    summary_code = '''
        # Log training summary
        if monitor:
            try:
                summary = {
                    'final_loss': getattr(trainer, 'final_loss', None),
                    'total_steps': getattr(trainer, 'total_steps', None),
                    'training_duration': getattr(trainer, 'training_duration', None),
                    'model_path': output_path,
                    'config_file': config_path
                }
                monitor.log_training_summary(summary)
                logger.info("✅ Training summary logged")
            except Exception as e:
                logger.error(f"Failed to log training summary: {e}")
'''
                    
                    # Find the training call and add summary after it
                    train_call_pattern = r'(trainer\.train\(\)\s*\n\s*logger\.info\("Training completed successfully!"\))'
                    train_call_match = re.search(train_call_pattern, content)
                    
                    if train_call_match:
                        insert_point = train_call_match.end()
                        content = content[:insert_point] + summary_code + content[insert_point:]
                
                # Add error handling and cleanup
                error_pattern = r'(except Exception as e:\s*\n\s*logger\.error\(f"Training failed: {e}"\)\s*\n\s*raise)'
                error_match = re.search(error_pattern, content)
                
                if error_match:
                    error_code = '''
        # Log error to monitoring
        if monitor:
            try:
                error_summary = {
                    'error': str(e),
                    'status': 'failed',
                    'model_path': output_path,
                    'config_file': config_path
                }
                monitor.log_training_summary(error_summary)
            except Exception as log_error:
                logger.error(f"Failed to log error to monitoring: {log_error}")
'''
                    
                    insert_point = error_match.end()
                    content = content[:insert_point] + error_code + content[insert_point:]
                
                # Add finally block for cleanup
                finally_pattern = r'(raise\s*\n\s*if __name__ == \'__main__\':)'
                finally_match = re.search(finally_pattern, content)
                
                if finally_match:
                    cleanup_code = '''
    finally:
        # Close monitoring
        if monitor:
            try:
                monitor.close()
                logger.info("✅ Monitoring session closed")
            except Exception as e:
                logger.error(f"Failed to close monitoring: {e}")

'''
                    
                    insert_point = finally_match.start()
                    content = content[:insert_point] + cleanup_code + content[insert_point:]
                
                monitoring_added = True
                break
    
    if monitoring_added:
        # Write updated content
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Updated {script_path} with monitoring integration")
        return True
    else:
        print(f"  ⚠️  Could not find main training function in {script_path}")
        return False

def update_config_files():
    """Update configuration files to include HF Datasets support"""
    
    config_dir = Path("config")
    config_files = list(config_dir.glob("*.py"))
    
    print(f"🔧 Updating configuration files...")
    
    for config_file in config_files:
        if config_file.name.startswith("__"):
            continue
            
        print(f"  📝 Checking {config_file.name}...")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if HF Datasets config is already present
        if 'TRACKIO_DATASET_REPO' in content:
            print(f"    ⚠️  HF Datasets config already present in {config_file.name}")
            continue
        
        # Add HF Datasets configuration
        trackio_pattern = r'(# Trackio monitoring configuration.*?experiment_name: Optional\[str\] = None)'
        trackio_match = re.search(trackio_pattern, content, re.DOTALL)
        
        if trackio_match:
            hf_config = '''
    # HF Datasets configuration
    hf_token: Optional[str] = None
    dataset_repo: Optional[str] = None
'''
            
            insert_point = trackio_match.end()
            content = content[:insert_point] + hf_config + content[insert_point:]
            
            # Write updated content
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"    ✅ Added HF Datasets config to {config_file.name}")
        else:
            print(f"    ⚠️  Could not find Trackio config section in {config_file.name}")

def main():
    """Main function to integrate monitoring into all training scripts"""
    
    print("🚀 Integrating improved monitoring with HF Datasets...")
    print("=" * 60)
    
    # Update main training script
    main_script = "train.py"
    if os.path.exists(main_script):
        update_training_script(main_script)
    else:
        print(f"⚠️  Main training script {main_script} not found")
    
    # Update configuration files
    update_config_files()
    
    # Update any other training scripts in config directory
    config_dir = Path("config")
    training_scripts = [
        "train_smollm3_openhermes_fr.py",
        "train_smollm3_openhermes_fr_a100_balanced.py",
        "train_smollm3_openhermes_fr_a100_large.py",
        "train_smollm3_openhermes_fr_a100_max_performance.py",
        "train_smollm3_openhermes_fr_a100_multiple_passes.py"
    ]
    
    print(f"\n🔧 Updating training scripts in config directory...")
    
    for script_name in training_scripts:
        script_path = config_dir / script_name
        if script_path.exists():
            update_training_script(str(script_path))
        else:
            print(f"  ⚠️  Training script {script_name} not found")
    
    print(f"\n✅ Monitoring integration completed!")
    print(f"\n📋 Next steps:")
    print(f"1. Set HF_TOKEN environment variable")
    print(f"2. Optionally set TRACKIO_DATASET_REPO")
    print(f"3. Run your training scripts with monitoring enabled")
    print(f"4. Check your HF Dataset repository for experiment data")

if __name__ == "__main__":
    main() 