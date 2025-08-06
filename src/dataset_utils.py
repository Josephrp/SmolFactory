#!/usr/bin/env python3
"""
Dataset utilities for Trackio experiment data management
Provides functions for safe dataset operations with data preservation
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

class TrackioDatasetManager:
    """
    Manager class for Trackio experiment datasets with data preservation.
    
    This class ensures that existing experiment data is always preserved
    when adding new experiments or updating existing ones.
    """
    
    def __init__(self, dataset_repo: str, hf_token: str):
        """
        Initialize the dataset manager.
        
        Args:
            dataset_repo (str): HF dataset repository ID (e.g., "username/dataset-name")
            hf_token (str): Hugging Face token for authentication
        """
        self.dataset_repo = dataset_repo
        self.hf_token = hf_token
        self._validate_repo_format()
    
    def _validate_repo_format(self):
        """Validate dataset repository format"""
        if not self.dataset_repo or '/' not in self.dataset_repo:
            raise ValueError(f"Invalid dataset repository format: {self.dataset_repo}")
    
    def check_dataset_exists(self) -> bool:
        """
        Check if the dataset repository exists and is accessible.
        
        Returns:
            bool: True if dataset exists and is accessible, False otherwise
        """
        try:
            load_dataset(self.dataset_repo, token=self.hf_token)
            logger.info(f"✅ Dataset {self.dataset_repo} exists and is accessible")
            return True
        except Exception as e:
            logger.info(f"📊 Dataset {self.dataset_repo} doesn't exist or isn't accessible: {e}")
            return False
    
    def load_existing_experiments(self) -> List[Dict[str, Any]]:
        """
        Load all existing experiments from the dataset.
        
        Returns:
            List[Dict[str, Any]]: List of existing experiment dictionaries
        """
        try:
            if not self.check_dataset_exists():
                logger.info("📊 No existing dataset found, returning empty list")
                return []
            
            dataset = load_dataset(self.dataset_repo, token=self.hf_token)
            
            if 'train' not in dataset:
                logger.info("📊 No 'train' split found in dataset")
                return []
            
            experiments = list(dataset['train'])
            logger.info(f"📊 Loaded {len(experiments)} existing experiments")
            
            # Validate experiment structure
            valid_experiments = []
            for exp in experiments:
                if self._validate_experiment_structure(exp):
                    valid_experiments.append(exp)
                else:
                    logger.warning(f"⚠️ Skipping invalid experiment: {exp.get('experiment_id', 'unknown')}")
            
            logger.info(f"📊 {len(valid_experiments)} valid experiments loaded")
            return valid_experiments
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing experiments: {e}")
            return []
    
    def _validate_experiment_structure(self, experiment: Dict[str, Any]) -> bool:
        """
        Validate that an experiment has the required structure.
        
        Args:
            experiment (Dict[str, Any]): Experiment dictionary to validate
            
        Returns:
            bool: True if experiment structure is valid
        """
        required_fields = [
            'experiment_id', 'name', 'description', 'created_at', 
            'status', 'metrics', 'parameters', 'artifacts', 'logs'
        ]
        
        for field in required_fields:
            if field not in experiment:
                logger.warning(f"⚠️ Missing required field '{field}' in experiment")
                return False
        
        # Validate JSON fields
        json_fields = ['metrics', 'parameters', 'artifacts', 'logs']
        for field in json_fields:
            if isinstance(experiment[field], str):
                try:
                    json.loads(experiment[field])
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON in field '{field}' for experiment {experiment.get('experiment_id')}")
                    return False
        
        return True
    
    def save_experiments(self, experiments: List[Dict[str, Any]], commit_message: Optional[str] = None) -> bool:
        """
        Save a list of experiments to the dataset, preserving data integrity.
        
        Args:
            experiments (List[Dict[str, Any]]): List of experiment dictionaries
            commit_message (Optional[str]): Custom commit message
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if not experiments:
                logger.warning("⚠️ No experiments to save")
                return False
            
            # Validate all experiments before saving
            valid_experiments = []
            for exp in experiments:
                if self._validate_experiment_structure(exp):
                    # Ensure last_updated is set
                    if 'last_updated' not in exp:
                        exp['last_updated'] = datetime.now().isoformat()
                    valid_experiments.append(exp)
                else:
                    logger.error(f"❌ Invalid experiment structure: {exp.get('experiment_id', 'unknown')}")
                    return False
            
            # Create dataset
            dataset = Dataset.from_list(valid_experiments)
            
            # Generate commit message if not provided
            if not commit_message:
                commit_message = f"Update dataset with {len(valid_experiments)} experiments ({datetime.now().isoformat()})"
            
            # Push to hub
            dataset.push_to_hub(
                self.dataset_repo,
                token=self.hf_token,
                private=True,
                commit_message=commit_message
            )
            
            logger.info(f"✅ Successfully saved {len(valid_experiments)} experiments to {self.dataset_repo}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save experiments to dataset: {e}")
            return False
    
    def upsert_experiment(self, experiment: Dict[str, Any]) -> bool:
        """
        Insert a new experiment or update an existing one, preserving all other data.
        
        Args:
            experiment (Dict[str, Any]): Experiment dictionary to upsert
            
        Returns:
            bool: True if operation was successful, False otherwise
        """
        try:
            # Validate the experiment structure
            if not self._validate_experiment_structure(experiment):
                logger.error(f"❌ Invalid experiment structure for {experiment.get('experiment_id', 'unknown')}")
                return False
            
            # Load existing experiments
            existing_experiments = self.load_existing_experiments()
            
            # Find if experiment already exists
            experiment_id = experiment['experiment_id']
            experiment_found = False
            updated_experiments = []
            
            for existing_exp in existing_experiments:
                if existing_exp.get('experiment_id') == experiment_id:
                    # Update existing experiment
                    logger.info(f"🔄 Updating existing experiment: {experiment_id}")
                    experiment['last_updated'] = datetime.now().isoformat()
                    updated_experiments.append(experiment)
                    experiment_found = True
                else:
                    # Preserve existing experiment
                    updated_experiments.append(existing_exp)
            
            # If experiment doesn't exist, add it
            if not experiment_found:
                logger.info(f"➕ Adding new experiment: {experiment_id}")
                experiment['last_updated'] = datetime.now().isoformat()
                updated_experiments.append(experiment)
            
            # Save all experiments
            commit_message = f"{'Update' if experiment_found else 'Add'} experiment {experiment_id} (preserving {len(existing_experiments)} existing experiments)"
            
            return self.save_experiments(updated_experiments, commit_message)
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert experiment: {e}")
            return False
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific experiment by its ID.
        
        Args:
            experiment_id (str): The experiment ID to search for
            
        Returns:
            Optional[Dict[str, Any]]: The experiment dictionary if found, None otherwise
        """
        try:
            experiments = self.load_existing_experiments()
            
            for exp in experiments:
                if exp.get('experiment_id') == experiment_id:
                    logger.info(f"✅ Found experiment: {experiment_id}")
                    return exp
            
            logger.info(f"📊 Experiment not found: {experiment_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get experiment {experiment_id}: {e}")
            return None
    
    def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status.
        
        Args:
            status_filter (Optional[str]): Filter by experiment status (running, completed, failed, paused)
            
        Returns:
            List[Dict[str, Any]]: List of experiments matching the filter
        """
        try:
            experiments = self.load_existing_experiments()
            
            if status_filter:
                filtered_experiments = [exp for exp in experiments if exp.get('status') == status_filter]
                logger.info(f"📊 Found {len(filtered_experiments)} experiments with status '{status_filter}'")
                return filtered_experiments
            
            logger.info(f"📊 Found {len(experiments)} total experiments")
            return experiments
            
        except Exception as e:
            logger.error(f"❌ Failed to list experiments: {e}")
            return []
    
    def backup_dataset(self, backup_suffix: Optional[str] = None) -> str:
        """
        Create a backup of the current dataset.
        
        Args:
            backup_suffix (Optional[str]): Optional suffix for backup repo name
            
        Returns:
            str: Backup repository name if successful, empty string otherwise
        """
        try:
            if not backup_suffix:
                backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            backup_repo = f"{self.dataset_repo}-backup-{backup_suffix}"
            
            # Load current experiments
            experiments = self.load_existing_experiments()
            
            if not experiments:
                logger.warning("⚠️ No experiments to backup")
                return ""
            
            # Create backup dataset manager
            backup_manager = TrackioDatasetManager(backup_repo, self.hf_token)
            
            # Save to backup
            success = backup_manager.save_experiments(
                experiments, 
                f"Backup of {self.dataset_repo} created on {datetime.now().isoformat()}"
            )
            
            if success:
                logger.info(f"✅ Backup created: {backup_repo}")
                return backup_repo
            else:
                logger.error("❌ Failed to create backup")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return ""


def create_dataset_manager(dataset_repo: str, hf_token: str) -> TrackioDatasetManager:
    """
    Factory function to create a TrackioDatasetManager instance.
    
    Args:
        dataset_repo (str): HF dataset repository ID
        hf_token (str): Hugging Face token
        
    Returns:
        TrackioDatasetManager: Configured dataset manager instance
    """
    return TrackioDatasetManager(dataset_repo, hf_token)
