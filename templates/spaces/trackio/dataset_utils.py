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
            # Try standard load first
            load_dataset(self.dataset_repo, token=self.hf_token)
            logger.info(f"✅ Dataset {self.dataset_repo} exists and is accessible")
            return True
        except Exception as e:
            # Retry with relaxed verification to handle split metadata mismatches
            try:
                logger.info(f"📊 Standard load failed: {e}. Retrying with relaxed verification...")
                load_dataset(
                    self.dataset_repo,
                    token=self.hf_token,
                    verification_mode="no_checks"  # type: ignore[arg-type]
                )
                logger.info(f"✅ Dataset {self.dataset_repo} accessible with relaxed verification")
                return True
            except Exception as e2:
                logger.info(f"📊 Dataset {self.dataset_repo} doesn't exist or isn't accessible: {e2}")
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
            
            # Load with relaxed verification to avoid split-metadata mismatches blocking reads
            try:
                dataset = load_dataset(self.dataset_repo, token=self.hf_token)
            except Exception:
                dataset = load_dataset(self.dataset_repo, token=self.hf_token, verification_mode="no_checks")  # type: ignore[arg-type]
            
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
        Validate and SANITIZE an experiment structure to prevent destructive failures.

        - Requires 'experiment_id'; otherwise skip the row.
        - Fills defaults for missing non-JSON fields.
        - Normalizes JSON fields to valid JSON strings.
        """
        if not experiment.get('experiment_id'):
            logger.warning("⚠️ Missing required field 'experiment_id' in experiment; skipping row")
            return False

        defaults = {
            'name': '',
            'description': '',
            'created_at': datetime.now().isoformat(),
            'status': 'running',
        }
        for key, default_value in defaults.items():
            if experiment.get(key) in (None, ''):
                experiment[key] = default_value

        def _ensure_json_string(field_name: str, default_value: Any):
            raw_value = experiment.get(field_name)
            try:
                if isinstance(raw_value, str):
                    if raw_value.strip() == '':
                        experiment[field_name] = json.dumps(default_value, default=str)
                    else:
                        json.loads(raw_value)
                else:
                    experiment[field_name] = json.dumps(
                        raw_value if raw_value is not None else default_value,
                        default=str
                    )
            except Exception:
                experiment[field_name] = json.dumps(default_value, default=str)

        for json_field, default in (('metrics', []), ('parameters', {}), ('artifacts', []), ('logs', [])):
            _ensure_json_string(json_field, default)

        return True
    
    def save_experiments(self, experiments: List[Dict[str, Any]], commit_message: Optional[str] = None) -> bool:
        """
        Save a list of experiments to the dataset using a non-destructive union merge.

        - Loads existing experiments (if any) and builds a union by `experiment_id`.
        - For overlapping IDs, merges JSON fields:
          - metrics: concatenates lists and de-duplicates by (step, timestamp) for nested entries
          - parameters: dict-update (new values override)
          - artifacts: union with de-dup
          - logs: concatenation with de-dup
        - Non-JSON scalar fields from incoming experiments take precedence.

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
            
            # Helpers
            def _parse_json_field(value, default):
                try:
                    if value is None:
                        return default
                    if isinstance(value, str):
                        return json.loads(value) if value else default
                    return value
                except Exception:
                    return default
            
            def _metrics_key(entry: Dict[str, Any]):
                if isinstance(entry, dict):
                    return (entry.get('step'), entry.get('timestamp'))
                return (None, json.dumps(entry, sort_keys=True))
            
            # Load existing experiments for union merge
            existing = {}
            dataset_exists = self.check_dataset_exists()
            try:
                existing_list = self.load_existing_experiments()
                for row in existing_list:
                    exp_id = row.get('experiment_id')
                    if exp_id:
                        existing[exp_id] = row
            except Exception:
                existing = {}

            # Safety guard: avoid destructive overwrite if dataset exists but
            # we failed to read any existing records (e.g., transient HF issue)
            if dataset_exists and len(existing) == 0 and len(experiments) <= 3:
                logger.error(
                    "❌ Refusing to overwrite dataset: existing records could not be loaded "
                    "but repository exists. Skipping save to prevent data loss."
                )
                return False
            
            # Validate and merge
            merged_map: Dict[str, Dict[str, Any]] = {}
            # Seed with existing
            for exp_id, row in existing.items():
                merged_map[exp_id] = row
            
            # Apply incoming
            for exp in experiments:
                if not self._validate_experiment_structure(exp):
                    logger.error(f"❌ Invalid experiment structure: {exp.get('experiment_id', 'unknown')}")
                    return False
                exp_id = exp['experiment_id']
                incoming = exp
                if exp_id not in merged_map:
                    incoming['last_updated'] = incoming.get('last_updated') or datetime.now().isoformat()
                    merged_map[exp_id] = incoming
                    continue
                # Merge with existing
                base = merged_map[exp_id]
                # Parse JSON fields
                base_metrics = _parse_json_field(base.get('metrics'), [])
                base_params = _parse_json_field(base.get('parameters'), {})
                base_artifacts = _parse_json_field(base.get('artifacts'), [])
                base_logs = _parse_json_field(base.get('logs'), [])
                inc_metrics = _parse_json_field(incoming.get('metrics'), [])
                inc_params = _parse_json_field(incoming.get('parameters'), {})
                inc_artifacts = _parse_json_field(incoming.get('artifacts'), [])
                inc_logs = _parse_json_field(incoming.get('logs'), [])
                # Merge metrics with de-dup
                merged_metrics = []
                seen = set()
                for entry in base_metrics + inc_metrics:
                    try:
                        # Use the original entry so _metrics_key can properly
                        # distinguish dict vs non-dict entries
                        key = _metrics_key(entry)
                    except Exception:
                        key = (None, None)
                    if key not in seen:
                        seen.add(key)
                        merged_metrics.append(entry)
                # Merge params
                merged_params = {}
                if isinstance(base_params, dict):
                    merged_params.update(base_params)
                if isinstance(inc_params, dict):
                    merged_params.update(inc_params)
                # Merge artifacts and logs with de-dup
                def _dedup_list(lst):
                    out = []
                    seen_local = set()
                    for item in lst:
                        key = json.dumps(item, sort_keys=True, default=str) if not isinstance(item, str) else item
                        if key not in seen_local:
                            seen_local.add(key)
                            out.append(item)
                    return out
                merged_artifacts = _dedup_list(list(base_artifacts) + list(inc_artifacts))
                merged_logs = _dedup_list(list(base_logs) + list(inc_logs))
                # Rebuild merged record preferring incoming scalars
                merged_rec = dict(base)
                merged_rec.update({k: v for k, v in incoming.items() if k not in ('metrics', 'parameters', 'artifacts', 'logs')})
                merged_rec['metrics'] = json.dumps(merged_metrics, default=str)
                merged_rec['parameters'] = json.dumps(merged_params, default=str)
                merged_rec['artifacts'] = json.dumps(merged_artifacts, default=str)
                merged_rec['logs'] = json.dumps(merged_logs, default=str)
                merged_rec['last_updated'] = datetime.now().isoformat()
                merged_map[exp_id] = merged_rec
            
            # Prepare final list
            valid_experiments = list(merged_map.values())
            # Ensure all have mandatory fields encoded
            normalized = []
            for rec in valid_experiments:
                # Normalize json fields to strings
                for f, default in (('metrics', []), ('parameters', {}), ('artifacts', []), ('logs', [])):
                    val = rec.get(f)
                    if not isinstance(val, str):
                        rec[f] = json.dumps(val if val is not None else default, default=str)
                if 'last_updated' not in rec:
                    rec['last_updated'] = datetime.now().isoformat()
                normalized.append(rec)
            
            dataset = Dataset.from_list(normalized)
            
            # Generate commit message if not provided
            if not commit_message:
                commit_message = f"Union-merge update with {len(normalized)} experiments ({datetime.now().isoformat()})"
            
            # Push to hub
            dataset.push_to_hub(
                self.dataset_repo,
                token=self.hf_token,
                private=True,
                commit_message=commit_message
            )
            
            logger.info(f"✅ Successfully saved {len(normalized)} experiments (union-merged) to {self.dataset_repo}")
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
