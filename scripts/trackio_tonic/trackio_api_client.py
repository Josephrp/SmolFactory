#!/usr/bin/env python3
"""
Trackio API Client for Hugging Face Spaces
Uses gradio_client for proper API communication with automatic Space URL resolution
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from gradio_client import Client
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    logger.warning("gradio_client not available. Install with: pip install gradio_client")

try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface-hub")

class TrackioAPIClient:
    """API client for Trackio Space using gradio_client with automatic Space URL resolution"""
    
    def __init__(self, space_id: str, hf_token: Optional[str] = None):
        self.space_id = space_id
        self.hf_token = hf_token
        self.client = None
        
        # Auto-resolve Space URL
        self.space_url = self._resolve_space_url()
        
        # Initialize gradio client
        if GRADIO_CLIENT_AVAILABLE and self.space_url:
            try:
                self.client = Client(self.space_url)
                logger.info(f"✅ Connected to Trackio Space: {self.space_id}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Trackio Space: {e}")
                self.client = None
        else:
            logger.error("❌ gradio_client not available. Install with: pip install gradio_client")
    
    def _resolve_space_url(self) -> Optional[str]:
        """Resolve Space URL using Hugging Face Hub API"""
        try:
            # Clean the space_id - remove any URL prefixes
            clean_space_id = self.space_id
            if clean_space_id.startswith('http'):
                # Extract space ID from URL
                if '/spaces/' in clean_space_id:
                    clean_space_id = clean_space_id.split('/spaces/')[-1]
                else:
                    # Try to extract from URL format
                    clean_space_id = clean_space_id.replace('https://', '').replace('http://', '')
                    if '.hf.space' in clean_space_id:
                        clean_space_id = clean_space_id.replace('.hf.space', '').replace('-', '/')
            
            logger.info(f"🔧 Resolving Space URL for ID: {clean_space_id}")
            
            if not HF_HUB_AVAILABLE:
                logger.warning("⚠️ Hugging Face Hub not available, using default URL format")
                # Fallback to default URL format
                space_name = clean_space_id.replace('/', '-')
                return f"https://{space_name}.hf.space"
            
            # Use Hugging Face Hub API to get Space info
            api = HfApi(token=self.hf_token)
            
            # Get Space info
            space_info = api.space_info(clean_space_id)
            if space_info and hasattr(space_info, 'host'):
                # Use the host directly from space_info
                space_url = space_info.host
                logger.info(f"✅ Resolved Space URL: {space_url}")
                return space_url
            else:
                # Fallback to default URL format
                space_name = clean_space_id.replace('/', '-')
                space_url = f"https://{space_name}.hf.space"
                logger.info(f"✅ Using fallback Space URL: {space_url}")
                return space_url
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to resolve Space URL: {e}")
            # Fallback to default URL format
            space_name = self.space_id.replace('/', '-')
            space_url = f"https://{space_name}.hf.space"
            logger.info(f"✅ Using fallback Space URL: {space_url}")
            return space_url
    
    def _make_api_call(self, api_name: str, *args) -> Dict[str, Any]:
        """Make an API call to the Trackio Space using gradio_client"""
        if not self.client:
            return {"error": "Client not available"}
        
        try:
            logger.debug(f"Making API call to {api_name} with args: {args}")
            
            # Use gradio_client to make the prediction
            result = self.client.predict(*args, api_name=api_name)
            
            logger.debug(f"API call result: {result}")
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"API call failed for {api_name}: {e}")
            return {"error": f"API call failed: {str(e)}"}
    
    def create_experiment(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new experiment"""
        logger.info(f"Creating experiment: {name}")
        
        result = self._make_api_call("/create_experiment_interface", name, description)
        
        if "success" in result:
            logger.info(f"Experiment created successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to create experiment: {result}")
            return result
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, Any], step: Optional[int] = None) -> Dict[str, Any]:
        """Log metrics for an experiment"""
        metrics_json = json.dumps(metrics)
        step_str = str(step) if step is not None else ""
        
        logger.info(f"Logging metrics for experiment {experiment_id} at step {step}")
        
        result = self._make_api_call("/log_metrics_interface", experiment_id, metrics_json, step_str)
        
        if "success" in result:
            logger.info(f"Metrics logged successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to log metrics: {result}")
            return result
    
    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Log parameters for an experiment"""
        parameters_json = json.dumps(parameters)
        
        logger.info(f"Logging parameters for experiment {experiment_id}")
        
        result = self._make_api_call("/log_parameters_interface", experiment_id, parameters_json)
        
        if "success" in result:
            logger.info(f"Parameters logged successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to log parameters: {result}")
            return result
    
    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details"""
        logger.info(f"Getting details for experiment {experiment_id}")
        
        result = self._make_api_call("/get_experiment_details", experiment_id)
        
        if "success" in result:
            logger.info(f"Experiment details retrieved: {result['data']}")
            return result
        else:
            logger.error(f"Failed to get experiment details: {result}")
            return result
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        logger.info("Listing experiments")
        
        result = self._make_api_call("/list_experiments_interface")
        
        if "success" in result:
            logger.info(f"Experiments listed successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to list experiments: {result}")
            return result
    
    def update_experiment_status(self, experiment_id: str, status: str) -> Dict[str, Any]:
        """Update experiment status"""
        logger.info(f"Updating experiment {experiment_id} status to {status}")
        
        result = self._make_api_call("/update_experiment_status_interface", experiment_id, status)
        
        if "success" in result:
            logger.info(f"Experiment status updated successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to update experiment status: {result}")
            return result
    
    def simulate_training_data(self, experiment_id: str) -> Dict[str, Any]:
        """Simulate training data for testing"""
        logger.info(f"Simulating training data for experiment {experiment_id}")
        
        result = self._make_api_call("/simulate_training_data", experiment_id)
        
        if "success" in result:
            logger.info(f"Training data simulated successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to simulate training data: {result}")
            return result
    
    def get_training_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get training metrics for an experiment"""
        logger.info(f"Getting training metrics for experiment {experiment_id}")
        
        result = self._make_api_call("/get_experiment_details", experiment_id)
        
        if "success" in result:
            logger.info(f"Training metrics retrieved: {result['data']}")
            return result
        else:
            logger.error(f"Failed to get training metrics: {result}")
            return result
    
    def create_metrics_plot(self, experiment_id: str, metric_name: str = "loss") -> Dict[str, Any]:
        """Create a metrics plot for an experiment"""
        logger.info(f"Creating metrics plot for experiment {experiment_id}, metric: {metric_name}")
        
        result = self._make_api_call("/create_metrics_plot", experiment_id, metric_name)
        
        if "success" in result:
            logger.info(f"Metrics plot created successfully")
            return result
        else:
            logger.error(f"Failed to create metrics plot: {result}")
            return result
    
    def create_experiment_comparison(self, experiment_ids: str) -> Dict[str, Any]:
        """Compare multiple experiments"""
        logger.info(f"Creating experiment comparison for: {experiment_ids}")
        
        result = self._make_api_call("/create_experiment_comparison", experiment_ids)
        
        if "success" in result:
            logger.info(f"Experiment comparison created successfully")
            return result
        else:
            logger.error(f"Failed to create experiment comparison: {result}")
            return result
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the Trackio Space"""
        logger.info("Testing connection to Trackio Space")
        
        try:
            # Try to list experiments as a connection test
            result = self.list_experiments()
            if "success" in result:
                return {"success": True, "message": "Connection successful"}
            else:
                return {"error": "Connection failed", "details": result}
        except Exception as e:
            return {"error": f"Connection test failed: {str(e)}"}
    
    def get_space_info(self) -> Dict[str, Any]:
        """Get information about the Space"""
        try:
            if not HF_HUB_AVAILABLE:
                return {"error": "Hugging Face Hub not available"}
            
            api = HfApi(token=self.hf_token)
            space_info = api.space_info(self.space_id)
            
            return {
                "success": True,
                "data": {
                    "space_id": self.space_id,
                    "space_url": self.space_url,
                    "space_info": {
                        "title": getattr(space_info, 'title', 'Unknown'),
                        "host": getattr(space_info, 'host', 'Unknown'),
                        "stage": getattr(space_info, 'stage', 'Unknown'),
                        "visibility": getattr(space_info, 'visibility', 'Unknown')
                    }
                }
            }
        except Exception as e:
            return {"error": f"Failed to get Space info: {str(e)}"}

# Factory function to create client with dynamic configuration
def create_trackio_client(space_id: Optional[str] = None, hf_token: Optional[str] = None) -> TrackioAPIClient:
    """Create a TrackioAPIClient with dynamic configuration"""
    
    # Get space_id from environment if not provided
    if not space_id:
        space_id = os.environ.get('TRACKIO_URL')
        if not space_id:
            # Try to construct from username and space name
            username = os.environ.get('HF_USERNAME')
            space_name = os.environ.get('TRACKIO_SPACE_NAME')
            if username and space_name:
                space_id = f"https://huggingface.co/spaces/{username}/{space_name}"
            else:
                logger.warning("⚠️ No space_id provided and could not determine from environment")
                return None
    
    # Get HF token from environment if not provided
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN')
    
    if not space_id:
        logger.error("❌ No space_id available for TrackioAPIClient")
        return None
    
    return TrackioAPIClient(space_id, hf_token) 