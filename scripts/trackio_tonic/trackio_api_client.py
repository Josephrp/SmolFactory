#!/usr/bin/env python3
"""
Trackio API Client for Hugging Face Spaces
Connects to the Trackio Space using the actual API endpoints
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackioAPIClient:
    """API client for Trackio Space"""
    
    def __init__(self, space_url: str):
        self.space_url = space_url.rstrip('/')
        # For Gradio Spaces, we need to use the direct function endpoints
        self.base_url = f"{self.space_url}/gradio_api/call"
        
    def _make_api_call(self, endpoint: str, data: list, max_retries: int = 3) -> Dict[str, Any]:
        """Make an API call to the Trackio Space"""
        url = f"{self.base_url}/{endpoint}"
        
        payload = {
            "data": data
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}: Making POST request to {url}")
                
                # POST request to get EVENT_ID
                response = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"POST request failed: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {"error": f"POST failed: {response.status_code}"}
                
                # Extract EVENT_ID from response
                response_data = response.json()
                logger.debug(f"POST response: {response_data}")
                
                # Check for event_id (correct field name)
                if "event_id" in response_data:
                    event_id = response_data["event_id"]
                elif "hash" in response_data:
                    event_id = response_data["hash"]
                else:
                    logger.error(f"No event_id or hash in response: {response_data}")
                    return {"error": "No EVENT_ID in response"}
                
                # GET request to get results
                get_url = f"{url}/{event_id}"
                logger.debug(f"Making GET request to: {get_url}")
                
                # Wait a bit for the processing to complete
                time.sleep(1)
                
                get_response = requests.get(get_url, timeout=30)
                
                if get_response.status_code != 200:
                    logger.error(f"GET request failed: {get_response.status_code} - {get_response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {"error": f"GET failed: {get_response.status_code}"}
                
                # Check if response is empty
                if not get_response.content:
                    logger.warning(f"Empty response from GET request (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return {"error": "Empty response from server"}
                
                # Parse the response - handle both JSON and SSE formats
                response_text = get_response.text.strip()
                logger.debug(f"Raw response: {response_text}")
                
                # Try to parse as JSON first
                try:
                    result_data = get_response.json()
                    logger.debug(f"Parsed as JSON: {result_data}")
                    
                    if "data" in result_data and len(result_data["data"]) > 0:
                        return {"success": True, "data": result_data["data"][0]}
                    else:
                        logger.warning(f"No data in JSON response (attempt {attempt + 1}): {result_data}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return {"error": "No data in JSON response", "raw": result_data}
                        
                except json.JSONDecodeError:
                    # Try to parse as Server-Sent Events (SSE) format
                    logger.debug("Response is not JSON, trying SSE format")
                    
                    # Parse SSE format: "event: complete\ndata: [\"message\"]"
                    lines = response_text.split('\n')
                    data_line = None
                    
                    for line in lines:
                        if line.startswith('data: '):
                            data_line = line[6:]  # Remove 'data: ' prefix
                            break
                    
                    if data_line:
                        try:
                            # Parse the data array from SSE
                            import ast
                            data_array = ast.literal_eval(data_line)
                            
                            if isinstance(data_array, list) and len(data_array) > 0:
                                result_message = data_array[0]
                                logger.debug(f"Parsed SSE data: {result_message}")
                                return {"success": True, "data": result_message}
                            else:
                                logger.warning(f"Invalid SSE data format (attempt {attempt + 1}): {data_array}")
                                if attempt < max_retries - 1:
                                    time.sleep(2 ** attempt)
                                    continue
                                return {"error": "Invalid SSE data format", "raw": data_array}
                                
                        except (ValueError, SyntaxError) as e:
                            logger.error(f"Failed to parse SSE data: {e}")
                            logger.debug(f"Raw SSE data: {data_line}")
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)
                                continue
                            return {"error": f"Failed to parse SSE data: {e}"}
                    else:
                        logger.error(f"No data line found in SSE response")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return {"error": "No data line in SSE response", "raw": response_text}
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": f"Request failed: {e}"}
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": f"Unexpected error: {e}"}
        
        return {"error": f"Failed after {max_retries} attempts"}
    
    def create_experiment(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new experiment"""
        logger.info(f"Creating experiment: {name}")
        
        result = self._make_api_call("create_experiment_interface", [name, description])
        
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
        
        result = self._make_api_call("log_metrics_interface", [experiment_id, metrics_json, step_str])
        
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
        
        result = self._make_api_call("log_parameters_interface", [experiment_id, parameters_json])
        
        if "success" in result:
            logger.info(f"Parameters logged successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to log parameters: {result}")
            return result
    
    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details"""
        logger.info(f"Getting details for experiment {experiment_id}")
        
        result = self._make_api_call("get_experiment_details", [experiment_id])
        
        if "success" in result:
            logger.info(f"Experiment details retrieved: {result['data']}")
            return result
        else:
            logger.error(f"Failed to get experiment details: {result}")
            return result
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        logger.info("Listing experiments")
        
        result = self._make_api_call("list_experiments_interface", [])
        
        if "success" in result:
            logger.info(f"Experiments listed successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to list experiments: {result}")
            return result
    
    def update_experiment_status(self, experiment_id: str, status: str) -> Dict[str, Any]:
        """Update experiment status"""
        logger.info(f"Updating experiment {experiment_id} status to {status}")
        
        result = self._make_api_call("update_experiment_status_interface", [experiment_id, status])
        
        if "success" in result:
            logger.info(f"Experiment status updated successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to update experiment status: {result}")
            return result
    
    def simulate_training_data(self, experiment_id: str) -> Dict[str, Any]:
        """Simulate training data for testing"""
        logger.info(f"Simulating training data for experiment {experiment_id}")
        
        result = self._make_api_call("simulate_training_data", [experiment_id])
        
        if "success" in result:
            logger.info(f"Training data simulated successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to simulate training data: {result}")
            return result
    
    def get_training_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get training metrics for an experiment"""
        logger.info(f"Getting training metrics for experiment {experiment_id}")
        
        result = self._make_api_call("get_training_metrics_interface", [experiment_id])
        
        if "success" in result:
            logger.info(f"Training metrics retrieved: {result['data']}")
            return result
        else:
            logger.error(f"Failed to get training metrics: {result}")
            return result
    
    def get_experiment_metrics_history(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment metrics history"""
        logger.info(f"Getting metrics history for experiment {experiment_id}")
        
        result = self._make_api_call("get_experiment_metrics_history_interface", [experiment_id])
        
        if "success" in result:
            logger.info(f"Metrics history retrieved: {result['data']}")
            return result
        else:
            logger.error(f"Failed to get metrics history: {result}")
            return result 