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
            logger.info(f"Experiment details retrieved: {result['data'][:100]}...")
            return result
        else:
            logger.error(f"Failed to get experiment details: {result}")
            return result
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        logger.info("Listing all experiments")
        
        result = self._make_api_call("list_experiments_interface", [])
        
        if "success" in result:
            logger.info(f"Experiments listed: {result['data'][:100]}...")
            return result
        else:
            logger.error(f"Failed to list experiments: {result}")
            return result
    
    def update_experiment_status(self, experiment_id: str, status: str) -> Dict[str, Any]:
        """Update experiment status"""
        logger.info(f"Updating experiment {experiment_id} status to {status}")
        
        result = self._make_api_call("update_experiment_status_interface", [experiment_id, status])
        
        if "success" in result:
            logger.info(f"Status updated successfully: {result['data']}")
            return result
        else:
            logger.error(f"Failed to update status: {result}")
            return result
    
    def simulate_training_data(self, experiment_id: str) -> Dict[str, Any]:
        """Simulate training data for demonstration"""
        logger.info(f"Simulating training data for experiment {experiment_id}")
        
        result = self._make_api_call("simulate_training_data", [experiment_id])
        
        if "success" in result:
            logger.info(f"Training data simulated: {result['data']}")
            return result
        else:
            logger.error(f"Failed to simulate training data: {result}")
            return result
    
    def get_training_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get training metrics for an experiment"""
        logger.info(f"Getting training metrics for experiment {experiment_id}")
        
        result = self._make_api_call("get_training_metrics", [experiment_id])
        
        if "success" in result:
            logger.info(f"Training metrics retrieved: {result['data'][:100]}...")
            return result
        else:
            logger.error(f"Failed to get training metrics: {result}")
            return result
    
    def get_experiment_metrics_history(self, experiment_id: str) -> Dict[str, Any]:
        """Get complete metrics history for an experiment"""
        logger.info(f"Getting metrics history for experiment {experiment_id}")
        
        result = self._make_api_call("get_metrics_history", [experiment_id])
        
        if "success" in result:
            logger.info(f"Metrics history retrieved: {result['data'][:100]}...")
            return result
        else:
            logger.error(f"Failed to get metrics history: {result}")
            return result

def test_simple_connection():
    """Test basic connectivity to the Space"""
    print("🔍 Testing Basic Space Connectivity")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        response = requests.get("https://tonic-test-trackio-test.hf.space", timeout=10)
        if response.status_code == 200:
            print("✅ Space is accessible")
            return True
        else:
            print(f"❌ Space returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Space: {e}")
        return False

def test_api_connection():
    """Test the API connection"""
    print("🔍 Testing Trackio API Connection")
    print("=" * 50)
    
    # First test basic connectivity
    if not test_simple_connection():
        return
    
    # Initialize client
    client = TrackioAPIClient("https://tonic-test-trackio-test.hf.space")
    
    # Test 1: Create experiment
    print("\n1. Testing experiment creation...")
    create_result = client.create_experiment(
        "test_experiment_api",
        "Test experiment created via API"
    )
    
    if "success" in create_result:
        print("✅ Experiment created successfully")
        
        # Extract experiment ID from the response
        response_text = create_result['data']
        # Look for experiment ID in the response
        if "exp_" in response_text:
            # Extract the experiment ID
            import re
            match = re.search(r'exp_\d{8}_\d{6}', response_text)
            if match:
                experiment_id = match.group()
                print(f"   Experiment ID: {experiment_id}")
                
                # Test 2: Log parameters
                print("\n2. Testing parameter logging...")
                parameters = {
                    "model_name": "HuggingFaceTB/SmolLM3-3B",
                    "batch_size": 8,
                    "learning_rate": 3.5e-6,
                    "max_iters": 18000
                }
                
                param_result = client.log_parameters(experiment_id, parameters)
                if "success" in param_result:
                    print("✅ Parameters logged successfully")
                else:
                    print(f"❌ Failed to log parameters: {param_result}")
                
                # Test 3: Log metrics
                print("\n3. Testing metrics logging...")
                metrics = {
                    "loss": 0.5234,
                    "accuracy": 0.8567,
                    "learning_rate": 3.5e-6,
                    "gpu_memory_gb": 22.5
                }
                
                metrics_result = client.log_metrics(experiment_id, metrics, 100)
                if "success" in metrics_result:
                    print("✅ Metrics logged successfully")
                else:
                    print(f"❌ Failed to log metrics: {metrics_result}")
                
                # Test 4: List experiments
                print("\n4. Testing experiment listing...")
                list_result = client.list_experiments()
                if "success" in list_result:
                    print("✅ Experiments listed successfully")
                    try:
                        response_preview = list_result['data'][:200]
                        print(f"   Response: {response_preview}...")
                    except UnicodeEncodeError:
                        print(f"   Response: {list_result['data'][:100].encode('utf-8', errors='ignore').decode('utf-8')}...")
                else:
                    print(f"❌ Failed to list experiments: {list_result}")
                
                # Test 5: Get experiment details
                print("\n5. Testing experiment details...")
                details_result = client.get_experiment_details(experiment_id)
                if "success" in details_result:
                    print("✅ Experiment details retrieved successfully")
                    try:
                        response_preview = details_result['data'][:200]
                        print(f"   Response: {response_preview}...")
                    except UnicodeEncodeError:
                        print(f"   Response: {details_result['data'][:100].encode('utf-8', errors='ignore').decode('utf-8')}...")
                else:
                    print(f"❌ Failed to get experiment details: {details_result}")
                
            else:
                print("❌ Could not extract experiment ID from response")
        else:
            print("❌ No experiment ID found in response")
    else:
        print(f"❌ Failed to create experiment: {create_result}")
    
    print("\n" + "=" * 50)
    print("🎯 API Test Complete")
    print("=" * 50)

def create_real_experiment():
    """Create a real experiment for your training"""
    print("🚀 Creating Real Experiment for Training")
    print("=" * 50)
    
    client = TrackioAPIClient("https://tonic-test-trackio-test.hf.space")
    
    # Create experiment
    create_result = client.create_experiment(
        "petit-elle-l-aime-3-balanced",
        "SmolLM3 fine-tuning on OpenHermes-FR dataset with balanced A100 configuration"
    )
    
    if "success" in create_result:
        print("✅ Experiment created successfully")
        print(f"Response: {create_result['data']}")
        
        # Extract experiment ID
        import re
        match = re.search(r'exp_\d{8}_\d{6}', create_result['data'])
        if match:
            experiment_id = match.group()
            print(f"📋 Experiment ID: {experiment_id}")
            
            # Log initial parameters
            parameters = {
                "model_name": "HuggingFaceTB/SmolLM3-3B",
                "dataset_name": "legmlai/openhermes-fr",
                "batch_size": 8,
                "gradient_accumulation_steps": 16,
                "effective_batch_size": 128,
                "learning_rate": 3.5e-6,
                "max_iters": 18000,
                "max_seq_length": 12288,
                "mixed_precision": "bf16",
                "use_flash_attention": True,
                "optimizer": "adamw_torch",
                "scheduler": "cosine",
                "warmup_steps": 1200,
                "save_steps": 2000,
                "eval_steps": 1000,
                "logging_steps": 25,
                "no_think_system_message": True
            }
            
            param_result = client.log_parameters(experiment_id, parameters)
            if "success" in param_result:
                print("✅ Initial parameters logged")
            else:
                print(f"❌ Failed to log parameters: {param_result}")
            
            return experiment_id
        else:
            print("❌ Could not extract experiment ID")
            return None
    else:
        print(f"❌ Failed to create experiment: {create_result}")
        return None

if __name__ == "__main__":
    # Test the API connection
    test_api_connection()
    
    print("\n" + "=" * 60)
    print("🎯 CREATING REAL EXPERIMENT")
    print("=" * 60)
    
    # Create real experiment
    experiment_id = create_real_experiment()
    
    if experiment_id:
        print(f"\n✅ SUCCESS! Your experiment is ready:")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Trackio Space: https://tonic-test-trackio-test.hf.space")
        print(f"   View experiments in the 'View Experiments' tab")
        
        print(f"\n📋 Next steps:")
        print(f"1. Use this experiment ID in your training script")
        print(f"2. Monitor progress in the Trackio Space")
        print(f"3. Log metrics as training progresses")
    else:
        print("\n❌ Failed to create experiment") 