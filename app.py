"""
Trackio Deployment on Hugging Face Spaces
A Gradio interface for experiment tracking and monitoring
"""

import gradio as gr
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackioSpace:
    """Trackio deployment for Hugging Face Spaces"""
    
    def __init__(self):
        self.experiments = {}
        self.current_experiment = None
        
    def create_experiment(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new experiment"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'metrics': [],
            'parameters': {},
            'artifacts': [],
            'logs': []
        }
        
        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment_id
        
        logger.info(f"Created experiment: {experiment_id} - {name}")
        return experiment
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics for an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'metrics': metrics
        }
        
        self.experiments[experiment_id]['metrics'].append(metric_entry)
        logger.info(f"Logged metrics for experiment {experiment_id}: {metrics}")
    
    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]):
        """Log parameters for an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.experiments[experiment_id]['parameters'].update(parameters)
        logger.info(f"Logged parameters for experiment {experiment_id}: {parameters}")
    
    def log_artifact(self, experiment_id: str, artifact_name: str, artifact_data: str):
        """Log an artifact for an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        artifact_entry = {
            'name': artifact_name,
            'timestamp': datetime.now().isoformat(),
            'data': artifact_data
        }
        
        self.experiments[experiment_id]['artifacts'].append(artifact_entry)
        logger.info(f"Logged artifact for experiment {experiment_id}: {artifact_name}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        return {
            'experiments': list(self.experiments.keys()),
            'current_experiment': self.current_experiment,
            'total_experiments': len(self.experiments)
        }
    
    def update_experiment_status(self, experiment_id: str, status: str):
        """Update experiment status"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = status
            logger.info(f"Updated experiment {experiment_id} status to {status}")

# Initialize Trackio space
trackio_space = TrackioSpace()

def create_experiment_interface(name: str, description: str) -> str:
    """Create a new experiment"""
    try:
        experiment = trackio_space.create_experiment(name, description)
        return f"✅ Experiment created successfully!\nID: {experiment['id']}\nName: {experiment['name']}"
    except Exception as e:
        return f"❌ Error creating experiment: {str(e)}"

def log_metrics_interface(experiment_id: str, metrics_json: str, step: str) -> str:
    """Log metrics for an experiment"""
    try:
        metrics = json.loads(metrics_json)
        step_int = int(step) if step else None
        trackio_space.log_metrics(experiment_id, metrics, step_int)
        return f"✅ Metrics logged successfully for experiment {experiment_id}"
    except Exception as e:
        return f"❌ Error logging metrics: {str(e)}"

def log_parameters_interface(experiment_id: str, parameters_json: str) -> str:
    """Log parameters for an experiment"""
    try:
        parameters = json.loads(parameters_json)
        trackio_space.log_parameters(experiment_id, parameters)
        return f"✅ Parameters logged successfully for experiment {experiment_id}"
    except Exception as e:
        return f"❌ Error logging parameters: {str(e)}"

def get_experiment_details(experiment_id: str) -> str:
    """Get experiment details"""
    try:
        experiment = trackio_space.get_experiment(experiment_id)
        if experiment:
            return json.dumps(experiment, indent=2)
        else:
            return f"❌ Experiment {experiment_id} not found"
    except Exception as e:
        return f"❌ Error getting experiment details: {str(e)}"

def list_experiments_interface() -> str:
    """List all experiments"""
    try:
        experiments_info = trackio_space.list_experiments()
        return json.dumps(experiments_info, indent=2)
    except Exception as e:
        return f"❌ Error listing experiments: {str(e)}"

def update_experiment_status_interface(experiment_id: str, status: str) -> str:
    """Update experiment status"""
    try:
        trackio_space.update_experiment_status(experiment_id, status)
        return f"✅ Experiment {experiment_id} status updated to {status}"
    except Exception as e:
        return f"❌ Error updating experiment status: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Trackio - Experiment Tracking", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Trackio Experiment Tracking")
    gr.Markdown("Monitor and track your ML experiments with ease!")
    
    with gr.Tabs():
        # Create Experiment Tab
        with gr.Tab("Create Experiment"):
            gr.Markdown("### Create a New Experiment")
            with gr.Row():
                with gr.Column():
                    experiment_name = gr.Textbox(
                        label="Experiment Name",
                        placeholder="my_smollm3_finetune",
                        value="smollm3_finetune"
                    )
                    experiment_description = gr.Textbox(
                        label="Description",
                        placeholder="Fine-tuning SmolLM3 model on custom dataset",
                        value="SmolLM3 fine-tuning experiment"
                    )
                    create_btn = gr.Button("Create Experiment", variant="primary")
                
                with gr.Column():
                    create_output = gr.Textbox(
                        label="Result",
                        lines=5,
                        interactive=False
                    )
            
            create_btn.click(
                create_experiment_interface,
                inputs=[experiment_name, experiment_description],
                outputs=create_output
            )
        
        # Log Metrics Tab
        with gr.Tab("Log Metrics"):
            gr.Markdown("### Log Training Metrics")
            with gr.Row():
                with gr.Column():
                    metrics_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    metrics_json = gr.Textbox(
                        label="Metrics (JSON)",
                        placeholder='{"loss": 0.5, "accuracy": 0.85}',
                        value='{"loss": 0.5, "accuracy": 0.85}'
                    )
                    metrics_step = gr.Textbox(
                        label="Step (optional)",
                        placeholder="100"
                    )
                    log_metrics_btn = gr.Button("Log Metrics", variant="primary")
                
                with gr.Column():
                    metrics_output = gr.Textbox(
                        label="Result",
                        lines=3,
                        interactive=False
                    )
            
            log_metrics_btn.click(
                log_metrics_interface,
                inputs=[metrics_exp_id, metrics_json, metrics_step],
                outputs=metrics_output
            )
        
        # Log Parameters Tab
        with gr.Tab("Log Parameters"):
            gr.Markdown("### Log Experiment Parameters")
            with gr.Row():
                with gr.Column():
                    params_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    parameters_json = gr.Textbox(
                        label="Parameters (JSON)",
                        placeholder='{"learning_rate": 2e-5, "batch_size": 4}',
                        value='{"learning_rate": 2e-5, "batch_size": 4, "model_name": "HuggingFaceTB/SmolLM3-3B"}'
                    )
                    log_params_btn = gr.Button("Log Parameters", variant="primary")
                
                with gr.Column():
                    params_output = gr.Textbox(
                        label="Result",
                        lines=3,
                        interactive=False
                    )
            
            log_params_btn.click(
                log_parameters_interface,
                inputs=[params_exp_id, parameters_json],
                outputs=params_output
            )
        
        # View Experiments Tab
        with gr.Tab("View Experiments"):
            gr.Markdown("### View Experiment Details")
            with gr.Row():
                with gr.Column():
                    view_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    view_btn = gr.Button("View Experiment", variant="primary")
                    list_btn = gr.Button("List All Experiments", variant="secondary")
                
                with gr.Column():
                    view_output = gr.Textbox(
                        label="Experiment Details",
                        lines=15,
                        interactive=False
                    )
            
            view_btn.click(
                get_experiment_details,
                inputs=[view_exp_id],
                outputs=view_output
            )
            
            list_btn.click(
                list_experiments_interface,
                inputs=[],
                outputs=view_output
            )
        
        # Update Status Tab
        with gr.Tab("Update Status"):
            gr.Markdown("### Update Experiment Status")
            with gr.Row():
                with gr.Column():
                    status_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    status_dropdown = gr.Dropdown(
                        label="Status",
                        choices=["running", "completed", "failed", "paused"],
                        value="running"
                    )
                    update_status_btn = gr.Button("Update Status", variant="primary")
                
                with gr.Column():
                    status_output = gr.Textbox(
                        label="Result",
                        lines=3,
                        interactive=False
                    )
            
            update_status_btn.click(
                update_experiment_status_interface,
                inputs=[status_exp_id, status_dropdown],
                outputs=status_output
            )

# Launch the app
if __name__ == "__main__":
    demo.launch() 