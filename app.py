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
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

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
    
    def get_metrics_dataframe(self, experiment_id: str) -> pd.DataFrame:
        """Get metrics as a pandas DataFrame for plotting"""
        if experiment_id not in self.experiments:
            return pd.DataFrame()
        
        experiment = self.experiments[experiment_id]
        if not experiment['metrics']:
            return pd.DataFrame()
        
        # Convert metrics to DataFrame
        data = []
        for metric_entry in experiment['metrics']:
            step = metric_entry.get('step', 0)
            timestamp = metric_entry.get('timestamp', '')
            metrics = metric_entry.get('metrics', {})
            
            row = {'step': step, 'timestamp': timestamp}
            row.update(metrics)
            data.append(row)
        
        return pd.DataFrame(data)

# Initialize Trackio space
trackio_space = TrackioSpace()

def create_experiment_interface(name: str, description: str) -> str:
    """Create a new experiment"""
    try:
        experiment = trackio_space.create_experiment(name, description)
        return f"✅ Experiment created successfully!\nID: {experiment['id']}\nName: {experiment['name']}\nStatus: {experiment['status']}"
    except Exception as e:
        return f"❌ Error creating experiment: {str(e)}"

def log_metrics_interface(experiment_id: str, metrics_json: str, step: str) -> str:
    """Log metrics for an experiment"""
    try:
        metrics = json.loads(metrics_json)
        step_int = int(step) if step else None
        trackio_space.log_metrics(experiment_id, metrics, step_int)
        return f"✅ Metrics logged successfully for experiment {experiment_id}\nStep: {step_int}\nMetrics: {json.dumps(metrics, indent=2)}"
    except Exception as e:
        return f"❌ Error logging metrics: {str(e)}"

def log_parameters_interface(experiment_id: str, parameters_json: str) -> str:
    """Log parameters for an experiment"""
    try:
        parameters = json.loads(parameters_json)
        trackio_space.log_parameters(experiment_id, parameters)
        return f"✅ Parameters logged successfully for experiment {experiment_id}\nParameters: {json.dumps(parameters, indent=2)}"
    except Exception as e:
        return f"❌ Error logging parameters: {str(e)}"

def get_experiment_details(experiment_id: str) -> str:
    """Get experiment details"""
    try:
        experiment = trackio_space.get_experiment(experiment_id)
        if experiment:
            # Format the output nicely
            details = f"""
📊 EXPERIMENT DETAILS
====================
ID: {experiment['id']}
Name: {experiment['name']}
Description: {experiment['description']}
Status: {experiment['status']}
Created: {experiment['created_at']}

📈 METRICS COUNT: {len(experiment['metrics'])}
📋 PARAMETERS COUNT: {len(experiment['parameters'])}
📦 ARTIFACTS COUNT: {len(experiment['artifacts'])}

🔧 PARAMETERS:
{json.dumps(experiment['parameters'], indent=2)}

📊 LATEST METRICS:
"""
            if experiment['metrics']:
                latest_metrics = experiment['metrics'][-1]
                details += f"Step: {latest_metrics.get('step', 'N/A')}\n"
                details += f"Timestamp: {latest_metrics.get('timestamp', 'N/A')}\n"
                details += f"Metrics: {json.dumps(latest_metrics.get('metrics', {}), indent=2)}"
            else:
                details += "No metrics logged yet."
            
            return details
        else:
            return f"❌ Experiment {experiment_id} not found"
    except Exception as e:
        return f"❌ Error getting experiment details: {str(e)}"

def list_experiments_interface() -> str:
    """List all experiments with details"""
    try:
        experiments_info = trackio_space.list_experiments()
        experiments = trackio_space.experiments
        
        if not experiments:
            return "📭 No experiments found. Create one first!"
        
        result = f"📋 EXPERIMENTS OVERVIEW\n{'='*50}\n"
        result += f"Total Experiments: {len(experiments)}\n"
        result += f"Current Experiment: {experiments_info['current_experiment']}\n\n"
        
        for exp_id, exp_data in experiments.items():
            status_emoji = {
                'running': '🟢',
                'completed': '✅',
                'failed': '❌',
                'paused': '⏸️'
            }.get(exp_data['status'], '❓')
            
            result += f"{status_emoji} {exp_id}\n"
            result += f"   Name: {exp_data['name']}\n"
            result += f"   Status: {exp_data['status']}\n"
            result += f"   Created: {exp_data['created_at']}\n"
            result += f"   Metrics: {len(exp_data['metrics'])} entries\n"
            result += f"   Parameters: {len(exp_data['parameters'])} entries\n"
            result += f"   Artifacts: {len(exp_data['artifacts'])} entries\n\n"
        
        return result
    except Exception as e:
        return f"❌ Error listing experiments: {str(e)}"

def update_experiment_status_interface(experiment_id: str, status: str) -> str:
    """Update experiment status"""
    try:
        trackio_space.update_experiment_status(experiment_id, status)
        return f"✅ Experiment {experiment_id} status updated to {status}"
    except Exception as e:
        return f"❌ Error updating experiment status: {str(e)}"

def create_metrics_plot(experiment_id: str, metric_name: str = "loss") -> go.Figure:
    """Create a plot for a specific metric"""
    try:
        df = trackio_space.get_metrics_dataframe(experiment_id)
        if df.empty:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        if metric_name not in df.columns:
            # Show available metrics
            available_metrics = [col for col in df.columns if col not in ['step', 'timestamp']]
            fig = go.Figure()
            fig.add_annotation(
                text=f"Available metrics: {', '.join(available_metrics)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = px.line(df, x='step', y=metric_name, title=f'{metric_name} over time')
        fig.update_layout(
            xaxis_title="Training Step",
            yaxis_title=metric_name.title(),
            hovermode='x unified'
        )
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_experiment_comparison(experiment_ids: str) -> go.Figure:
    """Compare multiple experiments"""
    try:
        exp_ids = [exp_id.strip() for exp_id in experiment_ids.split(',')]
        
        fig = go.Figure()
        
        for exp_id in exp_ids:
            df = trackio_space.get_metrics_dataframe(exp_id)
            if not df.empty and 'loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['step'],
                    y=df['loss'],
                    mode='lines+markers',
                    name=f"{exp_id} - Loss",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Experiment Comparison - Loss",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating comparison: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def simulate_training_data(experiment_id: str):
    """Simulate training data for demonstration"""
    try:
        # Simulate some realistic training metrics
        for step in range(0, 1000, 50):
            # Simulate loss decreasing over time
            loss = 2.0 * np.exp(-step / 500) + 0.1 * np.random.random()
            accuracy = 0.3 + 0.6 * (1 - np.exp(-step / 300)) + 0.05 * np.random.random()
            lr = 3.5e-6 * (0.9 ** (step // 200))
            
            metrics = {
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "learning_rate": round(lr, 8),
                "gpu_memory": round(20 + 5 * np.random.random(), 2),
                "training_time": round(0.5 + 0.2 * np.random.random(), 3)
            }
            
            trackio_space.log_metrics(experiment_id, metrics, step)
        
        return f"✅ Simulated training data for experiment {experiment_id}\nAdded 20 metric entries (steps 0-950)"
    except Exception as e:
        return f"❌ Error simulating data: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Trackio - Experiment Tracking", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Trackio Experiment Tracking & Monitoring")
    gr.Markdown("Monitor and track your ML experiments with real-time visualization!")
    
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
                        placeholder='{"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5}',
                        value='{"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5, "gpu_memory": 22.5}'
                    )
                    metrics_step = gr.Textbox(
                        label="Step (optional)",
                        placeholder="100"
                    )
                    log_metrics_btn = gr.Button("Log Metrics", variant="primary")
                
                with gr.Column():
                    metrics_output = gr.Textbox(
                        label="Result",
                        lines=5,
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
                        value='{"learning_rate": 3.5e-6, "batch_size": 8, "model_name": "HuggingFaceTB/SmolLM3-3B", "max_iters": 18000, "mixed_precision": "bf16"}'
                    )
                    log_params_btn = gr.Button("Log Parameters", variant="primary")
                
                with gr.Column():
                    params_output = gr.Textbox(
                        label="Result",
                        lines=5,
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
                        lines=20,
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
        
        # Visualization Tab
        with gr.Tab("📊 Visualizations"):
            gr.Markdown("### Training Metrics Visualization")
            with gr.Row():
                with gr.Column():
                    plot_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    metric_dropdown = gr.Dropdown(
                        label="Metric to Plot",
                        choices=["loss", "accuracy", "learning_rate", "gpu_memory", "training_time"],
                        value="loss"
                    )
                    plot_btn = gr.Button("Create Plot", variant="primary")
                
                with gr.Column():
                    plot_output = gr.Plot(label="Training Metrics")
            
            plot_btn.click(
                create_metrics_plot,
                inputs=[plot_exp_id, metric_dropdown],
                outputs=plot_output
            )
            
            gr.Markdown("### Experiment Comparison")
            with gr.Row():
                with gr.Column():
                    comparison_exp_ids = gr.Textbox(
                        label="Experiment IDs (comma-separated)",
                        placeholder="exp_1,exp_2,exp_3"
                    )
                    comparison_btn = gr.Button("Compare Experiments", variant="primary")
                
                with gr.Column():
                    comparison_plot = gr.Plot(label="Experiment Comparison")
            
            comparison_btn.click(
                create_experiment_comparison,
                inputs=[comparison_exp_ids],
                outputs=comparison_plot
            )
        
        # Demo Data Tab
        with gr.Tab("🎯 Demo Data"):
            gr.Markdown("### Generate Demo Training Data")
            gr.Markdown("Use this to simulate training data for testing the interface")
            with gr.Row():
                with gr.Column():
                    demo_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    demo_btn = gr.Button("Generate Demo Data", variant="primary")
                
                with gr.Column():
                    demo_output = gr.Textbox(
                        label="Result",
                        lines=3,
                        interactive=False
                    )
            
            demo_btn.click(
                simulate_training_data,
                inputs=[demo_exp_id],
                outputs=demo_output
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