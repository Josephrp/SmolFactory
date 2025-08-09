"""
SmolLM3 Trainer
Handles the training loop and integrates with Hugging Face Trainer
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
import json

# Import monitoring
from monitoring import create_monitor_from_config

logger = logging.getLogger(__name__)

class SmolLM3Trainer:
    """Trainer for SmolLM3 fine-tuning"""
    
    def __init__(
        self,
        model,
        dataset,
        config,
        output_dir: str,
        init_from: str = "scratch",
        use_sft_trainer: bool = True
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        self.init_from = init_from
        self.use_sft_trainer = use_sft_trainer
        
        # Initialize monitoring
        self.monitor = create_monitor_from_config(config)
        
        # Setup trainer
        self.trainer = self._setup_trainer()
        
    def _setup_trainer(self):
        """Setup the trainer"""
        logger.info("Setting up trainer")
        
        # Get training arguments
        training_args = self.model.get_training_arguments(
            output_dir=self.output_dir,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            max_steps=self.config.max_iters,
        )
        
        # Debug: Print training arguments
        logger.info("Training arguments keys: %s", list(training_args.__dict__.keys()))
        logger.info("Training arguments type: %s", type(training_args))
        
        # Get datasets
        logger.info("Getting train dataset...")
        train_dataset = self.dataset.get_train_dataset()
        logger.info("Train dataset: %s with %d samples", type(train_dataset), len(train_dataset))
        
        logger.info("Getting eval dataset...")
        eval_dataset = self.dataset.get_eval_dataset()
        logger.info("Eval dataset: %s with %d samples", type(eval_dataset), len(eval_dataset))
        
        # Get data collator
        logger.info("Getting data collator...")
        data_collator = self.dataset.get_data_collator()
        logger.info("Data collator: %s", type(data_collator))
        
        # Add monitoring callbacks
        callbacks = []
        
        # Add simple console callback for basic monitoring
        from transformers import TrainerCallback
        
        outer = self
        class SimpleConsoleCallback(TrainerCallback):
            def on_init_end(self, args, state, control, **kwargs):
                """Called when training initialization is complete"""
                print("🔧 Training initialization completed")
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Log metrics to console"""
                if logs and isinstance(logs, dict):
                    step = state.global_step if hasattr(state, 'global_step') else 'unknown'
                    loss = logs.get('loss', 'N/A')
                    lr = logs.get('learning_rate', 'N/A')
                    # Fix format string error by ensuring proper type conversion
                    if isinstance(loss, (int, float)):
                        loss_str = f"{loss:.4f}"
                    else:
                        loss_str = str(loss)
                    if isinstance(lr, (int, float)):
                        lr_str = f"{lr:.2e}"
                    else:
                        lr_str = str(lr)
                    print(f"Step {step}: loss={loss_str}, lr={lr_str}")

                    # Persist metrics via our monitor when Trackio callback isn't active
                    try:
                        if outer.monitor:
                            # Avoid double logging when Trackio callback is used
                            if not outer.monitor.enable_tracking:
                                outer.monitor.log_metrics(dict(logs), step if isinstance(step, int) else None)
                                outer.monitor.log_system_metrics(step if isinstance(step, int) else None)
                    except Exception as e:
                        logger.warning("SimpleConsoleCallback metrics persistence failed: %s", e)
            
            def on_train_begin(self, args, state, control, **kwargs):
                print("🚀 Training started!")
            
            def on_train_end(self, args, state, control, **kwargs):
                print("✅ Training completed!")
            
            def on_save(self, args, state, control, **kwargs):
                step = state.global_step if hasattr(state, 'global_step') else 'unknown'
                print(f"💾 Checkpoint saved at step {step}")
                try:
                    if outer.monitor and not outer.monitor.enable_tracking:
                        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                        if os.path.exists(checkpoint_path):
                            outer.monitor.log_model_checkpoint(checkpoint_path, step if isinstance(step, int) else None)
                except Exception as e:
                    logger.warning("SimpleConsoleCallback checkpoint persistence failed: %s", e)
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and isinstance(metrics, dict):
                    step = state.global_step if hasattr(state, 'global_step') else 'unknown'
                    eval_loss = metrics.get('eval_loss', 'N/A')
                    print(f"📊 Evaluation at step {step}: eval_loss={eval_loss}")
                    try:
                        if outer.monitor and not outer.monitor.enable_tracking:
                            outer.monitor.log_evaluation_results(dict(metrics), step if isinstance(step, int) else None)
                    except Exception as e:
                        logger.warning("SimpleConsoleCallback eval persistence failed: %s", e)
        
        # Add console callback
        callbacks.append(SimpleConsoleCallback())
        logger.info("Added simple console monitoring callback")
        
        # Add monitoring callback if available (always attach; it persists to dataset even if Trackio is disabled)
        if self.monitor:
            try:
                trackio_callback = self.monitor.create_monitoring_callback()
                if trackio_callback:
                    callbacks.append(trackio_callback)
                    logger.info("Added monitoring callback")
                else:
                    logger.warning("Failed to create monitoring callback")
            except Exception as e:
                logger.error("Error creating monitoring callback: %s", e)
                logger.info("Continuing with console monitoring only")
        
        logger.info("Total callbacks: %d", len(callbacks))
        
        # Initialize trackio for TRL compatibility without creating a second experiment
        try:
            import trackio
            if self.monitor:
                # Share the same monitor/experiment with the trackio shim
                try:
                    trackio.set_monitor(self.monitor)  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: ensure the shim at least knows the current ID
                    pass
                logger.info(
                    "Using shared Trackio monitor with experiment ID: %s",
                    getattr(self.monitor, 'experiment_id', None)
                )
            else:
                # Last resort: initialize via shim
                _ = trackio.init(
                    project_name=getattr(self.config, 'experiment_name', 'smollm3_experiment'),
                    experiment_name=getattr(self.config, 'experiment_name', 'smollm3_experiment'),
                    trackio_url=getattr(self.config, 'trackio_url', None),
                    trackio_token=getattr(self.config, 'trackio_token', None),
                    hf_token=getattr(self.config, 'hf_token', None),
                    dataset_repo=getattr(self.config, 'dataset_repo', None)
                )
        except Exception as e:
            logger.warning(f"Failed to wire trackio shim: {e}")
            logger.info("Continuing without trackio shim integration")
        
        # Try SFTTrainer first (better for instruction tuning)
        logger.info("Creating SFTTrainer with training arguments...")
        logger.info("Training args type: %s", type(training_args))
        try:
            trainer = SFTTrainer(
                model=self.model.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                data_collator=data_collator,
                callbacks=callbacks,
            )
            logger.info("Using SFTTrainer (optimized for instruction tuning)")
        except Exception as e:
            logger.warning("SFTTrainer failed: %s", e)
            logger.error("SFTTrainer creation error details: %s: %s", type(e).__name__, str(e))
            
            # Fallback to standard Trainer
            try:
                trainer = Trainer(
                    model=self.model.model,
                    tokenizer=self.model.tokenizer,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                    callbacks=callbacks,
                )
                logger.info("Using standard Hugging Face Trainer (fallback)")
            except Exception as e2:
                logger.error("Standard Trainer also failed: %s", e2)
                raise e2
        
        return trainer
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        logger.info("Loading checkpoint from %s", checkpoint_path)
        
        if self.init_from == "resume":
            # Load the model from checkpoint
            self.model.load_checkpoint(checkpoint_path)
            
            # Update trainer with loaded model
            self.trainer.model = self.model.model
            
            logger.info("Checkpoint loaded successfully")
        elif self.init_from == "pretrained":
            # Model is already loaded from pretrained
            logger.info("Using pretrained model")
        else:
            logger.info("Starting from scratch")
    
    def train(self):
        """Start training"""
        logger.info("Starting training")
        
        # Log configuration (always persist to dataset; Trackio if enabled)
        if self.monitor:
            try:
                config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
                self.monitor.log_config(config_dict)
            except Exception as e:
                logger.warning("Failed to log configuration: %s", e)
            # Log experiment URL only if available
            try:
                experiment_url = self.monitor.get_experiment_url()
                if experiment_url:
                    logger.info("Trackio experiment URL: %s", experiment_url)
            except Exception:
                pass
        
        # Load checkpoint if resuming
        if self.init_from == "resume":
            checkpoint_path = "/input-checkpoint"
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
            else:
                logger.warning("Checkpoint path %s not found, starting from scratch", checkpoint_path)
        
        # Start training
        try:
            logger.info("About to start trainer.train()")
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            
            # Save training results
            with open(os.path.join(self.output_dir, "train_results.json"), "w") as f:
                json.dump(train_result.metrics, f, indent=2)
            
            # Log training summary (always persist to dataset; Trackio if enabled)
            if self.monitor:
                try:
                    summary = {
                        'final_loss': train_result.metrics.get('train_loss', 0),
                        'total_steps': train_result.metrics.get('train_runtime', 0),
                        'training_time': train_result.metrics.get('train_runtime', 0),
                        'output_dir': self.output_dir,
                        'model_name': getattr(self.config, 'model_name', 'unknown'),
                    }
                    self.monitor.log_training_summary(summary)
                    self.monitor.close()
                except Exception as e:
                    logger.warning("Failed to log training summary: %s", e)
            
            # Finish trackio experiment
            try:
                import trackio
                trackio.finish()
                logger.info("Trackio experiment finished")
            except Exception as e:
                logger.warning(f"Failed to finish trackio experiment: {e}")
            
            logger.info("Training completed successfully!")
            logger.info("Training metrics: %s", train_result.metrics)
            
        except Exception as e:
            logger.error("Training failed: %s", e)
            # Close monitoring on error (still persist final status to dataset)
            if self.monitor:
                try:
                    self.monitor.close(final_status="failed")
                except Exception:
                    pass
            
            # Finish trackio experiment on error
            try:
                import trackio
                trackio.finish()
            except Exception as finish_error:
                logger.warning(f"Failed to finish trackio experiment on error: {finish_error}")
            
            raise
    
    def evaluate(self):
        """Evaluate the model"""
        logger.info("Starting evaluation")
        
        try:
            eval_results = self.trainer.evaluate()
            
            # Save evaluation results
            with open(os.path.join(self.output_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info("Evaluation completed: %s", eval_results)
            return eval_results
            
        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            raise
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        save_path = path or self.output_dir
        logger.info("Saving model to %s", save_path)
        
        try:
            self.trainer.save_model(save_path)
            self.model.tokenizer.save_pretrained(save_path)
            
            # Save training configuration
            if self.config:
                config_dict = {k: v for k, v in self.config.__dict__.items() 
                              if not k.startswith('_')}
                with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            
            logger.info("Model saved successfully!")
            
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            raise

class SmolLM3DPOTrainer:
    """DPO Trainer for SmolLM3 preference optimization"""
    
    def __init__(
        self,
        model,
        dataset,
        config,
        output_dir: str,
        ref_model=None
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        self.ref_model = ref_model
        
        # Setup DPO trainer
        self.trainer = self._setup_dpo_trainer()
    
    def _setup_dpo_trainer(self):
        """Setup DPO trainer"""
        from trl import DPOTrainer
        
        # Get training arguments
        training_args = self.model.get_training_arguments(
            output_dir=self.output_dir,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            max_steps=self.config.max_iters,
        )
        
        # Get preference dataset
        train_dataset = self.dataset.get_train_dataset()
        eval_dataset = self.dataset.get_eval_dataset()
        
        # Setup DPO trainer
        trainer = DPOTrainer(
            model=self.model.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model.tokenizer,
            max_prompt_length=self.config.max_seq_length // 2,
            max_length=self.config.max_seq_length,
        )
        
        return trainer
    
    def train(self):
        """Start DPO training"""
        logger.info("Starting DPO training")
        
        try:
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            
            # Save training results
            with open(os.path.join(self.output_dir, "dpo_train_results.json"), "w") as f:
                json.dump(train_result.metrics, f, indent=2)
            
            logger.info("DPO training completed successfully!")
            logger.info("Training metrics: %s", train_result.metrics)
            
        except Exception as e:
            logger.error("DPO training failed: %s", e)
            raise 