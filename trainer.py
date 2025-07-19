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
        
        # Get datasets
        train_dataset = self.dataset.get_train_dataset()
        eval_dataset = self.dataset.get_eval_dataset()
        
        # Get data collator
        data_collator = self.dataset.get_data_collator()
        
        # Add monitoring callback
        callbacks = []
        if self.monitor and self.monitor.enable_tracking:
            trackio_callback = self.monitor.create_monitoring_callback()
            if trackio_callback:
                callbacks.append(trackio_callback)
        
        if self.use_sft_trainer:
            # Use SFTTrainer for supervised fine-tuning
            trainer = SFTTrainer(
                model=self.model.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                data_collator=data_collator,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                packing=False,  # Disable packing for better control
                callbacks=callbacks,
            )
        else:
            # Use standard Trainer
            trainer = Trainer(
                model=self.model.model,
                tokenizer=self.model.tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
            )
        
        return trainer
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
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
        
        # Log configuration to Trackio
        if self.monitor and self.monitor.enable_tracking:
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_')}
            self.monitor.log_config(config_dict)
            
            # Log experiment URL
            experiment_url = self.monitor.get_experiment_url()
            if experiment_url:
                logger.info(f"Trackio experiment URL: {experiment_url}")
        
        # Load checkpoint if resuming
        if self.init_from == "resume":
            checkpoint_path = "/input-checkpoint"
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
            else:
                logger.warning(f"Checkpoint path {checkpoint_path} not found, starting from scratch")
        
        # Start training
        try:
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            
            # Save training results
            with open(os.path.join(self.output_dir, "train_results.json"), "w") as f:
                json.dump(train_result.metrics, f, indent=2)
            
            # Log training summary to Trackio
            if self.monitor and self.monitor.enable_tracking:
                summary = {
                    'final_loss': train_result.metrics.get('train_loss', 0),
                    'total_steps': train_result.metrics.get('train_runtime', 0),
                    'training_time': train_result.metrics.get('train_runtime', 0),
                    'output_dir': self.output_dir,
                    'model_name': getattr(self.config, 'model_name', 'unknown'),
                }
                self.monitor.log_training_summary(summary)
                self.monitor.close()
            
            logger.info("Training completed successfully!")
            logger.info(f"Training metrics: {train_result.metrics}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Close monitoring on error
            if self.monitor and self.monitor.enable_tracking:
                self.monitor.close()
            raise
    
    def evaluate(self):
        """Evaluate the model"""
        logger.info("Starting evaluation")
        
        try:
            eval_results = self.trainer.evaluate()
            
            # Save evaluation results
            with open(os.path.join(self.output_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info(f"Evaluation completed: {eval_results}")
            return eval_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        save_path = path or self.output_dir
        logger.info(f"Saving model to {save_path}")
        
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
            logger.error(f"Failed to save model: {e}")
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
            logger.info(f"Training metrics: {train_result.metrics}")
            
        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise 