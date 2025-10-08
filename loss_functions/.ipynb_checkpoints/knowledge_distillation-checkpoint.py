"""
Knowledge Distillation implementation for Smart Pixels models.
This module provides loss functions and training utilities for distilling 
knowledge from a transformer teacher model to a QKeras student model.
"""

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class KnowledgeDistillationLoss:
    """
    Knowledge Distillation Loss for multivariate normal distribution outputs.
    Combines NLL loss with distillation components for means and covariance matrices.
    """
    
    def __init__(self, temperature=1.0, alpha=0.5, minval=1e-9, maxval=1e9):
        """
        Initialize KD loss.
        
        Args:
            temperature: Temperature for softening distributions
            alpha: Weight for task loss (alpha=1 means only task loss, alpha=0 means only distillation)
            minval: Minimum value for numerical stability
            maxval: Maximum value for numerical stability
        """
        self.temperature = temperature
        self.alpha = alpha
        self.minval = minval
        self.maxval = maxval
        
    def _build_mvn(self, p):
        """
        Build multivariate normal distribution from model output.
        
        Args:
            p: Model output (batch_size, 14) containing means and covariance elements
            
        Returns:
            tfp.distributions.MultivariateNormalTriL distribution
        """
        mu = p[:, 0:8:2]  # Extract means (x, y, cotA, cotB)
        
        # Create matrix elements for covariance
        Mdia = self.minval + tf.math.maximum(p[:, 1:8:2], 0.0)  # Diagonal elements
        Mcov = p[:, 8:]  # Off-diagonal elements
        
        # Placeholder zero element
        zeros = tf.zeros_like(Mdia[:, 0])
        
        # Assemble scale_tril matrix (lower triangular)
        row1 = tf.stack([Mdia[:, 0], zeros, zeros, zeros])
        row2 = tf.stack([Mcov[:, 0], Mdia[:, 1], zeros, zeros])
        row3 = tf.stack([Mcov[:, 1], Mcov[:, 2], Mdia[:, 2], zeros])
        row4 = tf.stack([Mcov[:, 3], Mcov[:, 4], Mcov[:, 5], Mdia[:, 3]])
        
        scale_tril = tf.transpose(tf.stack([row1, row2, row3, row4]), perm=[2, 0, 1])
        
        return tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril)
    
    def _nll_loss(self, y_true, p):
        """
        Calculate negative log-likelihood loss.
        
        Args:
            y_true: Ground truth values [batch_size, 4]
            p: Model output [batch_size, 14]
            
        Returns:
            NLL loss
        """
        dist = self._build_mvn(p)
        likelihood = dist.prob(y_true)
        likelihood = tf.clip_by_value(likelihood, self.minval, self.maxval)
        return -tf.math.log(likelihood)
    
    def _kl_divergence(self, p_student, p_teacher):
        """
        Calculate KL divergence between student and teacher distributions.
        
        Args:
            p_student: Student model output [batch_size, 14]
            p_teacher: Teacher model output [batch_size, 14]
            
        Returns:
            KL divergence loss
        """
        # Build distributions
        student_dist = self._build_mvn(p_student)
        teacher_dist = self._build_mvn(p_teacher)
        
        # KL divergence for multivariate normal
        # We use Monte Carlo approximation to compute it efficiently
        # (exact solution requires matrix determinants and inverses)
        
        # First, sample from teacher distribution
        samples = teacher_dist.sample(10)  # Generate 10 samples per batch item
        
        # Calculate log probs under both distributions
        log_p_teacher = teacher_dist.log_prob(samples)
        log_p_student = student_dist.log_prob(samples)
        
        # KL divergence approximation: E_teacher[log(p_teacher/p_student)]
        kl_div = log_p_teacher - log_p_student
        
        # Average over samples
        return tf.reduce_mean(kl_div, axis=0)
    
    def __call__(self, y_true, student_output, teacher_output):
        """
        Calculate combined knowledge distillation loss.
        
        Args:
            y_true: Ground truth values [batch_size, 4]
            student_output: Student model predictions [batch_size, 14]
            teacher_output: Teacher model predictions [batch_size, 14]
            
        Returns:
            Combined loss
        """
        # Task loss - NLL with ground truth
        task_loss = self._nll_loss(y_true, student_output)
        
        # Distillation loss - KL divergence
        kl_loss = self._kl_divergence(student_output, teacher_output)
        
        # Additional MSE loss for direct parameter matching
        # Extract means from both models
        student_means = student_output[:, 0:8:2]
        teacher_means = teacher_output[:, 0:8:2]
        
        # Mean squared error between student and teacher means
        mse_loss = tf.reduce_mean(tf.square(student_means - teacher_means))
        
        # Combine losses - KL div can be numerically unstable, so we add MSE component
        distill_loss = 0.7 * kl_loss + 0.3 * mse_loss
        
        # Apply temperature scaling to soften the distributions
        distill_loss = distill_loss / self.temperature
        
        # Combined loss with alpha weighting
        combined_loss = self.alpha * task_loss + (1.0 - self.alpha) * distill_loss
        
        # Keep track of loss components for monitoring
        self.task_loss_value = tf.reduce_mean(task_loss)
        self.distill_loss_value = tf.reduce_mean(distill_loss)
        
        return tf.reduce_sum(combined_loss)


class DistillationTrainer:
    """
    Trainer class for knowledge distillation.
    """
    
    def __init__(self, student_model, teacher_model, base_dir, 
                 alpha=0.5, temperature=1.0, learning_rate=1e-4):
        """
        Initialize distillation trainer.
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model for knowledge distillation
            base_dir: Directory for saving checkpoints and logs
            alpha: Weight for task loss vs distillation loss
            temperature: Temperature for softening distributions
            learning_rate: Learning rate for optimizer
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.base_dir = base_dir
        self.alpha = alpha
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Nadam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        # Create loss function
        self.loss_fn = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
        
        # Create metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_task_loss = tf.keras.metrics.Mean(name='train_task_loss')
        self.train_distill_loss = tf.keras.metrics.Mean(name='train_distill_loss')
        
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_task_loss = tf.keras.metrics.Mean(name='val_task_loss')
        self.val_distill_loss = tf.keras.metrics.Mean(name='val_distill_loss')
        
        # Initialize history
        self.history = {
            'loss': [], 'task_loss': [], 'distill_loss': [],
            'val_loss': [], 'val_task_loss': [], 'val_distill_loss': []
        }
        
        # For early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Ensure teacher model is not trainable
        self.teacher_model.trainable = False
        
        # Create checkpoint manager
        self.checkpoint_path = os.path.join(base_dir, 'best_model.h5')
    
    @tf.function
    def train_step(self, x, y):
        """
        Single training step with gradient update.
        
        Args:
            x: Input data
            y: Ground truth
            
        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            # Forward pass for both models
            student_output = self.student_model(x, training=True)
            teacher_output = self.teacher_model(x, training=False)
            
            # Calculate loss
            loss = self.loss_fn(y, student_output, teacher_output)
        
        # Get gradients and update weights
        gradients = tape.gradient(loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_task_loss.update_state(self.loss_fn.task_loss_value)
        self.train_distill_loss.update_state(self.loss_fn.distill_loss_value)
        
        return loss
    
    @tf.function
    def val_step(self, x, y):
        """
        Single validation step.
        
        Args:
            x: Input data
            y: Ground truth
            
        Returns:
            Loss value
        """
        # Forward pass
        student_output = self.student_model(x, training=False)
        teacher_output = self.teacher_model(x, training=False)
        
        # Calculate loss
        loss = self.loss_fn(y, student_output, teacher_output)
        
        # Update metrics
        self.val_loss.update_state(loss)
        self.val_task_loss.update_state(self.loss_fn.task_loss_value)
        self.val_distill_loss.update_state(self.loss_fn.distill_loss_value)
        
        return loss
    
    def train(self, train_generator, validation_generator, epochs=100, patience=20):
        """
        Train the student model using knowledge distillation.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        print("Starting knowledge distillation training...")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Reset metrics
            self.train_loss.reset_states()
            self.train_task_loss.reset_states()
            self.train_distill_loss.reset_states()
            self.val_loss.reset_states()
            self.val_task_loss.reset_states()
            self.val_distill_loss.reset_states()
            
            # Training
            batch_count = 0
            for x_batch, y_batch in tqdm(train_generator, desc="Training"):
                self.train_step(x_batch, y_batch)
                batch_count += 1
            
            # Validation
            for x_val, y_val in tqdm(validation_generator, desc="Validation"):
                self.val_step(x_val, y_val)
            
            # Update history
            self.history['loss'].append(self.train_loss.result().numpy())
            self.history['task_loss'].append(self.train_task_loss.result().numpy())
            self.history['distill_loss'].append(self.train_distill_loss.result().numpy())
            self.history['val_loss'].append(self.val_loss.result().numpy())
            self.history['val_task_loss'].append(self.val_task_loss.result().numpy())
            self.history['val_distill_loss'].append(self.val_distill_loss.result().numpy())
            
            # Print results
            print(f"Loss: {self.train_loss.result():.4f} "
                  f"(Task: {self.train_task_loss.result():.4f}, "
                  f"Distill: {self.train_distill_loss.result():.4f})")
            print(f"Val Loss: {self.val_loss.result():.4f} "
                  f"(Task: {self.val_task_loss.result():.4f}, "
                  f"Distill: {self.val_distill_loss.result():.4f})")
            
            # Check early stopping
            val_loss = self.val_loss.result().numpy()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.student_model.save_weights(self.checkpoint_path)
                print(f"Model saved to {self.checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Reset generators for next epoch
            train_generator.on_epoch_end()
            validation_generator.on_epoch_end()
        
        # Load best model
        self.student_model.load_weights(self.checkpoint_path)
        
        # Plot and save training history
        self._plot_history()
        
        return self.history
    
    def _plot_history(self):
        """Plot and save training history."""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot task loss
        plt.subplot(1, 3, 2)
        plt.plot(self.history['task_loss'], label='Train Task Loss')
        plt.plot(self.history['val_task_loss'], label='Val Task Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Task Loss')
        plt.title('Task Loss (NLL)')
        plt.legend()
        plt.grid(True)
        
        # Plot distillation loss
        plt.subplot(1, 3, 3)
        plt.plot(self.history['distill_loss'], label='Train Distill Loss')
        plt.plot(self.history['val_distill_loss'], label='Val Distill Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Distillation Loss')
        plt.title('Distillation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'training_history.png'))
        plt.close()
    
    def evaluate(self, test_generator, scaling_factors=None):
        """
        Evaluate and compare student vs teacher models.
        
        Args:
            test_generator: Test data generator
            scaling_factors: Optional scaling factors for denormalization
            
        Returns:
            DataFrame with evaluation metrics
        """
        import pandas as pd
        
        print("Evaluating student vs teacher model...")
        
        # Collect predictions and ground truth
        student_preds = []
        teacher_preds = []
        ground_truth = []
        
        for x_batch, y_batch in tqdm(test_generator, desc="Getting predictions"):
            student_pred = self.student_model.predict(x_batch)
            teacher_pred = self.teacher_model.predict(x_batch)
            
            student_preds.append(student_pred)
            teacher_preds.append(teacher_pred)
            ground_truth.append(y_batch)
        
        # Concatenate batches
        student_preds = np.concatenate(student_preds, axis=0)
        teacher_preds = np.concatenate(teacher_preds, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)
        
        # Create DataFrames for analysis
        param_names = ['x', 'y', 'cotA', 'cotB']
        
        # Extract means (parameters)
        student_means = student_preds[:, 0:8:2]
        teacher_means = teacher_preds[:, 0:8:2]
        
        # Build DataFrames
        df_results = pd.DataFrame()
        
        # Add predictions
        for i, param in enumerate(param_names):
            df_results[f'student_{param}'] = student_means[:, i]
            df_results[f'teacher_{param}'] = teacher_means[:, i]
            df_results[f'true_{param}'] = ground_truth[:, i]
        
        # Calculate residuals
        for param in param_names:
            df_results[f'student_{param}_residual'] = df_results[f'true_{param}'] - df_results[f'student_{param}']
            df_results[f'teacher_{param}_residual'] = df_results[f'true_{param}'] - df_results[f'teacher_{param}']
        
        # Apply scaling for physical units if provided
        if scaling_factors is not None:
            for i, param in enumerate(param_names):
                scale = scaling_factors[i]
                df_results[f'student_{param}_real'] = df_results[f'student_{param}'] * scale
                df_results[f'teacher_{param}_real'] = df_results[f'teacher_{param}'] * scale
                df_results[f'true_{param}_real'] = df_results[f'true_{param}'] * scale
                df_results[f'student_{param}_residual_real'] = df_results[f'student_{param}_residual'] * scale
                df_results[f'teacher_{param}_residual_real'] = df_results[f'teacher_{param}_residual'] * scale
        
        # Plot comparison
        self._plot_comparison(df_results, param_names, scaling_factors)
        
        # Calculate metrics
        metrics = {}
        for param in param_names:
            # MSE
            metrics[f'student_{param}_mse'] = np.mean(df_results[f'student_{param}_residual'] ** 2)
            metrics[f'teacher_{param}_mse'] = np.mean(df_results[f'teacher_{param}_residual'] ** 2)
            
            # MAE
            metrics[f'student_{param}_mae'] = np.mean(np.abs(df_results[f'student_{param}_residual']))
            metrics[f'teacher_{param}_mae'] = np.mean(np.abs(df_results[f'teacher_{param}_residual']))
            
            # Standard deviation
            metrics[f'student_{param}_std'] = np.std(df_results[f'student_{param}_residual'])
            metrics[f'teacher_{param}_std'] = np.std(df_results[f'teacher_{param}_residual'])
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print("------------------")
        for param in param_names:
            print(f"\n{param.upper()} Parameter:")
            print(f"  Student MSE: {metrics[f'student_{param}_mse']:.6f}")
            print(f"  Teacher MSE: {metrics[f'teacher_{param}_mse']:.6f}")
            print(f"  Student MAE: {metrics[f'student_{param}_mae']:.6f}")
            print(f"  Teacher MAE: {metrics[f'teacher_{param}_mae']:.6f}")
            print(f"  Student STD: {metrics[f'student_{param}_std']:.6f}")
            print(f"  Teacher STD: {metrics[f'teacher_{param}_std']:.6f}")
        
        return df_results, metrics
    
    def _plot_comparison(self, df, param_names, scaling_factors=None):
        """
        Plot comparison of student vs teacher predictions.
        
        Args:
            df: DataFrame with predictions
            param_names: List of parameter names
            scaling_factors: Optional scaling factors for denormalization
        """
        plt.figure(figsize=(20, 15))
        n_params = len(param_names)
        
        # Plot residual histograms
        for i, param in enumerate(param_names):
            plt.subplot(n_params, 3, i*3 + 1)
            if scaling_factors is not None:
                scale = scaling_factors[i]
                plt.hist(df[f'student_{param}_residual'] * scale, bins=50, alpha=0.5, label='Student')
                plt.hist(df[f'teacher_{param}_residual'] * scale, bins=50, alpha=0.5, label='Teacher')
                plt.xlabel(f'{param} Residual [scaled]')
            else:
                plt.hist(df[f'student_{param}_residual'], bins=50, alpha=0.5, label='Student')
                plt.hist(df[f'teacher_{param}_residual'], bins=50, alpha=0.5, label='Teacher')
                plt.xlabel(f'{param} Residual')
            plt.ylabel('Count')
            plt.title(f'{param} Residual Distribution')
            plt.legend()
            plt.grid(True)
            
            # Plot correlation
            plt.subplot(n_params, 3, i*3 + 2)
            plt.scatter(df[f'true_{param}'], df[f'student_{param}'], alpha=0.3, label='Student')
            plt.scatter(df[f'true_{param}'], df[f'teacher_{param}'], alpha=0.3, label='Teacher')
            plt.plot([df[f'true_{param}'].min(), df[f'true_{param}'].max()], 
                    [df[f'true_{param}'].min(), df[f'true_{param}'].max()],
                    'k--', alpha=0.5)
            plt.xlabel(f'True {param}')
            plt.ylabel(f'Predicted {param}')
            plt.title(f'{param} Prediction Correlation')
            plt.legend()
            plt.grid(True)
            
            # Plot residual vs true
            plt.subplot(n_params, 3, i*3 + 3)
            plt.scatter(df[f'true_{param}'], df[f'student_{param}_residual'], alpha=0.3, label='Student')
            plt.scatter(df[f'true_{param}'], df[f'teacher_{param}_residual'], alpha=0.3, label='Teacher')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel(f'True {param}')
            plt.ylabel(f'Residual')
            plt.title(f'{param} Residual vs True Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, 'model_comparison.png'))
        plt.close()