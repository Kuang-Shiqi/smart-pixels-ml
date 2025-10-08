"""
Knowledge Distillation loss functions for Smart Pixels models.
This module imports implementation from knowledge_distillation.py
"""

from knowledge_distillation import KnowledgeDistillationLoss, DistillationTrainer

# Factory function to create the loss
def get_kd_loss(temperature=3.0, alpha=0.5):
    """
    Create a knowledge distillation loss function.
    
    Args:
        temperature: Temperature for softening distributions
        alpha: Weight balancing hard loss vs distillation loss
    
    Returns:
        KD loss function
    """
    return KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)

# For compatibility with existing code
def create_kd_wrapper(teacher_model, temperature=3.0, alpha=0.5):
    """
    Create a wrapper loss function that includes the teacher model predictions.
    
    Args:
        teacher_model: The teacher model to get predictions from
        temperature: Temperature for softening distributions
        alpha: Weight for balancing task loss vs distillation loss
        
    Returns:
        A loss function that can be used in model.compile()
    """
    kd_loss = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    
    def loss_fn(y_true, y_pred):
        # This is a placeholder - actual implementation is in KnowledgeDistillationLoss
        # This won't work directly with model.compile() since we need teacher predictions
        return kd_loss(y_true, y_pred, None)
    
    return loss_fn

# Get trainer class
def get_distillation_trainer(student_model, teacher_model, base_dir, 
                            alpha=0.5, temperature=1.0, learning_rate=1e-4):
    """
    Create a distillation trainer instance.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model for knowledge distillation
        base_dir: Directory for saving checkpoints and logs
        alpha: Weight for task loss vs distillation loss
        temperature: Temperature for softening distributions
        learning_rate: Learning rate for optimizer
        
    Returns:
        DistillationTrainer instance
    """
    return DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        base_dir=base_dir,
        alpha=alpha,
        temperature=temperature,
        learning_rate=learning_rate
    )