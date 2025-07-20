# callback.py

from transformers import TrainerCallback

class LossCollectorCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.student_losses = []
        self.mse_losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'student_loss' in logs and 'mse_loss' in logs:
            self.student_losses.append(logs['student_loss'])
            self.mse_losses.append(logs['mse_loss'])
            self.steps.append(state.global_step)

class EvalLossCollectorCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.eval_losses = []
        self.steps = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            if 'eval_loss' in metrics:
                self.eval_losses.append(metrics['eval_loss'])
                self.steps.append(state.global_step)
            elif 'loss' in metrics:
                self.eval_losses.append(metrics['loss'])
                self.steps.append(state.global_step)
