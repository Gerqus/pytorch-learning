from typing import Dict, TypedDict
import torch

from Logger import Logger

class Data(TypedDict):
    input: torch.Tensor
    target: torch.Tensor

class TrainEnv:
    def __init__(
        self,
        model,
        training_data: Data,
        validation_data: Data,
        epoch_length,
        learning_rate,
        scheduler_gamma,
        do_nudge,
        nudge_magnitude,
        data_precision_type,
        expected_end_loss,
        stale_loss_treshold_change,
        patience_window,
        hidden_layers_sizes,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.epoch_length = epoch_length
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.do_nudge = do_nudge
        self.nudge_magnitude = nudge_magnitude
        self.data_precision_type = data_precision_type
        self.expected_end_loss = expected_end_loss
        self.stale_loss_treshold_change = stale_loss_treshold_change
        self.patience_window = patience_window
        self.hidden_layers_sizes = hidden_layers_sizes
        self.device = device
        self.logger = Logger(model.__class__.__name__)
    
    def __log_initial_setup(self):
        self.logger.log_initial_setup(
            self.model,
            self.epoch_length,
            self.expected_end_loss,
            self.stale_loss_treshold_change,
            self.do_nudge,
            self.patience_window,
            self.data_precision_type,
            self.learning_rate,
            self.scheduler_gamma,
            self.nudge_magnitude,
        )

    def train(self):
        self.__log_initial_setup()
        # Training Loop
        step = 0
        epoch = 0
        last_nudge_step = 0
        prev_loss_measure = 0
        
        model = self.model
                
        while True:
            step += 1
            
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in self.training_data:
                indices = torch.randperm(data.size(0))
                data_shuffled = data[indices]
                target_shuffled = target[indices]

                optimizer.zero_grad()
                output = model(data_shuffled)
                loss = criterion(output, target_shuffled)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            train_loss /= len(train_loader)
            model.train()
            model.optimizer.zero_grad()
            output = model(self.training_data['input'])
            loss = model.criterion(output, self.training_data['target'])
            loss.backward()
            loss_measure = loss.item()
            model.optimizer.step()

            if step % self.epoch_length == 0:
                model.scheduler.step()
                current_lr = model.scheduler.get_last_lr()[0]

                epoch += 1
                loss_delta = abs(loss_measure - prev_loss_measure)
                with torch.no_grad():
                    raw_output = model(self.training_data['input'])
                    raw_output_list = torch.squeeze(raw_output, 1).tolist()
                    formatted_output = ["%.4f" % i for i in raw_output_list]
                    self.logger.logger.info(
                        f"Step #{step}, epoch {epoch}, Loss: {loss_measure:.4f}, LossΔ:{loss_delta:.7f}, LR: {current_lr:.7f}, predictions: {formatted_output}"
                    )

                adjusted_treshold = self.stale_loss_treshold_change * (current_lr**0.5)
                if (
                    self.do_nudge
                    and step - last_nudge_step >= self.patience_window
                    and loss_delta <= adjusted_treshold
                    and loss_measure > self.expected_end_loss
                ):
                    self.logger.logger.info(
                        f"--- Stuck at local minimum (loss: {loss_measure:.4f}) at step {step}. Nudging the model"
                    )
                    model.nudge()
                    last_nudge_step = step
                    model.optimizer.zero_grad()

                elif loss_measure <= self.expected_end_loss:
                    self.logger.logger.info(
                        f"No more progress (Δ{(prev_loss_measure - loss_measure):.5f}). Loss: {loss_measure:.4f} achieved in {step} steps ({step - last_nudge_step} since last nudge)"
                    )
                    break

            prev_loss_measure = loss_measure

        with torch.no_grad():
            model.eval()
            raw_output = model(self.validation_data['input'])
            prediction = raw_output.round()
            self.logger.log_model_training_results(raw_output, prediction, self.validation_data['target'])
