import torch
import torch.nn as nn
import BaseNet
from Logger import Logger

epoch_length = 10
learning_rate = 1e-2
scheduler_gamma = 0.99
do_nudge = True
nudge_magnitude = 0.5
data_precision_type = torch.float32
expected_end_loss = 5e-2
stale_loss_treshold_change = 1e-3
patience_window = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_layers_sizes = [4]

# Define the model
class XORNet(BaseNet.BaseNet):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(XORNet, self).__init__(
            input_dim,
            hidden_dims,
            output_dim,
            learning_rate,
            scheduler_gamma,
            nudge_magnitude,
            data_precision_type,
        )

    def forward(self, x):
        x = nn.functional.relu(self.layers[0](x))
        x = nn.functional.sigmoid(self.layers[1](x))
        return nn.functional.relu(x)

# Initialize the model
model = XORNet(2, hidden_layers_sizes, 1)
model.to(device)

# Input and Output Data (Truth Table for XOR)
X = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=data_precision_type,
    device=device,
)
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=data_precision_type, device=device)

logger = Logger("XOR")
logger.log_initial_setup(
    model,
    epoch_length,
    expected_end_loss,
    stale_loss_treshold_change,
    do_nudge,
    patience_window,
    data_precision_type,
    learning_rate,
    scheduler_gamma,
    nudge_magnitude,
)

# Training Loop
step = 0
epoch = 0
last_nudge_step = 0
prev_loss_measure = 0
while True:
    step += 1
    model.train()
    model.optimizer.zero_grad()
    output = model(X)
    loss = model.criterion(output, Y)
    loss.backward()
    loss_measure = loss.item()
    model.optimizer.step()

    if step % epoch_length == 0:
        model.scheduler.step()
        current_lr = model.scheduler.get_last_lr()[0]

        epoch += 1
        loss_delta = abs(loss_measure - prev_loss_measure)
        with torch.no_grad():
            raw_output = model(X)
            raw_output_list = torch.squeeze(raw_output, 1).tolist()
            formatted_output = ["%.4f" % i for i in raw_output_list]
            logger.logger.info(
                f"Step #{step}, epoch {epoch}, Loss: {loss_measure:.4f}, LossΔ:{loss_delta:.7f}, LR: {current_lr:.7f}, predictions: {formatted_output}"
            )

        adjusted_treshold = stale_loss_treshold_change * (current_lr**0.5)
        if (
            do_nudge
            and step - last_nudge_step >= patience_window
            and loss_delta <= adjusted_treshold
            and loss_measure > expected_end_loss
        ):
            logger.logger.info(
                f"--- Stuck at local minimum (loss: {loss_measure:.4f}) at step {step}. Nudging the model"
            )
            model.nudge()
            last_nudge_step = step
            model.optimizer.zero_grad()

        elif loss_measure <= expected_end_loss:
            logger.logger.info(
                f"No more progress (Δ{(prev_loss_measure - loss_measure):.5f}). Loss: {loss_measure:.4f} achieved in {step} steps ({step - last_nudge_step} since last nudge)"
            )
            break

    prev_loss_measure = loss_measure

with torch.no_grad():
    model.eval()
    raw_output = model(X)
    prediction = raw_output.round()
    logger.log_model_training_results(raw_output, prediction, Y)
