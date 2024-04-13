import numpy.random as random
import torch
import torch.nn as nn
import torch.optim as optim

class BaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate, scheduler_gamma, nudge_magnitude, data_precision_type = torch.float16):
        super().__init__()

        self.layers = nn.ModuleList()
        self.criterion = nn.MSELoss()
        self.nudge_magnitude = nudge_magnitude
        self.data_precision_type = data_precision_type
        self.scheduler_gamma = scheduler_gamma
        self.learning_rate = learning_rate

        for i in range(len(hidden_dims)):
            if i == 0:
                self.layers.append(
                    nn.Linear(input_dim, hidden_dims[i], dtype=data_precision_type)
                )
            else:
                self.layers.append(
                    nn.Linear(
                        hidden_dims[i - 1], hidden_dims[i], dtype=data_precision_type
                    )
                )

        self.layers.append(
            nn.Linear(hidden_dims[-1], output_dim, dtype=data_precision_type)
        )
        
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.scheduler_gamma)

    def nudge(self):
        with torch.no_grad():
            for (i, layer) in enumerate(self.layers):
                layer.weight.data += torch.randn_like(layer.weight, dtype=self.data_precision_type) * (self.nudge_magnitude)
                layer.bias.data += torch.randn_like(layer.bias, dtype=self.data_precision_type) * (self.nudge_magnitude)
            self.optimizer.param_groups[0]["lr"] = self.learning_rate
