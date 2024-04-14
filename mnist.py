import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datautils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Logger import Logger
from ShufflingDataLoader import ShufflingDataLoader

epoch_length = 10
learning_rate = 1e-2
scheduler_gamma = 0.99
do_nudge = True
nudge_magnitude = 0.5
data_precision_type = torch.float32
expected_end_loss = 5e-2
stale_loss_treshold_change = 1e-3
patience_window = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_layers_sizes = [4]


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 10, kernel_size=3, stride=1, padding=1
        )  # Input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2)  # Kernel size, stride
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(20 * 7 * 7, 50)  # Flattened size, outputs
        self.out = nn.Linear(50, 10)  # Output classes: 10 digits

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 20 * 7 * 7)  # Flatten the tensor for the fully connected layer
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.out(x)
        return nn.functional.log_softmax(x, dim=1)


# Load data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the image to [-1, 1] range
    ]
)

train_data = datasets.MNIST(
    "/mnt/d/projects-data/pytorch-learning/mnist",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.MNIST(
    "/mnt/d/projects-data/pytorch-learning/mnist",
    train=False,
    download=True,
    transform=transform,
)

train_loader = ShufflingDataLoader(train_data, batch_size=64, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in datautils.default_collate(x)))
test_loader = ShufflingDataLoader(test_data, batch_size=64, shuffle=False,  collate_fn=lambda x: tuple(x_.to(device) for x_ in datautils.default_collate(x)))

# Initialize the model
model = MNISTNet()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=1 - scheduler_gamma, patience=patience_window
)

logger = Logger("MNIST")
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

# Training loop
for epoch in range(epoch_length):  # Number of epochs
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
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

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    valid_loss /= len(test_loader)

    scheduler.step(valid_loss)

    logger.logger.info(
        f"Epoch {epoch+1}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}"
    )

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

logger.logger.info(
    f"Accuracy of the network on the 10000 test images: {100 * correct / total}%"
)
