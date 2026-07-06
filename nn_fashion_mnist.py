import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms

######################################### Model Creation #########################################

# 1. Transformations : convert to tensor et normalize images (Fashion-MNIST is black and white)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Download datasets
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 3. Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print("DataLoaders creation was a success!")

class FlexibleNet(nn.Module):
    def __init__(self, hidden_sizes, activation_type="relu"):
        super().__init__()

        # Keep activation type
        self.activation_type = activation_type.lower()

        # 1. Input layer : Fashion-MNIST = 28x28 images = 784 pixels
        # Dynamic list of linear layers
        layers = []
        input_dim = 784

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            input_dim = hidden_size  # the output becomes the input of the next layer

        # 2. Output layer : 10 clothes classes
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(input_dim, 10)

    def forward(self, x):
        # Plane image : from (Batch_size, 1, 28, 28) to (Batch_size, 784)
        x = torch.flatten(x, start_dim=1)

        # Go through hidden layers with the chosen activation
        for layer in self.hidden_layers:
            x = layer(x)
            if self.activation_type == "relu":
                x = F.relu(x)
            elif self.activation_type == "sigmoid":
                x = torch.sigmoid(x)

        # Raw output (logits) for all 10 classes
        x = self.output_layer(x)
        return x

######################################### Model Training & Evaluation #########################################

def train_and_evaluate(hidden_sizes, activation_type, optimizer_type, lr=0.01, epochs=5):
    # 1. Instanciate model, loss criterion and optimizer
    model = FlexibleNet(hidden_sizes, activation_type)
    criterion = nn.CrossEntropyLoss() # multi-class classification

    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining : {hidden_sizes} | Activation: {activation_type} | Optimizer: {optimizer_type}")

    # 2. Training Loop
    for epoch in range(epochs):
        model.train() # activate training mode
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()           # Clear previous gradients
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels) # Loss
            loss.backward()                 # Backward pass
            optimizer.step()                # Update weights

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 3. Evaluation on test dataset
    model.eval() # activate evaluation mode
    correct = 0
    total = 0

    with torch.no_grad(): # deactivate Autograd (faster, less RAM used)
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # get idex of the best score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Final accuracy : {accuracy:.2f}%")
    return accuracy, model

######################################### Saving best model #########################################

# Test configuration 2 (ReLu, Adam)
acc_1, model_1 = train_and_evaluate(hidden_sizes=[128], activation_type="relu", optimizer_type="adam", lr=0.001, epochs=5)

# Save the best model (lr = 0.001)
torch.save(model_1.state_dict(), 'model.pt')
print("Best model saved as 'model.pt' !")