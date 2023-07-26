import torch
from torch import nn

# Define the PyTorch model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        out = self.linear(x)
        return out

# Create a model instance - replace 13 with the number of features in your dataset
model = LinearRegression(13)  # replace 13 with the number of features in your dataset

# Load the saved parameters
model.load_state_dict(torch.load('model_previous_price.pth'))

# Put the model in evaluation mode
model.eval()

# Now the model is ready to make predictions
# Assume inputs is your input data as a PyTorch tensor
# predictions = model(inputs)
