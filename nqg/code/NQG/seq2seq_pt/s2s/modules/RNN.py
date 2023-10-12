#%%
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

try:
    import ipdb
except ImportError:
    pass

BIAS : bool = True
MULTIPLIER : int = 3

class GRUImpl(nn.Module):
    
    def __init__(self, input_size : int, hidden_size : int ) :
        super(GRUImpl, self).__init__()
        
        self.input_size : int = input_size
        self.hidden_size : int = hidden_size
        self.linear_input : nn.Module = nn.Linear(input_size, MULTIPLIER * hidden_size, bias=BIAS)
        self.linear_hidden :nn.Module = nn.Linear(hidden_size, MULTIPLIER * hidden_size, bias=BIAS )
        
        self.sigmoid : nn.Module = nn.Sigmoid()
        self.tanh : nn.Module = nn.Tanh()
        
    def forward(self, input, hidden, mask = None):
        x_W : torch.Tensor = self.linear_input(input) # This just Weight for the input
        h_U : torch.Tensor = self.linear_hidden(hidden) # This weight from the previous state
        
        x_Ws : torch.Tensor = x_W.split(self.hidden_size, dim=1) #  Making the x_W to -> (batch_size, self.hidden_size, ...) shape
        h_Us : torch.Tensor = h_U.split(self.hidden_size, dim=1) 
        
        resetGate : torch.Tensor = self.sigmoid(x_Ws[0] + h_Us[0] ) # Using the[0] Index for weights, Different weights for RNN
        updateGate : torch.Tensor = self.sigmoid(x_Ws[1] + h_Us[1]) # Using the[1] index for weights, Different weights for RNN
        currentMemoryContent : torch.Tensor = self.tanh(x_Ws[2] + resetGate * h_Us[2] ) #  How much should i retain from the hidden Weight ( previous state )
        finalMemory : torch.Tensor = updateGate * (currentMemoryContent - hidden ) + hidden # How much should i retain from my current weight

        if mask:
            finalMemory = (finalMemory - hidden) * mask.unsqueeze(1).expand_as(hidden) + hidden

        return finalMemory

    def __repr__(self):
        return self.__class__.__name__ + '({0}, {1})'.format(self.input_size, self.hidden_size)
#%% Code for testing whether its running or not
# Define # hyperparameters
""" 
input_size : int = 10
hidden_size : int = 20
sequence_length : int = 5

# Create an instance of the GRUImpl class
gru = GRUImpl(input_size, hidden_size)

# Generate some random input data
batch_size = 3
input_data = torch.randn(sequence_length, batch_size, input_size)
hidden_state = torch.randn(batch_size, hidden_size)

# Forward pass through the GRU
for i in range(sequence_length):
    input_step = input_data[i]
    hidden_state = gru(input_step, hidden_state)
    print(f"Time Step {i + 1} - Hidden State:\n{hidden_state}\n")
"""