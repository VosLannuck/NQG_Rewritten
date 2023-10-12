#%%
import torch
import math
from torch.optim.optimizer import Optimizer
from typing import Dict, Any
""" 
    Fileclass Changed from MyAdam to AdamImpl
"""

class AdamImpl(Optimizer):
    
    def __init__(self, params,
                 lr : float =1e-3, betas : float =(0.9, 0.999),
                 epsilon=1e-8, weight_decay : int = 0):
        if(lr < 0.0 ):
            raise ValueError("Invalid Learning rate: {} - should be >= 0.0".format(lr))
        
        if not (betas[0] >= 0.0 and betas[0] < 1.0 ):
            raise ValueError("Invalid beta parameter {} - should be in [0.0, 1.0]".format(betas[0]))
        
        if not (betas[1] >= 0.0 and betas[1] < 1.0):
            raise ValueError("Invalid beta parameter {} - should be in [0.0, 1.0]".format(betas[0]))

        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value {} - should be >= 0.0".format(epsilon))
        
        defaults : Dict[Any, Any] = dict(
                                lr = lr, betas = betas,
                                epsilon = epsilon,
                                weight_decay = weight_decay)

        super(AdamImpl, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """ 
            Perform single optimization step
        """
        
        loss = None 
        if closure is not None:
            loss = closure() 
        
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                
                grad : torch.Tensor= parameter.grad.data 
                state : Dict[str, torch.Tensor] = self.state[parameter]
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradientValues
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    #Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                
                exp_avg : torch.Tensor = state['exp_avg']
                exp_avg_sq : torch.Tensor = state['exp_avg_sq']
                
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay', parameter.data])
                
                exp_avg.mul_(beta1).add_( 1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                bias_correction1 : torch.Tensor = 1 - beta1 ** state['step']
                bias_correction2 : torch.Tensor = 1 - beta2 ** state['step']
                
                denom : torch.Tensor = exp_avg_sq.sqrt().add_(group["epsilon"] * math.sqrt(bias_correction2))
                step_size : torch.Tensor = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                parameter.data.addcdiv_(-step_size, exp_avg, denom)
        return loss

#%%
## TESTING Code, TESTED with the original code 
""" 
# Define a simple linear model
torch.manual_seed(5) # For testing

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# Create a simple dataset and dataloader
x = torch.rand(10, 2)
y = 2 * x
dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model and optimizer
model = SimpleModel()
optimizer = AdamImpl(model.parameters(), lr=0.01)

# Define a loss function
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# Print the learned parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)                     

"""