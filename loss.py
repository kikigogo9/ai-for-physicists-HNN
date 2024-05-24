from torch import nn

class Loss(nn.Module):
    def __init__(self, lambda_r, *args, **kwargs) -> None:
        super(Loss, self).__init__(*args, **kwargs)
        self.lambda_r = lambda_r
    
    def forward(self, x):
        # Example implements custom MSE
        # ((x - f(x))**2).sum()
        # output must be a number
        pass