import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=1)
        nn.init.constant_(m.bias, val=0)

def MLP(in_features, out_features, width):
    model = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=width),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=width, out_features=out_features)
    )
    return model.apply(init_weights)

class AutoregressiveLSA(torch.nn.Module):

    def __init__(self, dim, T, a, b):
        super(AutoregressiveLSA, self).__init__()
        self.dim = dim
        self.T = T

        WKQ = torch.zeros(size=(2*dim, 2*dim), dtype=torch.cfloat)
        WKQ[dim:,:dim] = a * torch.eye(dim)
        self.WKQ = nn.Parameter(WKQ)

        WPV = torch.zeros(size=(dim, 2*dim), dtype=torch.cfloat)
        WPV[:,:dim] = b * torch.eye(dim)
        self.WPV = nn.Parameter(WPV)

        self.rho_t = lambda t: (t-1)
        
    def forward(self, E_t:torch.Tensor):
        rho_t = self.rho_t(E_t.shape[2])
        e_t = E_t[:,:,-1][:,:, None]
        out = self.WPV @ E_t @ E_t.mH @ self.WKQ @ e_t / rho_t
        return out
    
    def forward_train(self, E_T):
        y_pred_list = []
        T = E_T.shape[2]

        for t in range(2, T):
            E_t = E_T[:,:,:t]
            y_pred_list.append(self.forward(E_t))

        y_pred = torch.concatenate(y_pred_list, dim=2)
        return y_pred
            
if __name__ == '__main__':
    x = torch.ones(size=(2,6))
    device = torch.device('cuda')
    model = AutoregressiveLSA(dim_prefix=5, dim_pred=1, device=device)

    x = x.to(device)
    y = model(x)
    print(y)
    print(nn.functional.sigmoid(y))
    print(y.shape)