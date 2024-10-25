import torch

def get_x1(dim, isotropic, scale):
    x1 = torch.ones(size=(dim,1), dtype=torch.cfloat)
    if isotropic == 1:
        x1_real = torch.randn(size=(dim,1)) * scale
        x1.real = x1_real
    elif isotropic == 2:
        x1.real = 0.
        index = torch.randint(dim, (1,)).item()
        x1[index].real = (torch.randint(2, (1,)).item() * 2. - 1) * scale
    return x1

def generate_ar_process(dim, T, m, isotropic, scale, if_train=True, if_test=True):
    assert dim >= 1
    assert T >= 3

    if if_train == True:

        train_set = []
        for i in range(m):
            W = torch.zeros(size=(dim, dim), dtype=torch.cfloat)
            for j in range(dim):
                theta = torch.rand(1)[0] * 2 * torch.pi
                W[j,j].real = torch.cos(theta)
                W[j,j].imag = torch.sin(theta)

            x1 = get_x1(dim, isotropic, scale)
            x_list = [x1]

            for t in range(T-1):
                x1 = W @ x1
                x_list.append(x1)
            
            x_list = torch.concatenate(x_list, dim=1)
            train_set.append(x_list)

        train_set = torch.stack(train_set,dim=0)
        
    test_set = None
    if if_test == True:
        test_set = []
        for i in range(10000):
            W = torch.zeros(size=(dim, dim), dtype=torch.cfloat)
            for j in range(dim):
                theta = torch.rand(1)[0] * 2 * torch.pi
                W[j,j].real = torch.cos(theta)
                W[j,j].imag = torch.sin(theta)

            x1 = get_x1(dim, isotropic, scale)
            x_list = [x1]

            for t in range(T-1):
                x1 = W @ x1
                x_list.append(x1)
            
            x_list = torch.concatenate(x_list, dim=1)
            test_set.append(x_list)

        test_set = torch.stack(test_set,dim=0)
    
    return train_set, test_set

if __name__ == '__main__':
    for dim in [5]:
        for T in [4]:
            generate_ar_process(dim, T, 5, if_test=True)
