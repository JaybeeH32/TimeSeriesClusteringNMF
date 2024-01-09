import torch
import torch.nn as nn
import numpy as np
from libraries.nmf import nmf_sklearn


class DeepNMF(nn.Module):
    def __init__(self, shapes) -> None:
        super(DeepNMF, self).__init__()
        self.nb_layers = len(shapes)
        self.shapes = shapes
        self.log_u_parameters = [nn.Parameter(torch.Tensor(shapes[i-1], shapes[i]).float()) for i in range(1, self.nb_layers)]
        self.log_vp_parameter = nn.Parameter(torch.Tensor(shapes[-1], shapes[0]).float())
        
    def initialize_weights(self, A) -> None:
        print(f'Initializing Layer {1}')
        ui, vi, _ = nmf_sklearn(A, self.shapes[1], max_iter=2000)
        self.log_u_parameters[0] = nn.Parameter(torch.log(torch.from_numpy(ui).contiguous()).float())
        for i in range(2, self.nb_layers):
            print(f'Initializing Layer {i}')
            ui, vi, _ = nmf_sklearn(vi, self.shapes[i], max_iter=2000)
            self.log_u_parameters[i-1] = nn.Parameter(torch.log(torch.from_numpy(ui).contiguous()).float())
        self.log_vp_parameter = nn.Parameter(torch.log(torch.from_numpy(vi).contiguous()).float())
        
    def forward(self):
        output = self.log_u_parameters[0].exp()
        for i in range(1, len(self.log_u_parameters)):
            output = torch.matmul(output, self.log_u_parameters[i].exp())
        return torch.matmul(output, self.log_vp_parameter.exp())
    
    def vp(self):
        return self.log_vp_parameter.exp()
    
    def infer_clusters(self):
        output = self.log_u_parameters[0].exp()
        for i in range(1, self.nb_layers - 1):
            output = torch.matmul(output, self.log_u_parameters[i].exp())
        return output
            
class DeepNMFLoss(nn.Module):
    def __init__(self):
        super(DeepNMFLoss, self).__init__()

    def forward(self, nmf_approx, A):
        return torch.linalg.norm(A - nmf_approx, ord='fro')**2

class DANMF(nn.Module):
    def __init__(self, shapes) -> None:
        super(DANMF, self).__init__()
        self.nb_layers = len(shapes)
        self.shapes = shapes
        self.log_u_parameters = [nn.Parameter(torch.Tensor(shapes[i-1], shapes[i]).float()) for i in range(1, self.nb_layers)]
        self.log_vp_parameter = nn.Parameter(torch.Tensor(shapes[-1], shapes[0]).float())
        
    def initialize_weights(self, A) -> None:
        print(f'Initializing Layer {1}')
        ui, vi, _ = nmf_sklearn(A, self.shapes[1], max_iter=2000)
        self.log_u_parameters[0] = nn.Parameter(torch.log(torch.from_numpy(ui).contiguous()).float())
        for i in range(2, self.nb_layers):
            print(f'Initializing Layer {i}')
            ui, vi, _ = nmf_sklearn(vi, self.shapes[i], max_iter=2000)
            self.log_u_parameters[i-1] = nn.Parameter(torch.log(torch.from_numpy(ui).contiguous()).float())
        self.log_vp_parameter = nn.Parameter(torch.log(torch.from_numpy(vi).contiguous()).float())
        
    def forward(self):
        output = self.log_u_parameters[0].exp()
        for i in range(1, len(self.log_u_parameters)):
            output = torch.matmul(output, self.log_u_parameters[i].exp())
        return torch.matmul(output, self.log_vp_parameter.exp())
    
    def approx_vp(self, A):
        output = self.log_u_parameters[-1].exp().T
        for i in range(1, len(self.log_u_parameters)):
            output = torch.matmul(output, self.log_u_parameters[-1 -i].exp().T)
        return torch.matmul(output, A)
    
    def vp(self):
        return self.log_vp_parameter.exp()
    
    def infer_clusters(self):
        output = self.log_u_parameters[0].exp()
        for i in range(1, self.nb_layers - 1):
            output = torch.matmul(output, self.log_u_parameters[i].exp())
        return output
            
class DANMFLoss(nn.Module):
    def __init__(self, regularization):
        super(DANMFLoss, self).__init__()
        self.regularization = regularization

    def forward(self, nmf_approx, vp_approx, A, vp, L):
        return torch.linalg.norm(A - nmf_approx, ord='fro')**2 + torch.linalg.norm(vp - vp_approx, ord='fro')**2 + self.regularization * torch.trace(torch.matmul(torch.matmul(vp, L), vp.T))

def deep_nmf(Anp, d, lr=0.01, nb_iters=1000, visualize=False):
    # Hyperparameters
    n = len(Anp)
    nb_clusters = d
    shapes = [n, n//4, n//16, nb_clusters]

    # transform to torch tensor
    A = torch.from_numpy(Anp).contiguous().float()

    # Model
    model = DeepNMF(shapes)
    model.initialize_weights(Anp)

    loss_fn = DeepNMFLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    lrs = []

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Compute the loss and its gradients
    losses = []
    for step in range(nb_iters):
        outputs = model()
        loss = loss_fn(outputs, A)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        
        # def closure():
        #     optimizer.zero_grad()
        #     outputs = model()
        #     loss = loss_fn(outputs, torch.Tensor(A))
        #     loss.backward()
        #     return loss
        # # Adjust learning weights
        # optimizer.step(closure)
    
    if visualize:
        import matplotlib.pyplot as plt
        plt.plot(losses)
    return model.infer_clusters().detach().numpy(), model.vp().detach().numpy(), losses[-1]


def deep_autoencoder_nmf(Anp, d, lr=0.01, nb_iters=1000, sparsity_regularization=1.0, visualize=False):
    # Hyperparameters
    n = len(Anp)
    nb_clusters = d
    shapes = [n, n//4, n//16, nb_clusters]

    # graph laplacian
    W = np.where(Anp > 0, -1, 0)
    D = - np.diag(np.sum(W, axis=0))
    L = torch.from_numpy(D + W).contiguous().float()
    A = torch.from_numpy(Anp).contiguous().float()

    # Model
    model = DANMF(shapes)
    model.initialize_weights(Anp)

    loss_fn = DANMFLoss(regularization=sparsity_regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    lrs = []

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Compute the loss and its gradients
    losses = []
    for step in range(nb_iters):
        outputs = model()
        loss = loss_fn(outputs, model.approx_vp(A), A, model.vp(), L)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        
        # def closure():
        #     optimizer.zero_grad()
        #     outputs = model()
        #     loss = loss_fn(outputs, torch.Tensor(A))
        #     loss.backward()
        #     return loss
        # # Adjust learning weights
        # optimizer.step(closure)
    
    if visualize:
        import matplotlib.pyplot as plt
        plt.plot(losses)
    return model.infer_clusters().detach().numpy(), model.vp().detach().numpy(), torch.linalg.norm((A - model.infer_clusters() @ model.vp()), ord='fro').detach().numpy()

