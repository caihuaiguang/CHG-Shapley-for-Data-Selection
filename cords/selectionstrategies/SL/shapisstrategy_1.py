import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from .dataselectionstrategy import DataSelectionStrategy
from ..helpers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
from torch.utils.data import Subset, DataLoader

import matplotlib.pyplot as plt 

class SHAPISStrategy(DataSelectionStrategy):
    """
    Implementation of SHAPIS Strategy from the paper :footcite:`pmlr-v139-killamsetty21a` for supervised learning frameworks.

    SHAPIS strategy tries to solve the optimization problem given below:

    .. math::
        \\min_{\\mathbf{w}, S: |S| \\leq k} \\Vert \\sum_{i \\in S} w_i \\nabla_{\\theta}L_T^i(\\theta) -  \\nabla_{\\theta}L(\\theta)\\Vert

    In the above equation, :math:`\\mathbf{w}` denotes the weight vector that contains the weights for each data instance, :math:`\mathcal{U}` training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`L` denotes either training loss or validation loss depending on the parameter valid,
    :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.

    The above optimization problem is solved using the Orthogonal Matching Pursuit(OMP) algorithm.

    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss: class
        PyTorch loss function for training
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    selection_type: str
        Type of selection -
        - 'PerClass': PerClass method is where OMP algorithm is applied on each class data points seperately.
        - 'PerBatch': PerBatch method is where OMP algorithm is applied on each minibatch data points.
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    logger : class
        - logger object for logging the information
    valid : bool
        If valid==True, we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    v1 : bool
        If v1==True, we use newer version of OMP solver that is more accurate
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """

    def __init__(self, trainloader, valloader, model, loss,
                 eta, device, num_classes, linear_layer,
                 selection_type, logger, valid=False, v1=True, lam=0, eps=1e-4):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps
        self.v1 = v1
        self.shapley_values = None
        self.sum_1 = self.sum_2 = None
        self.time = 0
        self.selection_only_once = False
        self.gammas = None
        self.idxs = None
        self.lipschitz = 20

    def draw_box(self,X,name):
        tensor = X.cpu().numpy()
        # 计算四分位数
        q1 = np.percentile(tensor, 25)
        median = np.percentile(tensor, 50)
        q3 = np.percentile(tensor, 75)

        # 计算上下边界
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        # 筛选出箱线图内部的数据点
        inner_points = tensor[(tensor >= lower_bound) & (tensor <= upper_bound)]

        # 绘制箱线图
        plt.figure(figsize=(6, 4))
        plt.boxplot(tensor, vert=False, widths=0.7)
        plt.scatter(inner_points, np.ones_like(inner_points), color='blue', alpha=0.5, label='Inner points')
        plt.scatter([median], [1], color='red', label='Median')
        plt.scatter([q1, q3], [1, 1], color='green', label='Quartiles')
        plt.legend()
        plt.title('Boxplot of Tensor')
        plt.xlabel('Values')
        plt.ylabel('Tensor')
        plt.grid(True)
        # 保存图像
        plt.savefig('vector_box_'+name+'.png')
        plt.close()

    def draw_gradients(self,tensor1,tensor2, name):
        from sklearn.decomposition import PCA
        n,d = tensor1.shape
        tensor1_mean = torch.mean(tensor1,dim=0)
        # 合并两个 tensor
        all_tensors = torch.cat([tensor1, tensor2.unsqueeze(0), tensor1_mean.unsqueeze(0)], dim=0)

        # 使用 PCA 进行降维
        pca = PCA(n_components=2)
        result = pca.fit_transform(all_tensors.cpu().numpy())

        # 将降维后的结果拆分成两组
        result_tensor1 = torch.tensor(result[:n])
        result_tensor2 = torch.tensor(result[n:n+1])
        result_tensor3 = torch.tensor(result[n+1:])

        # 绘制图像
        plt.scatter(result_tensor1[:, 0], result_tensor1[:, 1], color='gray', label='Gray') 
        plt.scatter(result_tensor2[:, 0], result_tensor2[:, 1], color='black', label='Validation', s = 40)  
        plt.scatter(result_tensor3[:, 0], result_tensor3[:, 1], color='Blue', label='Training_mean', s = 40)  
        print('Validation:',result_tensor2)
        print('Training_mean:', result_tensor3)
        print('Validation:',tensor2)
        print('Training_mean:', tensor1_mean)
        # 添加标签和图例
        plt.title('Visualization of Two Sets of Vectors (Reduced to 2D)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()

        # 保存图像
        plt.savefig('vector_visualization_2d'+name+'.png')
        plt.close()

        # # 显示图像（可选）
        # plt.show()
    # def phi_j(self, A_len, n, x_j, alpha, x_i_sum1, x_j_sum, x_j_2, x_j_2_sum):
    #     with torch.no_grad():
    #         # n_minus_A_len = n - A_len
    #         # k_range = torch.arange(2, n_minus_A_len + 1, dtype=torch.float32)
    #         # term1 = 0 if n_minus_A_len <= 0 else (torch.sum(1 / (k_range + A_len)**2)+1/(1 + A_len)**2) / n_minus_A_len
    #         # term2 = 0 if n_minus_A_len <= 1 else torch.sum((k_range - 1) / (k_range + A_len)**2) / (n_minus_A_len * (n_minus_A_len - 1))
    #         # term3 = 0 if n_minus_A_len <= 1 else torch.sum((2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2)) / (n_minus_A_len * (n_minus_A_len - 1))
    #         # term4 = 0 if n_minus_A_len <= 2 else torch.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2))/ (n_minus_A_len * (n_minus_A_len - 1) * (n_minus_A_len - 2))
    #         # term5 = 0 if A_len == 0 else (torch.sum((2 * k_range + 2 * A_len - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2))+(2 * A_len + 1) / ((1 + A_len)**2 * A_len**2))/ n_minus_A_len
    #         # term6 = torch.sum(1 / (k_range + A_len))
    #         n_minus_A_len = n - A_len
    #         k_range = np.arange(2, n_minus_A_len + 1, dtype=np.float32)
    #         term1 = 0 if n_minus_A_len <= 0 else (np.sum(1 / (k_range + A_len)**2)+1/(1 + A_len)**2) / n_minus_A_len
    #         term2 = 0 if n_minus_A_len <= 1 else np.sum((k_range - 1) / (k_range + A_len)**2) / (n_minus_A_len * (n_minus_A_len - 1))
    #         term3 = 0 if n_minus_A_len <= 1 else np.sum((2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2)) / (n_minus_A_len * (n_minus_A_len - 1))
    #         term4 = 0 if n_minus_A_len <= 2 else np.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2))/ (n_minus_A_len * (n_minus_A_len - 1) * (n_minus_A_len - 2))
    #         term5 = 0 if A_len == 0 else (np.sum((2 * k_range + 2 * A_len - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2))+(2 * A_len + 1) / ((1 + A_len)**2 * A_len**2))/ n_minus_A_len
    #         term6 = np.sum(1 / (k_range + A_len))+1/(A_len+1)
    #         term1 = torch.tensor(term1).to(self.device)
    #         term2 = torch.tensor(term2).to(self.device)
    #         term3 = torch.tensor(term3).to(self.device)
    #         term4 = torch.tensor(term4).to(self.device)
    #         term5 = torch.tensor(term5).to(self.device)
    #         term6 = torch.tensor(term6).to(self.device)

            
    #         phi =  (-term1 + 2 * term2  - term3  + 2 * term4 ) * x_j_2\
    #             - 2 * (term2 + term4) * torch.mv(x_j, x_j_sum)\
    #             + (term3 - term4) * x_j_2_sum \
    #             + term4  * x_j_sum.pow(2).sum(dim=0)\
    #             - 2* (term1 + term3) * torch.mv(x_j, x_i_sum1)\
    #             + 2 * term3 * torch.dot(x_j_sum, x_i_sum1)\
    #             + term5 * x_i_sum1.pow(2).sum(dim=0)\
    #             + 2 * ((term6 - 1/n) / (n_minus_A_len - 1)) * torch.mv(x_j, alpha)\
    #             - 2 * (term6 / n_minus_A_len - 1 / n) * (1 / (n_minus_A_len - 1)) * torch.dot(x_j_sum, alpha)
    #         phi -= 0 if A_len == 0 else 2 / (n * A_len) * torch.dot(x_i_sum1, alpha)
    #         return phi.to(self.device)
    def compute_gradients(self, valid=False, perBatch=False, perClass=False):
        """
        Computes the gradient of each element.

        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.

        Using different loss functions, the way we calculate the gradients will change.

        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,

        where :math:`N` is the batch size.


        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,

        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        perBatch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        if (perBatch and perClass):
            raise ValueError("batch and perClass are mutually exclusive. Only one of them can be true at a time")

        embDim = self.model.get_embedding_dim()
        if perClass:
            trainloader = self.pctrainloader
            if valid:
                valloader = self.pcvalloader
        else:
            trainloader = self.trainloader
            if valid:
                valloader = self.valloader
            
        if isinstance(trainloader.dataset[0], dict):
            for batch_idx, batch in enumerate(trainloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}        
                if batch_idx == 0:                
                    out, l1 = self.model(**batch, last=True, freeze=True)
                    self.init_out = out
                    self.init_l1 = l1
                    self.y = batch['labels'].view(-1, 1)
                    loss = self.loss(out, batch['labels'].view(-1)).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]                    
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)                    
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)                
                else:                    
                    out, l1 = self.model(**batch, last=True, freeze=True)
                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y = torch.cat((self.y, batch['labels'].view(-1, 1)), dim=0)
                    loss = self.loss(out, batch['labels'].view(-1)).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]                    
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)            
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)                    
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
        else:    
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    self.init_out = out
                    self.init_l1 = l1
                    self.y = targets.view(-1, 1)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y = torch.cat((self.y, targets.view(-1, 1)), dim=0)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        torch.cuda.empty_cache()

        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads

        if valid:
            if isinstance(valloader.dataset[0], dict):
                for batch_idx, batch in enumerate(valloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}        
                    if batch_idx == 0:                
                        out, l1 = self.model(**batch, last=True, freeze=True)
                        self.init_out = out
                        self.init_l1 = l1
                        self.y = batch['labels'].view(-1, 1)
                        loss = self.loss(out, batch['labels'].view(-1)).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]                    
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)                    
                        if perBatch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)                
                    else:                    
                        out, l1 = self.model(**batch, last=True, freeze=True)
                        self.init_out = torch.cat((self.init_out, out), dim=0)
                        self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                        self.y = torch.cat((self.y, batch['labels'].view(-1, 1)), dim=0)
                        loss = self.loss(out, batch['labels'].view(-1)).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]                    
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                        if perBatch:
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)            
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)                    
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            else:    
                for batch_idx, (inputs, targets) in enumerate(valloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if batch_idx == 0:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        self.init_out = out
                        self.init_l1 = l1
                        self.y = targets.view(-1, 1)
                        loss = self.loss(out, targets).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        if perBatch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        self.init_out = torch.cat((self.init_out, out), dim=0)
                        self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                        self.y = torch.cat((self.y, targets.view(-1, 1)), dim=0)
                        loss = self.loss(out, targets).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                        if perBatch:
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

            torch.cuda.empty_cache()
            if self.linear_layer:
                self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads

    def _update_grads(self, grads_curr):
        """
        Update the gradient values
        Parameters
        ----------
        grad_currX: OrderedDict, optional
            Gradients of the current element (default: None)
        perClass: bool
            if True, the function computes the validation gradients using perclass dataloaders
        perBatch: bool
            if True, the function computes the validation gradients of each mini-batch
        """
        # if (perBatch and perClass):
        #     raise ValueError("perBatch and perClass are mutually exclusive. Only one of them can be true at a time")
        self.model.zero_grad()
        embDim = self.model.get_embedding_dim()
        
        out_vec = self.init_out - (
                self.eta * grads_curr[0:self.num_classes].view(1, -1).expand(self.init_out.shape[0], -1))

        if self.linear_layer:
            out_vec = out_vec - (self.eta * torch.matmul(self.init_l1, grads_curr[self.num_classes:].view(
                self.num_classes, -1).transpose(0, 1)))

        loss = self.loss(out_vec, self.y.view(-1)).sum()
        l0_grads = torch.autograd.grad(loss, out_vec)[0]
        if self.linear_layer:
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes)
        if self.selection_type == 'PerBatch':
            b = int(self.y.shape[0]/self.valloader.batch_size)
            l0_grads = torch.chunk(l0_grads, b, dim=0)
            new_t = []
            for i in range(len(l0_grads)):
                new_t.append(torch.mean(l0_grads[i], dim=0).view(1, -1))
            l0_grads = torch.cat(new_t, dim=0)
            if self.linear_layer:
                l1_grads = torch.chunk(l1_grads, b, dim=0)
                new_t = []
                for i in range(len(l1_grads)):
                    new_t.append(torch.mean(l1_grads[i], dim=0).view(1, -1))
                l1_grads = torch.cat(new_t, dim=0)
        torch.cuda.empty_cache()
        if self.linear_layer:
            self.grads_val_curr = torch.mean(torch.cat((l0_grads, l1_grads), dim=1), dim=0)
        else:
            self.grads_val_curr = torch.mean(l0_grads, dim=0)
        return self.grads_val_curr

    def phi_j(self, A_len, n, item1, item2, item3, item4, item5, item6, item7, item8, item9, item10):
        with torch.no_grad():
            n_minus_A_len = n - A_len
            if n_minus_A_len == 1:
                return -1 / ((1 + A_len) ** 2) * item1 - 2 / ((1 + A_len) ** 2) * item5 + (2 * A_len + 1) / ((1 + A_len) ** 2 * A_len ** 2) * item7 + 2/(A_len+1) * item8[0] -2 / (n * A_len) * item9
            k_range = np.arange(2, n_minus_A_len + 1, dtype=np.float32)
            term1 = (np.sum(1 / (k_range + A_len) ** 2) + 1 / (1 + A_len) ** 2) / n_minus_A_len
            term2 = np.sum((k_range - 1) / (k_range + A_len) ** 2) / (
                        n_minus_A_len * (n_minus_A_len - 1))
            term3 = np.sum(
                (2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
                                                            n_minus_A_len * (n_minus_A_len - 1))
            term4 = 0 if n_minus_A_len <= 2 else np.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / (
                        (k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
                                                            n_minus_A_len * (n_minus_A_len - 1) * (n_minus_A_len - 2))
            term5 = 0 if A_len == 0 else (np.sum(
                (2 * k_range + 2 * A_len - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) + (2 * A_len + 1) / (
                                                    (1 + A_len) ** 2 * A_len ** 2)) / n_minus_A_len
            term6 = np.sum(1 / (k_range + A_len))+1/(A_len+1)
            term1 = torch.tensor(term1).to(self.device)
            term2 = torch.tensor(term2).to(self.device)
            term3 = torch.tensor(term3).to(self.device)
            term4 = torch.tensor(term4).to(self.device)
            term5 = torch.tensor(term5).to(self.device)
            term6 = torch.tensor(term6).to(self.device)

            phi = (-term1 + 2 * term2 - term3 + 2 * term4) * item1 + (-2 * term2 - 2 * term4) * item2 +  (-2 * term1 - 2 * term3) * item5 + 2 * ((term6 - 1 / n) / (n_minus_A_len - 1)) * item8

            phi += ((term3 - term4) * item3 + term4 * item4 + (2 * term3) * item6 + term5 * item7 -  2 * (term6 / n_minus_A_len - 1 / n) * (1 / (n_minus_A_len - 1)) * item10)

            if A_len > 0:
                phi -= 2 / (n * A_len) * item9
            return phi.to(self.device)
    def shap_value_iterative_selection(self, X, alpha, bud): 
        with torch.no_grad():
            N, d = X.shape
            if self.shapley_values is None:
                self.shapley_values = torch.zeros(N).to(self.device)
            A = []  # Initial empty set A
            gamma_list = []
            NA = set(range(N)) - set(A)
            NA_list = list(NA)
            x_j_sum = torch.sum(X, dim=0).to(self.device)
            x_i_sum1 = torch.zeros_like(x_j_sum).to(self.device)
            
            v1 = X.pow(2).sum(dim=1).to(self.device)
            v2 = torch.mv(X,x_j_sum).to(self.device)
            v3 = torch.sum(X ** 2).to(self.device)
            v4 = x_j_sum.pow(2).sum(dim=0).to(self.device)
            v5 = torch.mv(X, x_i_sum1).to(self.device)
            v6 = torch.dot(x_j_sum, x_i_sum1).to(self.device)
            v7 = x_i_sum1.pow(2).sum(dim=0).to(self.device)
            v8 = torch.mv(X, alpha).to(self.device)
            v9 = torch.dot(x_i_sum1, alpha).to(self.device)
            v10 = torch.dot(x_j_sum, alpha).to(self.device)
            # Loop over until A reaches desired size
            phis_max = -1
            while len(A) < bud:
                # print(1e-4 * alpha.pow(2).sum())
                phis = self.phi_j(len(A), N, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)[NA_list]
                # print(phis.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum1/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))
                # phis = self.phi_j(len(A), N, v1[NA_list], v2[NA_list], v3, v4, v5[NA_list], v6, v7, v8[NA_list], v9, v10)
                phis_max = phis.max()
                # if phis_max<=0:
                #     break
                # else:
                if 1==1:
                    # phis = (phis>0)*phis
                    # phis = (phis-phis.min())/(phis.max()-phis.min())
                    # phis = phis + alpha.norm()**2
                    # prob_dist = phis / torch.sum(phis)
                    # phis = phis/phis.max()
                    # prob_dist = F.softmax(phis, dim=0)

                    # 从概率分布中采样一个数据位置
                    # max_phi_idx = NA_list[np.random.choice(len(NA_list), size = 1, p = prob_dist.cpu().numpy())[0]]
                    # print(max_phi_idx)
                    # max_phi_idx = NA_list[torch.multinomial(prob_dist, 1).item()]

                    max_phi_idx = NA_list[phis.argmax().item()]
                    A.append(max_phi_idx)
                    NA_list.remove(max_phi_idx)
                    # self.shapley_values[max_phi_idx]+=phis_max
                    # gamma_list.append(self.shapley_values[max_phi_idx]) # cifar10: 0.7303
                    gamma_list.append(max_phi_idx)
                    choosen_X = X[max_phi_idx].to(self.device)
                    x_j_sum -= choosen_X
                    x_i_sum1 += choosen_X
                    v2_v5_item = torch.mv(X, choosen_X).to(self.device)
                    v2 -= v2_v5_item
                    v5 += v2_v5_item
                    v3 -= choosen_X.pow(2).sum(dim=0).to(self.device)
                    v4 = x_j_sum.pow(2).sum(dim=0).to(self.device)
                    v6 = torch.dot(x_j_sum, x_i_sum1).to(self.device)
                    v7 = x_i_sum1.pow(2).sum(dim=0).to(self.device)
                    v9_v10_item = torch.dot(choosen_X, alpha).to(self.device)
                    v9 += v9_v10_item
                    v10 -= v9_v10_item
                    # if (x_i_sum1/len(A) - alpha).pow(2).sum() < 1e-2 * alpha.pow(2).sum():
                    #     break

            gamma_list = torch.ones(len(A)).tolist()
            # # gamma_list = self.shapley_values[A].tolist() # cifar10:  0.1002
            # print("[len(A)]:", len(A), "[max_phi]:", phis_max, "[diff]:", (x_i_sum1/len(A) - alpha).pow(2).sum())
            # gamma_list = torch.tensor(gamma_list).to(self.device)
            # gamma_list = gamma_list - gamma_list.min()
            # self.shapley_values[A] = (self.shapley_values[A])*(1-1/self.time)+ (gamma_list/gamma_list.sum())/self.time
            # # gamma_list = (gamma_list - gamma_list.min()).tolist()
            # gamma_list = self.shapley_values[A].tolist()
            return A, gamma_list, NA_list, x_i_sum1
    
    def shap_value_selection(self, X, alpha, bud): 
        with torch.no_grad():
            N, d = X.shape
            if self.shapley_values is None:
                self.shapley_values = torch.zeros(N).to(self.device)
            A = []  # Initial empty set A
            gamma_list = []
            NA = set(range(N)) - set(A)
            NA_list = list(NA)
            x_i_sum1 = torch.zeros_like(alpha).to(self.device)

            # Loop over until A reaches desired size
            phis_max = -1
            while len(A) < bud:
                # print(1e-4 * alpha.pow(2).sum())
                # phis = self.shap_value_evaluation(X[NA_list], (alpha*self.trainloader.batch_size-x_i_sum1)/(self.trainloader.batch_size-len(A)) )
                phis = self.shap_value_evaluation(X[NA_list], (alpha*self.trainloader.batch_size-x_i_sum1)/(self.trainloader.batch_size-len(A)) )
                prob_dist = F.softmax(phis, dim=0)
                max_phi_idx = NA_list[np.random.choice(len(NA_list), size = 1, p = prob_dist.cpu().numpy())[0]]
                # if phis.max()<0:
                #     break
                # max_phi_idx = NA_list[phis.argmax().item()] 
                A.append(max_phi_idx)
                NA_list.remove(max_phi_idx)
                gamma_list.append(max_phi_idx)
                choosen_X = X[max_phi_idx].to(self.device)
                x_i_sum1 += choosen_X

            gamma_list = torch.ones(len(A)).tolist()
            return A, gamma_list, NA_list, x_i_sum1
        
    def suppleyment_greedy(self, X, alpha, bud, x_i_sum1=None, A_len=0):
        if x_i_sum1 is None:
            x_i_sum1 = torch.zeros_like(alpha)
        _N, d = X.shape
        _A = []  # Initial empty set _A
        _A_len = 0
        NA = set(range(_N)) - set(_A)
        NA_list = list(NA)
        v1 = X.pow(2).sum(dim=1).to(self.device)
        gains_max = -1
        i_sum = torch.zeros_like(x_i_sum1)
        while _A_len < bud:
            self.sum_1 -= 1/(len(NA_list)+1)
            self.sum_2 -= 1/(len(NA_list)+1)**2
            gains = -(v1+2*torch.mv(X, x_i_sum1-(_A_len+A_len+1)*alpha))/((_A_len+A_len+1)**2)
            if  A_len + _A_len > 0:
                gains += (2*(_A_len+A_len)+1)/((_A_len+A_len+1)**2 * (_A_len+A_len)**2)*x_i_sum1.pow(2).sum(dim=0) \
                    - 2/((_A_len+A_len+1)*(_A_len+A_len))*torch.dot(alpha,x_i_sum1)
            # gains = torch.mv(X,alpha)
            gains = gains[NA_list]
            gains_max = gains.max()
            
            # # gains = (gains>0)*gains
            # # prob_dist = gains / torch.sum(gains)
            # prob_dist = F.softmax(gains, dim=0)
            # # 从概率分布中采样一个数据位置
            # max_gain_idx = NA_list[np.random.choice(len(NA_list), size = 1, p = prob_dist.cpu().numpy())[0]]
            
            max_gain_idx = NA_list[gains.argmax().item()] 

            # if gains_max<=0:
            #     break
            # else:

            _A.append(max_gain_idx)
            _A_len+=1
            NA_list.remove(max_gain_idx)
            x_i_sum1 += X[max_gain_idx]
            i_sum += X[max_gain_idx]
            
        gamma_list = torch.ones(_A_len).tolist()

        # self.draw_gradients(X[_A], alpha, "greedy")
        # print("---len(_A)---:", _A_len, "max_gain", gains_max)
        return _A, gamma_list, NA_list, i_sum
    
    


    def select(self, budget, cur_epoch, model_params):
        self.time += 1
        """
        Apply OMP Algorithm for data selection

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters

        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        omp_start_time = time.time()
        self.update_model(model_params)

        if self.selection_type == 'PerClass':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)

                self.compute_gradients(self.valid, perBatch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                if self.valid:
                    mean_grad = torch.mean(self.val_grads_per_elem, dim=0)
                else:
                    mean_grad = torch.mean(trn_gradients, dim=0)

                idxs_temp, gammas_temp,_,_ = self.shap_value_iterative_selection(trn_gradients,
                                                         mean_grad,
                                                         math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        elif self.selection_type == 'PerBatch':
            self.compute_gradients(self.valid, perBatch=True, perClass=False)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            if self.valid:
                sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            else:
                sum_val_grad = torch.sum(trn_gradients, dim=0)
            idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                     sum_val_grad, math.ceil(budget / self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        elif self.selection_type == 'PerClassPerGradient':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            embDim = self.model.get_embedding_dim()
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                self.compute_gradients(self.valid, perBatch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                tmp_gradients = trn_gradients[:, i].view(-1, 1)
                tmp1_gradients = trn_gradients[:,
                                 self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)

                if self.valid:
                    val_gradients = self.val_grads_per_elem
                    tmp_gradients = val_gradients[:, i].view(-1, 1)
                    tmp1_gradients = val_gradients[:,
                                     self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                    val_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
                    sum_val_grad = torch.sum(val_gradients, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)

                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                         sum_val_grad,
                                                         math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)
        
        elif self.selection_type == 'shap':
            if self.selection_only_once is False:
                self.compute_gradients(self.valid, perBatch=False, perClass=False)
                idxs = []
                gammas = []
                trn_gradients = self.grads_per_elem
                if self.valid:
                    mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(self.device)
                else:
                    mean_grad = torch.mean(trn_gradients, dim=0).to(self.device)
                # self.select_every = 1
                # step_mean_grad = self._update_grads(mean_grad)
                # print(step_mean_grad.norm(),mean_grad.norm(),(mean_grad-step_mean_grad).norm(), self.eta * self.lipschitz)
                # mean_grad += step_mean_grad/(self.eta * self.lipschitz)
                # mean_grad *= 2
                self.culmulate_grad = torch.zeros_like(mean_grad)
                NA_list = list(set(range(trn_gradients.shape[0])))
                
                self.sum_1 = torch.sum(1 / torch.arange(1, trn_gradients.shape[0] + 1+1).float())
                self.sum_2 = torch.sum(1 / torch.arange(1, trn_gradients.shape[0] + 1+1).float() ** 2)

                for iteration in range(math.ceil(budget/self.trainloader.batch_size)):
                    if len(NA_list) < self.trainloader.batch_size:
                        break
                # while len(idxs) < budget and len(NA_list) >= self.trainloader.batch_size:
                    _idxs1, _gammas1, _NA_list, x_i_sum1= self.shap_value_iterative_selection(trn_gradients[NA_list],
                                                            mean_grad,
                                                            math.ceil(self.trainloader.batch_size))
                    # _idxs1, _gammas1, _NA_list, x_i_sum1= self.shap_value_selection(trn_gradients[NA_list],
                    #                                         mean_grad*(iteration+1) - self.culmulate_grad/self.trainloader.batch_size,
                    #                                         math.ceil(self.trainloader.batch_size * p))
                    # _idxs1, _gammas1, _NA_list, x_i_sum1= self.shap_value_selection(trn_gradients[NA_list],
                    #                                         mean_grad,
                    #                                         math.ceil(self.trainloader.batch_size ))
                    _idxs1 = (torch.tensor(NA_list)[_idxs1]).tolist()
                    NA_list = (torch.tensor(NA_list)[_NA_list]).tolist()
                    # self.culmulate_grad += x_i_sum1
                    # _idxs2, _gammas2, _NA_list2, x_i_sum2 = self.suppleyment_greedy(trn_gradients[NA_list], 
                    #                                         mean_grad,
                    #                                         math.ceil(self.trainloader.batch_size-len(_idxs1)),
                    #                                         x_i_sum1,
                    #                                         len(_idxs1))
                    # _idxs2 = (torch.tensor(NA_list)[_idxs2]).tolist()
                    # NA_list = (torch.tensor(NA_list)[_NA_list2]).tolist()
                    # self.culmulate_grad += x_i_sum2
                    
                    idxs.extend(_idxs1)
                    # idxs.extend(_idxs2)

                    gammas.extend(_gammas1)
                    # gammas.extend(_gammas2)    
            else:
                if self.gammas is not None:
                    gammas = self.gammas
                    idxs = self.idxs
                else:
                    self.compute_gradients(self.valid, perBatch=False, perClass=False)
                    idxs = []
                    gammas = []
                    trn_gradients = self.grads_per_elem
                    if self.valid:
                        mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(self.device)
                    else:
                        mean_grad = torch.mean(trn_gradients, dim=0).to(self.device)
                    
                    NA_list = list(set(range(trn_gradients.shape[0])))
                    # for _ in range(10):
                    while len(idxs) < budget and len(NA_list) >= self.trainloader.batch_size:
                        for _ in range(2):
                            _idxs1, _gammas1, _NA_list, x_i_sum1= self.shap_value_iterative_selection(trn_gradients[NA_list],
                                                                    mean_grad,
                                                                    math.ceil(self.trainloader.batch_size * p/2))
                            _idxs1 = (torch.tensor(NA_list)[_idxs1]).tolist()
                            NA_list = (torch.tensor(NA_list)[_NA_list]).tolist()
                            
                            _idxs2, _gammas2, _NA_list2 = self.suppleyment_greedy(trn_gradients[NA_list], 
                                                                    mean_grad,
                                                                    math.ceil(self.trainloader.batch_size/2-len(_idxs1)),
                                                                    x_i_sum1,
                                                                    len(_idxs1))
                            _idxs2 = (torch.tensor(NA_list)[_idxs2]).tolist()
                            NA_list = (torch.tensor(NA_list)[_NA_list2]).tolist()
                            
                            idxs.extend(_idxs1)
                            idxs.extend(_idxs2)

                            gammas.extend(_gammas1)
                            gammas.extend(_gammas2) 
                    self.gammas = gammas
                    self.idxs = idxs
                    
            # self.compute_gradients(self.valid, perBatch=False, perClass=False)
            # idxs = []
            # gammas = []
            # trn_gradients = self.grads_per_elem
            # if self.valid:
            #     mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(self.device)
            # else:
            #     mean_grad = torch.mean(trn_gradients, dim=0).to(self.device)
            
            # NA_list = list(set(range(trn_gradients.shape[0])))
            # # for _ in range(10):
            # while len(idxs) < budget and len(NA_list) >= self.trainloader.batch_size:
            #     for _ in range(2):
            #         _idxs1, _gammas1, _NA_list, x_i_sum1= self.shap_value_iterative_selection(trn_gradients[NA_list],
            #                                                 mean_grad,
            #                                                 math.ceil(self.trainloader.batch_size * p/2))
            #         _idxs1 = (torch.tensor(NA_list)[_idxs1]).tolist()
            #         NA_list = (torch.tensor(NA_list)[_NA_list]).tolist()
                    
            #         _idxs2, _gammas2, _NA_list2 = self.suppleyment_greedy(trn_gradients[NA_list], 
            #                                                 mean_grad,
            #                                                 math.ceil(self.trainloader.batch_size/2-len(_idxs1)),
            #                                                 x_i_sum1,
            #                                                 len(_idxs1))
            #         _idxs2 = (torch.tensor(NA_list)[_idxs2]).tolist()
            #         NA_list = (torch.tensor(NA_list)[_NA_list2]).tolist()
                    
            #         idxs.extend(_idxs1)
            #         idxs.extend(_idxs2)

            #         gammas.extend(_gammas1)
            #         gammas.extend(_gammas2)



                # _, _gammas_1_2 = self.shap_value_evaluation(trn_gradients,
                #                                         mean_grad,
                #                                         _idxs1+_idxs2)
                # gammas.extend(_gammas_1_2)
                
                # _idxs1, _gammas1, _NA_list = self.shap_value_plus_instance_gain_iterative_selection(trn_gradients[NA_list],
                #                                         mean_grad,
                #                                         math.ceil(self.trainloader.batch_size) )
                # _idxs1 = (torch.tensor(NA_list)[_idxs1]).tolist()
                # NA_list = (torch.tensor(NA_list)[_NA_list]).tolist()
                # idxs.extend(_idxs1)
                # gammas.extend(_gammas1)
                


            # idxs, gammas, NA_list, x_i_sum1= self.shap_value_iterative_selection(trn_gradients,
            #                                             mean_grad,
            #                                             math.ceil(budget*p))
            # idxs2, gammas2, NA_list2, x_i_sum2= self.shap_value_iterative_selection(trn_gradients[NA_list],
            #                                             mean_grad,
            #                                             math.ceil(budget*p))
            # idxs2 = (torch.tensor(NA_list)[idxs2]).tolist()
            # idxs2, gammas2 = self.suppleyment_greedy(trn_gradients[NA_list], 
            #                                          mean_grad,
            #                                          math.ceil(budget-len(idxs)),
            #                                          NA_list,
            #                                          x_i_sum1,
            #                                          len(idxs))
            # # 使用集合(set)来检查重复元素
            # if len(set(idxs + idxs2)) < len(idxs) + len(idxs2):
            #     print("存在重复元素")
            # else:
            #     print("没有重复元素")
            # idxs.extend(idxs2)
            # gammas.extend(gammas2)
            
            # idxs, gammas, _ = self.shap_value_plus_instance_gain_iterative_selection(trn_gradients,
            #                                             mean_grad,
            #                                             math.ceil(budget))
            
            # idxs, gammas = self.shap_value_evaluation(trn_gradients,
            #                                             mean_grad, 
            #                                             idxs)
            
            # idxs, gammas = self.shap_value_evaluation(trn_gradients,
            #                                             mean_grad)
            # idxs, gammas = self.shap_value_iterative_selection(trn_gradients,
            #                                             mean_grad,
            #                                             -1)
            
            torch.cuda.empty_cache()
            # if cur_epoch < 100:
            #     idxs, gammas = self.shap_value_iterative_selection(trn_gradients,
            #                                             mean_grad,
            #                                             -2)
            #     # 将 Shapley 值保存到文件
            #     np.savetxt('shapley_values_100.txt', self.shapley_values.cpu())
            # else:
            # # if 1==1:
            #     idxs = np.array([i for i in range(trn_gradients.shape[0])])
            #     gammas = np.loadtxt('shapley_values_100.txt')
            #     gammas = gammas - np.min(gammas)
                # gammas = gammas*(gammas>0)
                # gammas = np.where(gammas > 0, 1, 0)
                # print(len(gammas))
        # full best accuracy: 0.9539 
        # -min 0.9543
        
        diff = budget - len(idxs)
        self.logger.debug("Random points added: %d ", diff)

        if self.selection_type == "shap":
            rand_indices = np.random.permutation(int(len(idxs)/self.trainloader.batch_size))
            split_idxs =  np.split(np.array(idxs), len(idxs)/self.trainloader.batch_size)
            idxs = list(np.concatenate([split_idxs[i] for i in rand_indices]))
            # idxs = list(np.array(idxs)[rand_indices])
            split_gammas = np.split(np.array(gammas), int(len(gammas)/self.trainloader.batch_size))
            gammas = list(np.concatenate([split_gammas[i] for i in rand_indices]))
            # gammas = list(np.array(gammas)[rand_indices])
        
        if diff > 0:
            print("diff>0")
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)
        if self.selection_type in ["PerClass", "PerClassPerGradient"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])

        idxs = [int(x) for x in idxs]
        omp_end_time = time.time()
        self.logger.debug("SHAPIS algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)
    
    def shap_value_plus_instance_gain_iterative_selection(self, X, alpha, bud):
        with torch.no_grad():
            N, d = X.shape
            # if self.shapley_values is None:
            #     self.shapley_values = torch.zeros(N).to(self.device)
            A = []  # Initial empty set A
            A_len = 0
            gamma_list = []
            NA = set(range(N)) - set(A)
            NA_list = list(NA)
            x_j_sum = torch.sum(X, dim=0).to(self.device)
            x_i_sum1 = torch.zeros_like(x_j_sum).to(self.device)
            
            v1 = X.pow(2).sum(dim=1).to(self.device)
            v2 = torch.mv(X,x_j_sum).to(self.device)
            v3 = torch.sum(X ** 2).to(self.device)
            v4 = x_j_sum.pow(2).sum(dim=0).to(self.device)
            v5 = torch.mv(X, x_i_sum1).to(self.device)
            v6 = torch.dot(x_j_sum, x_i_sum1).to(self.device)
            v7 = x_i_sum1.pow(2).sum(dim=0).to(self.device)
            v8 = torch.mv(X, alpha).to(self.device)
            v9 = torch.dot(x_i_sum1, alpha).to(self.device)
            v10 = torch.dot(x_j_sum, alpha).to(self.device)
            # Loop over until A reaches desired size
            phis_max = -1
            while A_len < bud:
                p =  1- (A_len+1)/bud
                phis = self.phi_j(A_len, N, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)[NA_list]
                gains = -(v1+2*torch.mv(X, x_i_sum1-(A_len+1)*alpha))/((A_len+1)**2)
                if  A_len > 0:
                    gains += (2*(A_len)+1)/((A_len+1)**2 * (A_len)**2)*x_i_sum1.pow(2).sum(dim=0) \
                    - 2/((A_len+1)*(A_len))*torch.dot(alpha,x_i_sum1)
                gains = gains[NA_list]
                
                # phis = (phis-phis.min())/(phis.max()-phis.min())
                # gains = (gains-gains.min())/(gains.max()-gains.min())
                
                phis = F.softmax(phis, dim=0)
                gains = F.softmax(gains, dim=0)
                phis = p * phis + (1-p) * gains

                phis_max = phis.max()
                # if phis_max<=0:
                #     break
                # else:
                max_phi_idx = NA_list[phis.argmax().item()]
                A.append(max_phi_idx)
                A_len += 1
                NA_list.remove(max_phi_idx)
                # self.shapley_values[max_phi_idx]+=phis_max
                # gamma_list.append(self.shapley_values[max_phi_idx]) # cifar10: 0.7303
                gamma_list.append(max_phi_idx)
                choosen_X = X[max_phi_idx].to(self.device)
                x_j_sum -= choosen_X
                x_i_sum1 += choosen_X
                v2_v5_item = torch.mv(X, choosen_X).to(self.device)
                v2 -= v2_v5_item
                v5 += v2_v5_item
                v3 -= choosen_X.pow(2).sum(dim=0).to(self.device)
                v4 = x_j_sum.pow(2).sum(dim=0).to(self.device)
                v6 = torch.dot(x_j_sum, x_i_sum1).to(self.device)
                v7 = x_i_sum1.pow(2).sum(dim=0).to(self.device)
                v9_v10_item = torch.dot(choosen_X, alpha).to(self.device)
                v9 += v9_v10_item
                v10 -= v9_v10_item
                # if (x_i_sum1/len(A) - alpha).pow(2).sum() < 1e-4 * alpha.pow(2).sum():
                #     break
            gamma_list = torch.ones(A_len).tolist()
            # # gamma_list = self.shapley_values[A].tolist() # cifar10:  0.1002
            print("---len(A)---:", A_len, "max_phi", phis_max)
            # gamma_list = torch.tensor(gamma_list).to(self.device)
            # gamma_list = gamma_list - gamma_list.min()
            # self.shapley_values[A] = (self.shapley_values[A])*(1-1/self.time)+ (gamma_list/gamma_list.sum())/self.time
            # # gamma_list = (gamma_list - gamma_list.min()).tolist()
            # gamma_list = self.shapley_values[A].tolist()
            return A, gamma_list, NA_list

    # def shap_value_iterative_selection(self, X, alpha, bud): 
        # with torch.no_grad():
        #     N, d = X.shape
        #     A = []  # Initial empty set A
        #     gamma = []
        #     NA = set(range(N)) - set(A)
        #     NA_list = list(NA)
        #     x_j_sum = torch.sum(X, dim=0).to(self.device)
        #     x_i_sum1 = torch.zeros_like(x_j_sum).to(self.device)
        #     x_j_2_sum = torch.sum(X ** 2).to(self.device)
        #     X_2 = X.pow(2).sum(dim=1).to(self.device)

        #     # Loop over until A reaches desired size
        #     while len(A) < bud:
        #         x_j = X[NA_list]  # Select the elements not in A
        #         x_j_2 = X_2[NA_list]
        #         phis = self.phi_j(len(A), N, x_j, alpha, x_i_sum1, x_j_sum, x_j_2, x_j_2_sum)
        #         if len(A) == 0:
        #             if self.shapley_values is None:
        #                 self.shapley_values = phis
        #             else:
        #                 self.shapley_values += phis
        #         if phis.max()<=0:
        #             break
        #         else:
        #             max_phi_idx = phis.argmax().item()
        #             max_origin_idx = NA_list[max_phi_idx]
        #             A.append(max_origin_idx)
        #             # gamma.append(phis[max_phi_idx].cpu())
        #             NA_list.remove(max_origin_idx)
        #             x_j_sum -= X[max_origin_idx]
        #             x_i_sum1 += X[max_origin_idx]
        #             x_j_2_sum -= X[max_origin_idx].pow(2).sum(dim=0)

        #     # gamma_list = torch.zeros_like(self.shapley_values)
        #     # # gamma_list[positive_indices] = self.shapley_values[positive_indices]
        #     # gamma_list[A] = 1
        #     # with open("shapis_gamma.txt", "a") as f:
        #     #     print(A,file=f)
        #     #     print(gamma_list,file=f) 
        #     # print(len(A))
        #     # self.draw_gradients(X[A],alpha,"X[A]")
        #     # self.draw_gradients(X,alpha,"X")
                    
        #     gamma_list = torch.ones(len(A)).tolist()
        #     # print("---len(A)---:", len(A))
        #     # gamma_list = gamma
        #     return A, gamma_list

    # def shap_value_evaluation(self, X, alpha, bud, choosen_idxs = None): 
    #     with torch.no_grad():
    #         X = X[choosen_idxs]
    #         n, d = X.shape
    #         x_sum = torch.sum(X, dim=0)
    #         if self.sum_1 is None and self.sum_2 is None: 
    #             sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
    #             sum_2 = torch.sum(1 / torch.arange(1, n + 1).float() ** 2)
    #             self.sum_1 = sum_1
    #             self.sum_2 = sum_2
    #         else:
    #             sum_1 = self.sum_1
    #             sum_2 = self.sum_2
    #         term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
    #                             + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
    #         term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
    #         term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
    #         term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
    #         term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
    #         term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)

    #         shapley_values = term_1 + term_2 + term_3 + term_4 + term_5 + term_6
    #         gamma_list = (shapley_values - shapley_values.min())/(shapley_values.max() - shapley_values.min())
    #         return choosen_idxs, gamma_list.tolist()

    def shap_value_evaluation(self, X, alpha): 
        with torch.no_grad():
            n, d = X.shape
            x_sum = torch.sum(X, dim=0)
            
            self.sum_1 -= 1/(n+1)
            self.sum_2 -= 1/(n+1)**2
            # print(self.sum_1, torch.sum(1 / torch.arange(1, n + 1).float()))
            # print(self.sum_2, torch.sum(1 / torch.arange(1, n + 1).float()**2))
            sum_1 = self.sum_1 
            sum_2 = self.sum_2
            term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
                                + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
            term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
            term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
            term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
            term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
            term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)

            shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6).to(self.device)

            return shapley_values



    def shap_value_evaluation2(self, X, alpha, choosen_idxs = None): 
        if choosen_idxs is not None: # evaluate choosen_idxs
            with torch.no_grad():
                X = X[choosen_idxs]
                n, d = X.shape
                x_sum = torch.sum(X, dim=0)
                if self.sum_1 is None and self.sum_2 is None: 
                    sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
                    sum_2 = torch.sum(1 / torch.arange(1, n + 1).float() ** 2)
                    self.sum_1 = sum_1
                    self.sum_2 = sum_2
                else:
                    sum_1 = self.sum_1
                    sum_2 = self.sum_2
                term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
                                    + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
                term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
                term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
                term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
                term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
                term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)

                shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6).to(self.device)
                
                # if self.shapley_values is None:
                #     self.shapley_values = torch.zeros(n).to(self.device)
                #     self.shapley_values[choosen_idxs] = shapley_values
                # else:
                #     # self.shapley_values[choosen_idxs] = self.shapley_values[choosen_idxs]*(1-1/self.time)+ shapley_values/self.time
                #     # self.shapley_values[choosen_idxs] = self.shapley_values[choosen_idxs]*0.7 + shapley_values*0.3
                #     for index, i in enumerate(choosen_idxs):
                #         if self.shapley_values[i] == 0:
                #             self.shapley_values[i] = shapley_values[index]
                #         else:
                #             self.shapley_values[i] = self.shapley_values[i]*0.7 + shapley_values[index]*0.3
                # gamma_list = shapley_values
                # shapley_values = self.shapley_values[choosen_idxs]
                gamma_list = (shapley_values - shapley_values.min())/(shapley_values.max() - shapley_values.min())
                # gamma_list = F.softmax(shapley_values, dim=0)
                gamma_list = gamma_list/gamma_list.sum() * len(gamma_list)
                # gamma_list = (shapley_values - torch.mean(shapley_values))/(torch.std(shapley_values))
                # gamma_list = torch.sigmoid(gamma_list)
                # gamma_list = (shapley_values - shapley_values.min())/(shapley_values.max() - shapley_values.min())
                return choosen_idxs, gamma_list.tolist()
        else:   # evaluate all
            with torch.no_grad():
                n, d = X.shape
                x_sum = torch.sum(X, dim=0)
                
                
                if self.sum_1 is None and self.sum_2 is None: 
                    sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
                    sum_2 = torch.sum(1 / torch.arange(1, n + 1).float() ** 2)
                    self.sum_1 = sum_1
                    self.sum_2 = sum_2
                else:
                    self.sum_1 -= 1/(n+1)
                    self.sum_2 -= 1/(n+1)**2
                    sum_1 = self.sum_1 
                    sum_2 = self.sum_2
                term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
                                    + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
                term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
                term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
                term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
                term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
                term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)

                shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6).to(self.device)
                # shapley_values = (shapley_values-shapley_values.min())/(shapley_values.max()-shapley_values.min())
                # if self.shapley_values is None:
                #     self.shapley_values = torch.zeros(n)
                #     self.shapley_values[choosen_idxs] = shapley_values
                # else:
                #     # self.shapley_values[choosen_idxs] = self.shapley_values[choosen_idxs]*(1-1/self.time)+ shapley_values/self.time
                #     # self.shapley_values[choosen_idxs] = self.shapley_values[choosen_idxs]*0.7 + shapley_values*0.3
                #     for index, i in enumerate(choosen_idxs):
                #         if self.shapley_values[i] == 0:
                #             self.shapley_values[i] = shapley_values[index]
                #         else:
                #             self.shapley_values[i] = self.shapley_values[i]*0.7 + shapley_values[index]*0.3
                
                # if self.shapley_values is None:
                #     self.shapley_values = shapley_values
                # else:
                #     # self.shapley_values[choosen_idxs] = self.shapley_values[choosen_idxs]*(1-1/self.time)+ shapley_values/self.time
                #     self.shapley_values = self.shapley_values*0.7 + shapley_values*0.3
                # gamma_list  = self.shapley_values
                # gamma_list = (gamma_list - gamma_list.min())/(gamma_list.max() - gamma_list.min())
                # return list(range(n)), gamma_list.tolist()
                
                return shapley_values



    