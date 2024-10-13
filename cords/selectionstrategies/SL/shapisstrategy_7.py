import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import copy
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
        # self.r_1 = [torch.sum(1 / torch.arange(1, r + 1).float()) for r in range(self.trainloader.batch_size+1)]
        # self.r_2 = [torch.sum(1 / torch.arange(1, r + 1).float()**2) for r in range(self.trainloader.batch_size+1)]
        

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


    def shap_value_evaluation(self, X, alpha, x_xum, r, X_row_norm_2, origin_NA_list): 
        with torch.no_grad():
            n = len(origin_NA_list)
            x_sum = self.X_sum - self.culmulate_grad
            # sum_1 = self.r_1[r]
            # sum_2 = self.r_2[r]
            # sum_1 = torch.sum(1 / torch.arange(1, r + 1).float()) 
            # sum_2 = torch.sum(1 / torch.arange(1, r + 1).float()**2) 
            sum_1 = self.sum_1
            sum_2 = self.sum_2

            term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / r)
                                + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / r)) * X_row_norm_2
            term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / r + 1 / (n * r)) * torch.mv(X, x_sum)
            term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / r) * torch.norm(x_sum, p=2) ** 2
            term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / r) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / r)) * self.X_norm_2
            term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
            term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)
            shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6).to(self.device) # cifar10 0.05(0.8734) 0.1(0.9128). imagenet: 0.1(0.4658)

            # shapley_values = -X_row_norm_2 + 2* torch.mv(X, alpha).to(self.device)               # cifar10 dot:      0.05(0.8721) and 0.1(0.9073). imagenet: 0.1(0.4776)
            # shapley_values = -torch.norm(X-alpha, dim=1, p=2).to(self.device)   # Euler distance:   0.05(0.7984) and 0.1(0.8899). imagenet:0.1(0.4054)            
            # shapley_values = X_row_norm_2
            # shapley_values = torch.mv(X, alpha)
            # shapley_values = torch.div(torch.mv(X+x_xum, alpha),torch.norm(X+x_xum, dim=1, p=2))
            return shapley_values[origin_NA_list]

    def shap_value_selection(self, trn, alpha, bud): 
        with torch.no_grad():
            A = []  # Initial empty set A
            NA_list = list(set(range(trn.shape[0])))
            x_i_sum1 = torch.zeros_like(alpha).to(self.device)
            X_row_norm_2 = self.X_row_norm_2

            # Loop over until A reaches desired size
            while len(A) < bud:
                # phis = self.shap_value_evaluation(X[NA_list], (alpha*self.trainloader.batch_size-x_i_sum1)/(self.trainloader.batch_size-len(A)),self.trainloader.batch_size-len(A) )
                phis = self.shap_value_evaluation(trn, 
                                                #   (alpha*(len(A)+1)-x_i_sum1), 
                                                #   (alpha*bud-x_i_sum1)/(self.trainloader.batch_size-len(A)) ,
                                                alpha,
                                                x_i_sum1,
                                                bud-len(A),
                                                # len(origin_NA_list),
                                                X_row_norm_2,
                                                NA_list)
                # print(torch.sum(phis)-torch.norm(alpha*(len(A)+1)-x_i_sum1)**2 + torch.norm(alpha*(len(A)+1)-x_i_sum1 - torch.mean(X,dim=0))**2)
                NA_max_phi_idx = phis.argmax().item()
                max_phi_idx = NA_list[NA_max_phi_idx]
                A.append(max_phi_idx)
                choosen_X = trn[max_phi_idx].to(self.device)
                NA_list.pop(NA_max_phi_idx)

                x_i_sum1 += choosen_X
                self.culmulate_grad += choosen_X
                self.X_norm_2 -= torch.norm(choosen_X, p=2)**2

            gamma_list = torch.ones(len(A)).tolist()
            
            # print(alpha.norm().cpu(), (x_i_sum1/bud).norm().cpu(), (alpha - x_i_sum1/bud).norm().cpu(), (alpha - x_i_sum1/bud).norm().cpu()/alpha.norm().cpu())
            # print((alpha - x_i_sum1/bud).norm().cpu()/alpha.norm().cpu(), ((torch.dot(alpha,x_i_sum1/bud))/(alpha.norm()*(x_i_sum1/bud).norm())).cpu())

            return A, gamma_list

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
        self.compute_gradients(self.valid, perBatch=False, perClass=False)
        idxs = []
        gammas = []
        
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

                self.culmulate_grad = torch.zeros_like(mean_grad)
                self.X_sum = torch.sum(trn_gradients, dim=0).to(self.device)
                self.X_row_norm_2 = (torch.norm(trn_gradients, dim=1, p=2) ** 2).to(self.device)
                self.X_norm_2 = torch.sum(self.X_row_norm_2).to(self.device)
                r = int(budget * len(trn_subset_idx) / self.N_trn)
                self.sum_1 = torch.sum(1 / torch.arange(1, r + 1).float()) 
                self.sum_2 = torch.sum(1 / torch.arange(1, r + 1).float()**2) 
                _idxs1, _gammas1= self.shap_value_selection(trn_gradients,
                                                        # (mean_grad*math.ceil(budget/self.trainloader.batch_size)*self.trainloader.batch_size - self.culmulate_grad)/(self.trainloader.batch_size*(math.ceil(budget/self.trainloader.batch_size) - iteration))
                                                        # (mean_grad*(iteration+1) - self.culmulate_grad/self.trainloader.batch_size).to(self.device),
                                                        mean_grad,
                                                        r)
                idxs.extend(list(np.array(trn_subset_idx)[_idxs1]))
                gammas.extend(_gammas1)
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
                trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1).to(self.device)
                if self.valid:
                    mean_grad = torch.mean(self.val_grads_per_elem, dim=0)
                else:
                    mean_grad = torch.mean(trn_gradients, dim=0)
                self.culmulate_grad = torch.zeros_like(mean_grad)
                self.X_sum = torch.sum(trn_gradients, dim=0).to(self.device)
                self.X_row_norm_2 = (torch.norm(trn_gradients, dim=1, p=2) ** 2).to(self.device)
                self.X_norm_2 = torch.sum(self.X_row_norm_2).to(self.device)
                r = int(budget * len(trn_subset_idx) / self.N_trn)
                self.sum_1 = torch.sum(1 / torch.arange(1, r + 1).float()) 
                self.sum_2 = torch.sum(1 / torch.arange(1, r + 1).float()**2) 
                _idxs1, _gammas1 = self.shap_value_selection(trn_gradients,
                                                        # (mean_grad*math.ceil(budget/self.trainloader.batch_size)*self.trainloader.batch_size - self.culmulate_grad)/(self.trainloader.batch_size*(math.ceil(budget/self.trainloader.batch_size) - iteration))
                                                        # (mean_grad*(iteration+1) - self.culmulate_grad/self.trainloader.batch_size).to(self.device),
                                                        mean_grad,
                                                        r)
                idxs.extend(list(np.array(trn_subset_idx)[_idxs1]))
                gammas.extend(_gammas1)
        torch.cuda.empty_cache()
        diff = budget - len(idxs)
        self.logger.info("Random points added: %d ", diff)

        if self.selection_type == "shap":
            rand_indices = np.random.permutation(int(len(idxs)/self.trainloader.batch_size))
            split_idxs =  np.split(np.array(idxs), len(idxs)/self.trainloader.batch_size)
            idxs = list(np.concatenate([split_idxs[i] for i in rand_indices]))
            split_gammas = np.split(np.array(gammas), int(len(gammas)/self.trainloader.batch_size))
            gammas = list(np.concatenate([split_gammas[i] for i in rand_indices]))
        
        if diff > 0:
            # print("diff>0")
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
    
    

    