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

    def phi_j(self, A_len, r,  n, item1, item2, item3, item4, item5, item6, item7, item8, item9, item10):
        with torch.no_grad():
            n_minus_A_len = n - A_len
            r_minus_A_len = r - A_len
            if n_minus_A_len == 1:
                return -1 / ((1 + A_len) ** 2) * item1 - 2 / ((1 + A_len) ** 2) * item5 + (2 * A_len + 1) / ((1 + A_len) ** 2 * A_len ** 2) * item7 + 2/(A_len+1) * item8[0] -2 / (n * A_len) * item9
            k_range = np.arange(2, r_minus_A_len + 1, dtype=np.float32)
            term1 = (np.sum(1 / (k_range + A_len) ** 2) + 1 / (1 + A_len) ** 2) / n_minus_A_len
            term2 = np.sum((k_range - 1) / (k_range + A_len) ** 2) / (
                        n_minus_A_len * (n_minus_A_len - 1))
            term3 = np.sum(
                (2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
                                                            n_minus_A_len * (n_minus_A_len - 1))
            term4 = 0 if r_minus_A_len <= 2 else np.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / (
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
            phi = (-term1 + 2 * term2 - term3 + 2 * term4) * item1 + (-2 * term2 - 2 * term4) * item2 +  (-2 * term1 - 2 * term3) * item5 + 2 * (term6 / n_minus_A_len + (term6 - r_minus_A_len/r) / (n_minus_A_len*(n_minus_A_len - 1)) ) * item8

            phi += (term3 - term4) * item3 + term4 * item4 + (2* term3) * item6 + term5 * item7 -  2 * (term6 - r_minus_A_len/r) / (n_minus_A_len*(n_minus_A_len - 1)) * item10

            if A_len > 0:
                phi -= 2 *r_minus_A_len/ (r * A_len * n_minus_A_len) * item9
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
                phis = self.phi_j(len(A), bud, N, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)[NA_list]
                
                _, _items = torch.topk(phis , k = bud-len(A), largest=True, dim=0)
                max_phi_idx = NA_list[_items[-1]]
                # max_phi_idx = NA_list[phis.argmax().item()]

                # phis_max = phis.max()
                # prob_dist = F.softmax(phis/(self.time), dim=0)
                # prob_dist = (phis-phis.min())/(phis.max()-phis.min())
                # prob_dist /= prob_dist.sum()
                # max_phi_idx = NA_list[np.random.choice(len(NA_list), size = 1, p = prob_dist.cpu().numpy())[0]]
                
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

            gamma_list = torch.ones(len(A)).tolist()
            return A, gamma_list, NA_list, x_i_sum1
    


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

                self.culmulate_grad = torch.zeros_like(mean_grad)
                self.X_sum = torch.sum(trn_gradients, dim=0).to(self.device)
                self.X_row_norm_2 = (torch.norm(trn_gradients, dim=1, p=2) ** 2).to(self.device)
                self.X_norm_2 = torch.sum(self.X_row_norm_2).to(self.device)
                r = int(budget * len(trn_subset_idx) / self.N_trn)
                self.sum_1 = torch.sum(1 / torch.arange(1, r + 1).float()) 
                self.sum_2 = torch.sum(1 / torch.arange(1, r + 1).float()**2) 
                _idxs1, _gammas1,_,_= self.shap_value_iterative_selection(trn_gradients,
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
                _idxs1, _gammas1,_,_ = self.shap_value_iterative_selection(trn_gradients,
                                                        # (mean_grad*math.ceil(budget/self.trainloader.batch_size)*self.trainloader.batch_size - self.culmulate_grad)/(self.trainloader.batch_size*(math.ceil(budget/self.trainloader.batch_size) - iteration))
                                                        # (mean_grad*(iteration+1) - self.culmulate_grad/self.trainloader.batch_size).to(self.device),
                                                        mean_grad,
                                                        r)
                _gammas1 = self.shap_value_evaluation(trn_gradients[_idxs1],mean_grad)
                _gammas1 = _gammas1 - _gammas1.min()+1
                _gammas1 = (_gammas1/_gammas1.sum()*len(_idxs1)).tolist()
                idxs.extend(list(np.array(trn_subset_idx)[_idxs1]))
                gammas.extend(_gammas1)

 
        
        elif self.selection_type == 'shap':
            self.compute_gradients(self.valid, perBatch=False, perClass=False)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            if self.valid:
                mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(self.device)
            else:
                mean_grad = torch.mean(trn_gradients, dim=0).to(self.device)
            self.culmulate_grad = torch.zeros_like(mean_grad)
            NA_list = list(set(range(trn_gradients.shape[0])))
            for iteration in range(math.ceil(budget/self.trainloader.batch_size)):
                if len(NA_list) < self.trainloader.batch_size:
                    break
                _idxs1, _gammas1, _NA_list, x_i_sum1= self.shap_value_iterative_selection(trn_gradients[NA_list],
                                                        mean_grad,
                                                        math.ceil(self.trainloader.batch_size))
                _idxs1 = (torch.tensor(NA_list)[_idxs1]).tolist()
                NA_list = (torch.tensor(NA_list)[_NA_list]).tolist()
                idxs.extend(_idxs1)
                gammas.extend(_gammas1) 
            
            
            torch.cuda.empty_cache()
        
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
        # mean_grad = torch.mean(self.grads_per_elem, dim=0).to(self.device)
        # gammas = self.shap_value_evaluation(self.grads_per_elem[idxs], mean_grad).cpu()
        # gammas -= gammas.min()
        omp_end_time = time.time()
        self.logger.debug("SHAPIS algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)
    
    def shap_value_evaluation(self, X, alpha): 
        with torch.no_grad():
            n, d = X.shape
            x_sum = torch.sum(X, dim=0)
            
            sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
            sum_2 = torch.sum(1 / torch.arange(1, n + 1).float()**2)
            
            term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
                                + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
            term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
            term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
            term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
            term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
            term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)

            shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6).to(self.device)

            return shapley_values