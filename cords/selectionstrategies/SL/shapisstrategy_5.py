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
        self.time = 0
        self.selection_only_once = False
        self.gammas = None
        self.idxs = None
        self.lipschitz = 20

    def r_restirctive_shap_value_evaluation(self, X, n, A, r, alpha, X_row_norm_2, X_N_sum, X_A_sum, X_N_norm_2, X_A_norm_2, X_i_dot_alpha, X_N_sum_dot_alpha, X_i_dot_X_N_sum, origin_NA_list): 
        with torch.no_grad(): # really poor performance 
            phis_1 = (n-r) * X_i_dot_alpha + (n-r) * torch.dot(X_A_sum, alpha) + (r-A-1) * X_N_sum_dot_alpha
            factor_1 = 0 if r - A - 1<=0 else (r - A - 1)/(n - A - 1)
            factor_2 = 0 if r - A - 2<=0 else ((r - A - 1)*(r - A - 2))/((n - A - 1)*(n - A - 2))
            term_1 = (1 - 3 * factor_1 + 2 * factor_2 ) * X_row_norm_2 
            _term_1 = (1 - 2 * factor_1 + factor_2)* (2 * torch.mv(X, X_A_sum) + torch.norm(X_A_sum, p=2) ** 2)
            term_2 = 2 * (factor_1 - factor_2) * (X_i_dot_X_N_sum + torch.dot(X_A_sum, X_N_sum))
            term_3 = -1 * (factor_1 - factor_2) *  X_A_norm_2
            term_4 = (factor_1 - factor_2) * X_N_norm_2
            term_5 = factor_2 * torch.norm(X_N_sum, p=2) ** 2
            phis_2 = (term_1 + _term_1+  term_2 + term_3 + term_4 + term_5)
            shapley_values = 2/((n-A)*r*(n-A-1)) * phis_1 - 1/((n-A)*r*r) * phis_2 
            # print( 2*X_N_sum_dot_alpha/(n*n) -1/(n*n*n)*torch.norm(X_N_sum, p=2) ** 2)
            return shapley_values[origin_NA_list]
        
    def shap_value_selection(self, trn, alpha, bud, origin_NA_list): 
        with torch.no_grad():
            A = []  # Initial empty set A
            NA_list = list(set(range(len(origin_NA_list))))
            X_i_sum = torch.zeros_like(alpha).to(self.device)
            X_i_norm_2 = 0
            X_row_norm_2 = self.X_row_norm_2[origin_NA_list]
            X_i_dot_alpha = torch.mv(trn, alpha)
            X_N_sum_dot_alpha = torch.dot(self.X_sum, alpha)
            X_i_dot_X_N_sum = torch.mv(trn, self.X_sum)

            # Loop over until A reaches desired size
            while len(A) < bud:
                phis = self.r_restirctive_shap_value_evaluation(
                    X = trn, 
                    n = trn.shape[0],
                    A = len(A),
                    r = bud,
                    alpha = alpha,
                    X_row_norm_2 = X_row_norm_2,
                    X_N_sum = self.X_sum,
                    X_A_sum = X_i_sum,
                    X_N_norm_2 = self.X_norm_2,
                    X_A_norm_2 = X_i_norm_2,
                    X_i_dot_alpha = X_i_dot_alpha,
                    X_N_sum_dot_alpha = X_N_sum_dot_alpha,
                    X_i_dot_X_N_sum = X_i_dot_X_N_sum,
                    origin_NA_list = NA_list)
                NA_max_phi_idx = phis.argmax().item()
                # max_phi_idx = NA_list[NA_max_phi_idx]
                # prob_dist = F.softmax(phis, dim=0)
                # phis = (phis-phis.min())/(phis.max()-phis.min())
                # prob_dist = phis / torch.sum(phis)
                # NA_max_phi_idx = np.random.choice(len(NA_list), size = 1, p = prob_dist.cpu().numpy())[0]
                max_phi_idx = NA_list[NA_max_phi_idx]

                A.append(origin_NA_list[max_phi_idx])
                choosen_X = trn[max_phi_idx].to(self.device)
                NA_list.pop(NA_max_phi_idx)

                X_i_sum += choosen_X
                X_i_norm_2 += torch.norm(choosen_X, p=2)**2

            gamma_list = torch.ones(len(A)).tolist()
            origin_NA_list = (torch.tensor(origin_NA_list)[NA_list]).tolist()
            # print(alpha.norm().cpu(), (X_i_sum/bud).norm().cpu(),(alpha - X_i_sum/bud).norm().cpu(), (alpha - X_i_sum/bud).norm().cpu()/alpha.norm().cpu())
            print((alpha - X_i_sum/bud).norm().cpu()/alpha.norm().cpu(), ((torch.dot(alpha,X_i_sum/bud))/(alpha.norm()*(X_i_sum/bud).norm())).cpu())

            return A, gamma_list, origin_NA_list, X_i_sum, X_i_norm_2

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
        trn_gradients_full = self.grads_per_elem.float()
        idx = torch.randperm(trn_gradients_full.shape[0])
        trn_gradients_full = trn_gradients_full[idx]

        batch_sum = int(self.trainloader.batch_size/(budget/trn_gradients_full.shape[0]))
        for index in range(math.ceil(trn_gradients_full.shape[0]/batch_sum-1)):
            trn_gradients = trn_gradients_full[index*batch_sum : (index+1)*batch_sum,:]
            if self.valid:
                mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(self.device)
            else:
                mean_grad = torch.mean(trn_gradients, dim=0).to(self.device)
            NA_list = list(set(range(trn_gradients.shape[0])))
            # self.sum_1 = torch.sum(1 / torch.arange(1, trn_gradients.shape[0] + 1+1).float())
            # self.sum_2 = torch.sum(1 / torch.arange(1, trn_gradients.shape[0] + 1+1).float() ** 2)
            self.X_sum = torch.sum(trn_gradients, dim=0).to(self.device)
            self.X_row_norm_2 = (torch.norm(trn_gradients, dim=1, p=2) ** 2).to(self.device)
            self.X_norm_2 = torch.sum(self.X_row_norm_2).to(self.device)
            self.culmulate_grad = torch.zeros_like(mean_grad).to(self.device)
            _idxs1, _gammas1, NA_list, X_i_sum, X_i_norm_2 = self.shap_value_selection(trn_gradients[NA_list],
                                                    mean_grad,
                                                    self.trainloader.batch_size,
                                                    NA_list)
            idxs.extend(_idxs1)
            gammas.extend(_gammas1)
            self.X_sum -= X_i_sum
            self.X_norm_2 -= X_i_norm_2
            self.culmulate_grad += X_i_sum
            # print((mean_grad - self.culmulate_grad/len(_idxs1)).norm().cpu()/mean_grad.norm().cpu(), 
            #     ((torch.dot(mean_grad,self.culmulate_grad/len(_idxs1)))/(mean_grad.norm()*(self.culmulate_grad/len(_idxs1)).norm())).cpu())

        # _idxs1, _gammas1, NA_list, X_i_sum, X_i_norm_2 = self.shap_value_selection(trn_gradients[NA_list],
        #                                         mean_grad,
        #                                         budget,
        #                                         NA_list)
        # idxs.extend(_idxs1)
        # gammas.extend(_gammas1)
        # self.X_sum -= X_i_sum
        # self.X_norm_2 -= X_i_norm_2


        del trn_gradients, self.X_sum, self.X_row_norm_2
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

        # if self.selection_type in ["PerClass", "PerClassPerGradient", "shap"]:
        #     rand_indices = np.random.permutation(len(idxs))
        #     idxs = list(np.array(idxs)[rand_indices])
        #     gammas = list(np.array(gammas)[rand_indices])
        idxs = [int(x) for x in idxs]
        omp_end_time = time.time()
        self.logger.debug("SHAPIS algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)
    
    

    