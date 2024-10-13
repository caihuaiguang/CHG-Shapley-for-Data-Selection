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
    This version implement the shapley value based topkselection algorithm.
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
                 selection_type, varients, logger, valid=False, v1=True, lam=0, eps=1e-4):
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
        self.varients = varients
        self.gammas = None
        self.idxs = None
        self.lipschitz = 20
        self.r_1 = [torch.sum(1 / torch.arange(1, r + 1).float()) for r in range(self.trainloader.batch_size+1)]
        self.r_2 = [torch.sum(1 / torch.arange(1, r + 1).float()**2) for r in range(self.trainloader.batch_size+1)]

    def graw_subset_class(self, idxs):
        self.get_labels()
        label_counts = {}
        for i in idxs:
            label = int(self.trn_lbls[i])
            if label in label_counts:
                label_counts[label] += 1/len(idxs)
            else:
                label_counts[label] = 1/len(idxs)
                # Plotting the bar chart
        plt.bar(label_counts.keys(), label_counts.values())
        plt.xlabel('Labels')
        plt.ylabel('Counts')
        plt.ylim(0, 1)
        plt.title('Label Counts')
        plt.savefig("Label Counts")
        plt.close()

    def shap_value_evaluation(self, X, alpha): 
        with torch.no_grad():
            n, d = X.shape
            x_sum = torch.sum(X, dim=0)
            
            # self.sum_1 -= 1/(n+1)
            # self.sum_2 -= 1/(n+1)**2
            # print(self.sum_1, torch.sum(1 / torch.arange(1, n + 1).float()))
            # print(self.sum_2, torch.sum(1 / torch.arange(1, n + 1).float()**2))
            sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
            sum_2 = torch.sum(1 / torch.arange(1, n + 1).float()**2)
            term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3*sum_2 + 1 / n)
                                + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
            term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
            term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
            term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1/ (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
            term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
            term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)
            # term_7 = 1 / (n - 1) * (sum_1 - 1 / n) * self.train_loss - 1 / (n * (n - 1)) * (sum_1 - 1) * torch.sum(self.train_loss)
            # term_7 = loss
            shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6)
            # shapley_values -= shapley_values.min()
            # term_7 = term_7.max() - term_7
            # shapley_values *= term_7
            # shapley_values = (term_1 + term_2 + term_3 + term_4 + term_5 + term_6).to(self.device)

            return shapley_values


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

        if self.selection_type in ['PerClass' ,'PerClassandShap']:
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

                if self.varients == "HardnessShapley":
                    feature_per_elem = self.train_loss[:, None]
                elif self.varients in ["GradientShapley", "TracIn"]:
                    feature_per_elem = self.grads_per_elem
                else: # self.varients == "CHGShapley"
                    feature_per_elem = self.train_loss[:, None] * self.grads_per_elem
                
                if self.valid:
                    mean_grad = torch.mean(self.val_grads_per_elem, dim=0)
                else:
                    mean_grad = torch.mean(feature_per_elem, dim=0)

                if self.varients is "TracIn":
                    shapley_values = torch.mv(feature_per_elem, mean_grad)
                else:
                    shapley_values = self.shap_value_evaluation(feature_per_elem, mean_grad)
                r = int(budget * len(trn_subset_idx) / self.N_trn)
                _gammas, _idxs = torch.topk(shapley_values , k = r, largest=True, dim=0)
                _idxs = _idxs.cpu()
                _gammas = torch.ones(len(_idxs)).tolist()
                idxs.extend(list(np.array(trn_subset_idx)[_idxs]))
                gammas.extend(_gammas)
        elif self.selection_type in ['PerClassPerGradient', 'PerClassPerGradientandShap']:
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
                self.grads_per_elem = self.train_loss[:, None]* self.grads_per_elem
                feature_per_elem = self.grads_per_elem
                tmp_gradients = feature_per_elem[:, i].view(-1, 1)
                tmp1_gradients = feature_per_elem[:,
                                 self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                feature_per_elem = torch.cat((tmp_gradients, tmp1_gradients), dim=1).to(self.device)
                if self.valid:
                    mean_grad = torch.mean(self.val_grads_per_elem, dim=0)
                else:
                    mean_grad = torch.mean(feature_per_elem, dim=0)
                shapley_values = self.shap_value_evaluation(feature_per_elem, mean_grad)
                r = int(budget * len(trn_subset_idx) / self.N_trn)
                _gammas, _idxs = torch.topk(shapley_values , k = r, largest=True, dim=0)
                _idxs = _idxs.cpu()
                _gammas = torch.ones(len(_idxs)).tolist()
                idxs.extend(list(np.array(trn_subset_idx)[_idxs]))
                gammas.extend(_gammas)
        elif self.selection_type in ['SHAPIS', "SHAPISandShap"]:
            self.compute_gradients(self.valid, perBatch=False, perClass=False)
            self.grads_per_elem = self.train_loss[:, None]* self.grads_per_elem
            idxs = []
            gammas = []
            feature_per_elem = self.grads_per_elem.float().to(torch.float16)
            if self.valid:
                mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(torch.float16).to(self.device)
            else:
                mean_grad = torch.mean(feature_per_elem, dim=0).to(torch.float16).to(self.device)
            shapley_values = self.shap_value_evaluation(feature_per_elem, mean_grad)
            gammas, idxs = torch.topk(shapley_values , k = budget, largest=True, dim=0)
            gammas = torch.ones(len(idxs)).tolist()
            idxs = idxs.tolist()

        torch.cuda.empty_cache()
        diff = budget - len(idxs)
        self.logger.debug("Random points added: %d ", diff)

        if diff > 0:
            print("diff>0")
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
        if self.selection_type in ["PerClassPerGradientandShap", "PerClassandShap", "SHAPISandShap"]:
            # self.compute_gradients(self.valid, perBatch=False, perClass=False)
            
            #     if self.varients == "HardnessShapley":
            #         feature_per_elem = self.train_loss[:, None]
            #     elif self.varients in ["GradientShapley", "TracIn"]:
            #         feature_per_elem = self.grads_per_elem
            #     else: # self.varients == "CHGShapley"
            #         feature_per_elem = self.train_loss[:, None] * self.grads_per_elem
            # self.grads_per_elem = self.train_loss[:, None]* self.grads_per_elem
            # feature_per_elem = self.grads_per_elem
            # if self.valid:
            #     mean_grad = torch.mean(self.val_grads_per_elem, dim=0).to(self.device)
            # else:
            #     mean_grad = torch.mean(feature_per_elem, dim=0).to(self.device)
            loss_weights = self.shap_value_evaluation(feature_per_elem[idxs], mean_grad)
            # loss_weights *= self.train_loss[idxs]
            loss_weights -= loss_weights.min()
            gammas = (loss_weights/torch.sum(loss_weights)*len(idxs)).tolist()
        idxs = np.array(idxs)
        gammas = np.array(gammas)
        if self.selection_type in ["PerClass", "PerClassPerGradient", "PerClassPerGradientandShap", "PerClassandShap", "SHAPISandShap"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])

        idxs = [int(x) for x in idxs]
        omp_end_time = time.time()
        self.logger.debug("SHAPIS algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)
    
    

    