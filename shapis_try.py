# import torch
#
#
# def draw_gradients(tensor1, tensor2):
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     n, d = tensor1.shape
#     tensor1_mean = torch.mean(tensor1, dim=0)
#     # 合并两个 tensor
#     all_tensors = torch.cat([tensor1, tensor2.unsqueeze(0), tensor1_mean.unsqueeze(0)], dim=0)
#
#     # 使用 PCA 进行降维
#     pca = PCA(n_components=2)
#     result = pca.fit_transform(all_tensors.cpu().numpy())
#
#     # 将降维后的结果拆分成两组
#     result_tensor1 = torch.tensor(result[:n])
#     result_tensor2 = torch.tensor(result[n:n + 1])
#     result_tensor3 = torch.tensor(result[n + 1:])
#
#     # 绘制图像
#     plt.scatter(result_tensor1[:, 0], result_tensor1[:, 1], color='gray', label='Gray')
#     plt.scatter(result_tensor2[:, 0], result_tensor2[:, 1], color='black', label='Validation', s=40)
#     plt.scatter(result_tensor3[:, 0], result_tensor3[:, 1], color='Blue', label='Training_mean', s=40)
#     print('Validation:', result_tensor2)
#     print('Training_mean:', result_tensor3)
#     print('Validation:', tensor2)
#     print('Training_mean:', tensor1_mean)
#     # 添加标签和图例
#     plt.title('Visualization of Two Sets of Vectors (Reduced to 2D)')
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.legend()
#
#     # 保存图像
#     # plt.savefig('vector_visualization_2d.png')
#     # plt.close()
#
#     # # 显示图像（可选）
#     plt.show()
#     plt.close()
#
# def phi_j(A_len, n, x_j, alpha, x_i_sum, x_j_sum, x_j_2, x_j_2_sum):
#     n_minus_A_len = n - A_len
#     k_range = torch.arange(2, n_minus_A_len + 1, dtype=torch.float32)
#     term1 = 0 if n_minus_A_len <= 0 else (torch.sum(1 / (k_range + A_len) ** 2) + 1 / (1 + A_len) ** 2) / n_minus_A_len
#     term2 = 0 if n_minus_A_len <= 1 else torch.sum((k_range - 1) / (k_range + A_len) ** 2) / (
#                 n_minus_A_len * (n_minus_A_len - 1))
#     term3 = 0 if n_minus_A_len <= 1 else torch.sum(
#         (2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
#                                                      n_minus_A_len * (n_minus_A_len - 1))
#     term4 = 0 if n_minus_A_len <= 2 else torch.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / (
#                 (k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
#                                                      n_minus_A_len * (n_minus_A_len - 1) * (n_minus_A_len - 2))
#     term5 = 0 if A_len == 0 else (torch.sum(
#         (2 * k_range + 2 * A_len - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) + (2 * A_len + 1) / (
#                                               (1 + A_len) ** 2 * A_len ** 2)) / n_minus_A_len
#     term6 = torch.sum(1 / (k_range + A_len))
#
#     phi = (-term1 + 2 * term2 - term3 + 2 * term4) * x_j_2
#
#     phi += (-2 * term2 - 2 * term4) * torch.mv(x_j, x_j_sum)
#
#     phi += (term3 - term4) * x_j_2_sum
#
#     phi += term4 * x_j_sum.pow(2).sum(dim=0)
#
#     phi += (-2 * term1 - 2 * term3) * torch.mv(x_j, x_i_sum)
#
#     phi += (2 * term3) * torch.dot(x_j_sum, x_i_sum)
#
#     phi += term5 * x_i_sum.pow(2).sum(dim=0)
#
#     phi += 2 * ((term6 - 1 / n) / (n_minus_A_len - 1)) * torch.mv(x_j, alpha)
#     phi -= 0 if A_len == 0 else 2 / (n * A_len) * torch.dot(x_i_sum, alpha)
#     phi -= 2 * (term6 / n_minus_A_len - 1 / n) * (1 / (n_minus_A_len - 1)) * torch.dot(x_j_sum, alpha)
#
#     return phi
#
#
# def main():
#     # Example usage:
#     N = 100  # Number of elements
#     d = 2  # Dimensionality of each vector
#     X = torch.randn(N, d) + 1 # Input tensor
#     alpha = torch.randn(d)  # Input tensor
#     A = []  # Initial empty set A
#     NA = set(range(N)) - set(A)
#     NA_list = list(NA)
#     x_j_sum = torch.sum(X, dim=0)
#     x_i_sum = torch.zeros_like(x_j_sum)
#     x_j_2_sum = torch.sum(X ** 2)
#     X_2 = X.pow(2).sum(dim=1)
#
#     # Loop over until A reaches desired size
#     while len(A) < 100:
#         x_j = X[NA_list]  # Select the elements not in A
#         x_j_2 = X_2[NA_list]
#         phis = phi_j(len(A), N, x_j, alpha, x_i_sum, x_j_sum, x_j_2, x_j_2_sum)
#         if phis.max() <= 0:
#             break
#         else:
#             max_phi_idx = NA_list[phis.argmax().item()]
#             A.append(max_phi_idx)
#             NA_list.remove(max_phi_idx)
#             x_j_sum -= X[max_phi_idx]
#             x_i_sum += X[max_phi_idx]
#             x_j_2_sum -= X[max_phi_idx].pow(2).sum(dim=0)
#
#     print("Selected elements:", A)
#     print("len(A)",len(A))
#     draw_gradients(X[A],alpha)
#     draw_gradients(X,alpha)
#
#
# main()
#
# # class YourClass:
# #     def shap_value_iterative_selection(self, X, alpha, bud):
# #         # n, d = X.shape
# #         # shap_values = torch.zeros(n)
# #         # x_sum = torch.sum(X, dim=0)
# #         # sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
# #         # sum_2 = torch.sum(1 / torch.arange(1, n + 1).float() ** 2)
# #         # term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 / n) * torch.norm(x_sum, p=2)**2
# #         # term_4 = 1 / (n * (n - 1)) * (sum_2 - 1 / n) * torch.norm(X, p=2)**2
# #         # for j in range(n):
# #         #     x_j = X[j, :]
# #         #     term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - sum_2 - 1 / n)
# #         #               + 1 / (n * (n - 1) * (n-2)) * (2 * sum_1 - 2* sum_2 - 1 / n)) * torch.norm(x_j, p=2)**2
# #         #     term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / (n*n)) * torch.dot(x_sum , x_j)
# #         #     term_5 = 2 / (n - 1) * (sum_1 - 1 / n) *  torch.dot(alpha , x_j)
# #         #     term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum , x_j)
# #         #     shap_values[j] = term_1 + term_2 + term_5 + term_6
# #         # shap_values += term_3 + term_4
#
#
# #         if bud == -1:
# #             positive_indices = torch.where(shap_values > 0)[0]
# #             gamma_list = torch.where(shap_values > 0, 1, 0)
# #             return positive_indices.tolist(), gamma_list.tolist()
# #         elif bud > 0:
# #             top_k_indices = torch.argsort(shap_values)[-bud:][::-1]
# #             gamma_list = torch.where(shap_values[top_k_indices] > 0, 1, 0)
# #             return top_k_indices.tolist(), gamma_list.tolist()
#
# # # 创建一个示例对象
# # your_instance = YourClass()
# # # 提供输入矩阵 X，alpha 和 bud
# # X = torch.rand(10, 5)  # 替换为你的实际数据
# # alpha = torch.rand(5)  # 替换为你的实际值
# # print(X)
# # print(alpha)
# # bud = -1  # 替换为你的实际值
#
# # # 调用函数计算 Shapley 值
# # result_indices, result_gamma = your_instance.shap_value_iterative_selection(X, alpha, bud)
# # print("Top k indices:", result_indices)
# # print("Gamma list:", result_gamma)

import torch
import random
import numpy as np
def shap_value_iterative_selection(X, alpha):
    n, d = X.shape
    x_sum = torch.sum(X, dim=0)
    sum_1 = torch.sum(1 / torch.arange(1, n + 1).float())
    sum_2 = torch.sum(1 / torch.arange(1, n + 1).float() ** 2)
    term_1 = (-1 / n * sum_2 + 1 / (n * (n - 1)) * (2 * sum_1 - 3 * sum_2 + 1 / n)
              + 2 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, dim=1, p=2) ** 2
    term_2 = -2 / ((n - 1) * (n - 2)) * (sum_1 - sum_2 - 1 / n + 1 / (n * n)) * torch.mv(X, x_sum)
    term_3 = 1 / (n * (n - 1) * (n - 2)) * (2 * sum_1 - 2 * sum_2 - 1 + 1 / n) * torch.norm(x_sum, p=2) ** 2
    term_4 = (1 / (n * (n - 1)) * (sum_2 - 1 / n) - 1 / (n * (n - 1) * (n - 2)) * (
                2 * sum_1 - 2 * sum_2 - 1 + 1 / n)) * torch.norm(X, p=2) ** 2
    term_5 = 2 / (n - 1) * (sum_1 - 1 / n) * torch.mv(X, alpha)
    term_6 = -2 / (n * (n - 1)) * (sum_1 - 1) * torch.dot(x_sum, alpha)

    shapley_values = term_1 + term_2 + term_3 + term_4 + term_5 + term_6

    return shapley_values

def phi_j(A_len, n, item1, item2, item3, item4, item5, item6, item7, item8, item9, item10):
    n_minus_A_len = n - A_len
    if n_minus_A_len == 1:
        return -1 / ((1 + A_len) ** 2) * item1 - 2 / ((1 + A_len) ** 2) * item5 + (2 * A_len + 1) / ((1 + A_len) ** 2 * A_len ** 2) * item7 + 2/(A_len+1) * item8 -2 / ( (A_len+1) * A_len) * item9
    k_range = torch.arange(2, n_minus_A_len + 1, dtype=torch.float32)
    term1 = (torch.sum(1 / (k_range + A_len) ** 2) + 1 / (1 + A_len) ** 2) / n_minus_A_len
    term2 = torch.sum((k_range - 1) / (k_range + A_len) ** 2) / (
                n_minus_A_len * (n_minus_A_len - 1))
    term3 =  torch.sum(
        (2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
                                                     n_minus_A_len * (n_minus_A_len - 1))
    term4 = 0 if n_minus_A_len <= 2 else torch.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / (
                (k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) / (
                                                     n_minus_A_len * (n_minus_A_len - 1) * (n_minus_A_len - 2))
    term5 = 0 if A_len == 0 else (torch.sum(
        (2 * k_range + 2 * A_len - 1) / ((k_range + A_len) ** 2 * (k_range + A_len - 1) ** 2)) + (2 * A_len + 1) / (
                                              (1 + A_len) ** 2 * A_len ** 2)) / n_minus_A_len
    term6 = torch.sum(1 / (k_range + A_len))+1/(A_len+1)
    # print(A_len,": ",term1,term2,term3,term4,term5,term6)

    phi = (-term1 + 2 * term2 - term3 + 2 * term4) * item1 + (-2 * term2 - 2 * term4) * item2 +  (-2 * term1 - 2 * term3) * item5 + 2 * ((term6 - 1 / n) / (n_minus_A_len - 1)) * item8

    phi += ((term3 - term4) * item3 + term4 * item4 + (2 * term3) * item6 + term5 * item7 -  2 * (term6 / n_minus_A_len - 1 / n) * (1 / (n_minus_A_len - 1)) * item10)

    if A_len > 0:
        phi -= 2 / (n * A_len) * item9

    return phi


def monte_caluo(A_len, n, x_j, alpha, x_i_sum):
    t = 0
    shap = torch.zeros(n - A_len)
    T = 10000
    while t < T:
        t += 1
        # 使用torch.randperm()生成随机排列的索引
        indices = torch.randperm(n - A_len)
        v_0 = 0
        x_j_pre_sum = torch.zeros_like(x_j[0])
        l = torch.tensor(0)
        for i in indices:
            item = x_j[i]
            l += 1
            x_j_pre_sum += item
            sum = alpha.pow(2).sum(dim=0) if A_len == 0 else (x_i_sum/ A_len - alpha).pow(2).sum(dim=0)
            v = sum - ((x_j_pre_sum + x_i_sum) / (l + A_len) - alpha).pow(2).sum(dim=0)
            shap[i] = shap[i] + (v - v_0)
            v_0 = v
    return shap/torch.tensor(T)


def main():
    # Example usage:
    N = 10  # Number of elements
    d = 2  # Dimensionality of each vector
    X = torch.randn(N, d)  *10# Input tensor
    alpha = torch.randn(d) *10 # Input tensor
    A = []  # Initial empty set A
    NA = set(range(N)) - set(A)
    NA_list = list(NA)

    x_j_sum = torch.sum(X, dim=0)
    x_i_sum = torch.zeros_like(x_j_sum)

    v1 = X.pow(2).sum(dim=1)
    v2 = torch.mv(X,x_j_sum)
    v3 = torch.sum(X ** 2)
    v4 = x_j_sum.pow(2).sum(dim=0)
    v5 = torch.mv(X, x_i_sum)
    v6 = torch.dot(x_j_sum, x_i_sum)
    v7 = x_i_sum.pow(2).sum(dim=0)
    v8 = torch.mv(X, alpha)
    v9 = torch.dot(x_i_sum, alpha)
    v10 = torch.dot(x_j_sum, alpha)
    # Loop over until A reaches desired size
    while len(A) < N and len(A)>=0:
        x_j = X[NA_list]  # Select the elements not in A
        if (len(NA_list)==1):
            k=1
        phis = phi_j(len(A), N, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)[NA_list]
        # print( v1[NA_list],"\n", v2[NA_list],"\n",v3,"\n",v4,"\n", v5[NA_list],"\n", v6,"\n", v7,"\n", v8[NA_list],"\n", v9,"\n", v10)
        # print("--------------j--------------------")
        # phis = phi_j(len(A), N, v1, v2[NA_list], v3, v4, v5[NA_list], v6, v7, v8[NA_list], v9, v10)
        # phis3 = shap_value_iterative_selection(x_j, alpha)
        phis2 = monte_caluo(len(A), N, x_j, alpha, x_i_sum)
        print("differ:",(phis - phis2).pow(2).sum(dim=0))
        print(phis.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))
        print(phis2.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))
        # print("phis:",phis,phis.sum())
        # print("montecalo:",phis2,phis2.sum())
        # print("differ:",phis-phis2)
        # print((phis2 - phis3).pow(2).sum(dim=0))
        # print((phis - phis3).pow(2).sum(dim=0))
        # if phis.max() <= 0:
        #     break
        # else:
        max_phi_idx = NA_list[phis.argmax().item()]
        A.append(max_phi_idx)
        NA_list.remove(max_phi_idx)
        choosen_X = X[max_phi_idx]
        x_j_sum -= choosen_X
        x_i_sum += choosen_X
        v2_v5_item = torch.mv(X, choosen_X)
        v2 -= v2_v5_item
        v5 += v2_v5_item
        v3 -= choosen_X.pow(2).sum(dim=0)
        v4 = x_j_sum.pow(2).sum(dim=0)
        v6 = torch.dot(x_j_sum, x_i_sum)
        v7 = x_i_sum.pow(2).sum(dim=0)
        v9_v10_item = torch.dot(choosen_X, alpha)
        v9 += v9_v10_item
        v10 -= v9_v10_item

    print("Selected elements:", A)

    print("----------------------------------------")
    A = []  # Initial empty set A
    NA = set(range(N)) - set(A)
    NA_list = list(NA)

    x_j_sum = torch.sum(X, dim=0)
    x_i_sum = torch.zeros_like(x_j_sum)
    x_j_2_sum = torch.sum(X ** 2)
    X_2 = X.pow(2).sum(dim=1)

    # Loop over until A reaches desired size
    while len(A) < N:
        x_j = X[NA_list]  # Select the elements not in A
        x_j_2 = X_2[NA_list]
        phis = phi_j2(len(A), N, x_j, alpha, x_i_sum, x_j_sum, x_j_2, x_j_2_sum)
        phis2 = monte_caluo(len(A), N, x_j, alpha, x_i_sum)
        print("differ:",(phis - phis2).pow(2).sum(dim=0))
        print(phis.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))
        print(phis2.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))

        max_phi_idx = phis.argmax().item()
        max_origin_idx = NA_list[max_phi_idx]
        A.append(max_origin_idx)
        # gamma.append(phis[max_phi_idx].cpu())
        NA_list.remove(max_origin_idx)
        x_j_sum -= X[max_origin_idx]
        x_i_sum += X[max_origin_idx]
        x_j_2_sum -= X[max_origin_idx].pow(2).sum(dim=0)


def phi_j2(A_len, n, x_j, alpha, x_i_sum, x_j_sum, x_j_2, x_j_2_sum):
    with torch.no_grad():
        n_minus_A_len = n - A_len
        k_range = np.arange(2, n_minus_A_len + 1, dtype=np.float32)
        term1 = 0 if n_minus_A_len <= 0 else (np.sum(1 / (k_range + A_len)**2)+1/(1 + A_len)**2) / n_minus_A_len
        term2 = 0 if n_minus_A_len <= 1 else np.sum((k_range - 1) / (k_range + A_len)**2) / (n_minus_A_len * (n_minus_A_len - 1))
        term3 = 0 if n_minus_A_len <= 1 else np.sum((2 * k_range + 2 * A_len - 1) * (k_range - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2)) / (n_minus_A_len * (n_minus_A_len - 1))
        term4 = 0 if n_minus_A_len <= 2 else np.sum((2 * k_range + 2 * A_len - 1) * (k_range - 2) * (k_range - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2))/ (n_minus_A_len * (n_minus_A_len - 1) * (n_minus_A_len - 2))
        term5 = 0 if A_len == 0 else (np.sum((2 * k_range + 2 * A_len - 1) / ((k_range + A_len)**2 * (k_range + A_len - 1)**2))+(2 * A_len + 1) / ((1 + A_len)**2 * A_len**2))/ n_minus_A_len
        term6 = np.sum(1 / (k_range + A_len))+1/(A_len+1)
        term1 = torch.tensor(term1)
        term2 = torch.tensor(term2)
        term3 = torch.tensor(term3)
        term4 = torch.tensor(term4)
        term5 = torch.tensor(term5)
        term6 = torch.tensor(term6)
        # print(A_len,": ",term1,term2,term3,term4,term5,term6)


        phi =  (-term1 + 2 * term2  - term3  + 2 * term4 ) * x_j_2\
            - 2 * (term2 + term4) * torch.mv(x_j, x_j_sum)\
            + (term3 - term4) * x_j_2_sum \
            + term4  * x_j_sum.pow(2).sum(dim=0)\
            - 2* (term1 + term3) * torch.mv(x_j, x_i_sum)\
            + 2 * term3 * torch.dot(x_j_sum, x_i_sum)\
            + term5 * x_i_sum.pow(2).sum(dim=0)\
            + 2 * ((term6 - 1/n) / (n_minus_A_len - 1)) * torch.mv(x_j, alpha) - 2 * (term6 / n_minus_A_len - 1 / n) * (1 / (n_minus_A_len - 1)) * torch.dot(x_j_sum, alpha)
        phi -= 0 if A_len == 0 else 2 / (n * A_len) * torch.dot(x_i_sum, alpha)
        # print( x_j_2,"\n",torch.mv(x_j, x_j_sum),"\n",x_j_2_sum,"\n", x_j_sum.pow(2).sum(dim=0),"\n", torch.mv(x_j, x_i_sum),"\n", torch.dot(x_j_sum, x_i_sum),"\n", x_i_sum.pow(2).sum(dim=0),"\n", torch.mv(x_j, alpha),"\n", torch.dot(x_i_sum, alpha), "\n",torch.dot(x_j_sum, alpha))
        #
        # print("--------------j2--------------------")
        return phi


def main2():
    # Example usage:
    N = 10  # Number of elements
    d = 2  # Dimensionality of each vector
    X = torch.randn(N, d)  *10# Input tensor
    alpha = torch.randn(d) *10 # Input tensor
    A = []  # Initial empty set A
    NA = set(range(N)) - set(A)
    NA_list = list(NA)

    x_j_sum = torch.sum(X, dim=0)
    x_i_sum = torch.zeros_like(x_j_sum)
    x_j_2_sum = torch.sum(X ** 2)
    X_2 = X.pow(2).sum(dim=1)

    # Loop over until A reaches desired size
    while len(A) < N-1:
        x_j = X[NA_list]  # Select the elements not in A
        x_j_2 = X_2[NA_list]
        phis = phi_j2(len(A), N, x_j, alpha, x_i_sum, x_j_sum, x_j_2, x_j_2_sum)
        phis2 = monte_caluo(len(A), N, x_j, alpha, x_i_sum)
        print("differ:",(phis - phis2).pow(2).sum(dim=0))
        print(phis.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))
        print(phis2.sum(),"==?", alpha.pow(2).sum(dim=0) - (alpha- X.mean(dim=0)).pow(2).sum(dim=0) if len(A) == 0 else (alpha-x_i_sum/len(A)).pow(2).sum(dim=0)-(alpha - X.mean(dim=0)).pow(2).sum(dim=0))

        max_phi_idx = phis.argmax().item()
        max_origin_idx = NA_list[max_phi_idx]
        A.append(max_origin_idx)
        # gamma.append(phis[max_phi_idx].cpu())
        NA_list.remove(max_origin_idx)
        x_j_sum -= X[max_origin_idx]
        x_i_sum += X[max_origin_idx]
        x_j_2_sum -= X[max_origin_idx].pow(2).sum(dim=0)

def suppleyment_greedy(X, alpha, bud, x_i_sum=None, A_len=0):
        if x_i_sum is None:
            x_i_sum_minus_alpha = - (A_len+1) * alpha 
        else:
            x_i_sum_minus_alpha = x_i_sum - (A_len+1) * alpha 
        _N, d = X.shape
        _A = []  # Initial empty set _A
        NA = set(range(_N)) - set(_A)
        NA_list = list(NA)
        v1 = X.pow(2).sum(dim=1)
        while len(_A) < bud:
            print(NA_list)
            gains = v1+2*torch.mv(X, x_i_sum_minus_alpha)
            gains = gains[NA_list]
            print(gains)
            gains_max = gains.max()
            if gains_max<=0:
                break
            else:
                max_gain_idx = NA_list[gains.argmax().item()] 
                _A.append(max_gain_idx)
                NA_list.remove(max_gain_idx)
                x_i_sum_minus_alpha += X[max_gain_idx] - alpha
        gamma_list = torch.ones(len(_A)).tolist()
        print("---len(_A)---:", len(_A), "max_gain", gains_max)
        return _A, gamma_list
def main3():
    # Example usage:
    N = 10  # Number of elements
    d = 2  # Dimensionality of each vector
    X = torch.randn(N, d)  *10# Input tensor
    alpha = torch.randn(d) *10 # Input tensor
    print(suppleyment_greedy(X,alpha,N))
# main()
# main3()


def monte_caluo_r_restictive(X, X_i_sum, A_len, r, n, alpha):
    t = 0
    shap = torch.zeros(n - A_len)
    T = 100000
    while t < T:
        t += 1
        # 使用torch.randperm()生成随机排列的索引
        indices = torch.randperm(n - A_len) 
        # print(indices)
        # print(indices[r-A_len-1])
        # print(len(X[indices][0:r-A_len]))
        # print(r-A_len)
        v = torch.norm(alpha, p=2) ** 2 - torch.norm((X_i_sum + torch.sum(X[indices][0:r-A_len],dim=0))/ r - alpha, p=2) ** 2 
        # print(v)
        # for i in indices[0:r-A_len]:
        #     shap[i] = shap[i] + v
        shap[indices[r-A_len-1]] = shap[indices[r-A_len-1]] + v
    # shap = shap/(torch.tensor(T)*(r-A_len))
    shap/=torch.tensor(T)
    print("shap:",shap)

    return shap
def r_restirctive_shap_value_evaluation(X, A, r, n, alpha, X_row_norm_2, X_N_sum, X_A_sum, X_N_norm_2, X_A_norm_2, X_i_dot_alpha, X_N_sum_dot_alpha, X_i_dot_X_N_sum, origin_NA_list):
    with torch.no_grad():
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
        # print( 2*X_N_sum_dot_alpha/(n*(r-A)) -1/(n*n*(r-A))*torch.norm(X_N_sum, p=2) ** 2)
        return shapley_values[origin_NA_list]

def main4():
    # Example usage:
    N = 10  # Number of elements
    d = 1  # Dimensionality of each vector
    bud = 4
    X = torch.randn(N, d)  *10# Input tensor
    alpha = torch.randn(d) *10 # Input tensor
    print("X:", X[:,0])
    print("alpha:", alpha)
    A = []  # Initial empty set A
    NA = set(range(N)) - set(A)
    NA_list = list(NA)

    X_i_sum = torch.zeros_like(alpha)
    X_i_norm_2 = 0
    X_row_norm_2 = (torch.norm(X, dim=1, p=2) ** 2)
    X_i_dot_alpha = torch.mv(X, alpha)
    X_sum = torch.sum(X, dim=0)
    X_N_sum_dot_alpha = torch.dot(X_sum, alpha)
    X_i_dot_X_N_sum = torch.mv(X, X_sum)
    X_norm_2 = torch.sum(X_row_norm_2)
    # Loop over until A reaches desired size
    while len(A) < bud:
        phis = r_restirctive_shap_value_evaluation(
            X=X,
            n=X.shape[0],
            A=len(A),
            r=bud,
            alpha=alpha,
            X_row_norm_2=X_row_norm_2,
            X_N_sum=X_sum,
            X_A_sum=X_i_sum,
            X_N_norm_2=X_norm_2,
            X_A_norm_2=X_i_norm_2,
            X_i_dot_alpha=X_i_dot_alpha,
            X_N_sum_dot_alpha=X_N_sum_dot_alpha,
            X_i_dot_X_N_sum=X_i_dot_X_N_sum,
            origin_NA_list=NA_list)
        print("NA_list", NA_list)
        print("phis:", phis)

        # phis2 = monte_caluo_r_restictive(X[NA_list], X_i_sum, A_len=len(A), r=bud, n=N, alpha=alpha)
        # print("differ:",(phis - phis2).pow(2).sum(dim=0), "phis.sum():", phis.sum(), "phis2.sum():", phis2.sum())

        # phis3 = alpha.pow(2).sum(dim=0) - torch.norm(X - alpha, dim=1,p=2)**2
        # print("phis3:", phis3/N)

        NA_max_phi_idx = phis.argmax().item()
        max_phi_idx = NA_list[NA_max_phi_idx]
        A.append(max_phi_idx)
        choosen_X = X[max_phi_idx]
        NA_list.pop(NA_max_phi_idx)
        print(choosen_X)
        X_i_sum += choosen_X
        X_i_norm_2 += torch.norm(choosen_X, p=2) ** 2

main4()

