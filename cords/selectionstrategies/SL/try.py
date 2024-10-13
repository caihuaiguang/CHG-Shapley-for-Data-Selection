import random 

# Function to calculate the value for a given number based on the formula
def calculate_value(x, M, N):
    n = len(N)
    A = len(M)
    if n == 1:
        term1 = 1 / (A + n) * x
        term3 = 0
    else:
        term1 = (1 / (n - 1)) * sum(1 / (A + k) for k in range(1, n))
        term3 = ((1 / n) * sum(1 / (A + k) for k in range(1, n + 1)) - (1 / (A + n))) * (1 / (n - 1)) * sum(x_i for x_i in N)
    if A == 0:
        term2 = 0
    else:
        term2 = (1 / (A * (A + n))) * sum(x_i for x_i in M)
    return term1 * x - term2 - term3, term1, term2, term3

# Function to find the maximum value in N, move it to M, and return the moved value
def move_max_to_M(M, N):
    max_value = max(N, key=lambda x: calculate_value(x, M, N)[0])
    shap = calculate_value(max_value, M, N)[0]
    N.remove(max_value)
    M.append(max_value)
    return max_value, shap

# Generate 20 random positive floating-point numbers
S = [random.uniform(1, 1000000000) for _ in range(100)]
# S = [2 for _ in range(100)]
M = []
N = S.copy()

# print("Set S:", S)
# Iterate and move numbers from N to M based on the described algorithm
cnt = 0
sum_shap = 0
for _ in range(len(S)):
    max_value, values = move_max_to_M(M, N)
    # print(f"Moved {max_value} from N to M with calculated value: {values}")
    if values>-1e-5:
        cnt+=1
        sum_shap+=max_value

# Print the final sets M and N
# print("Final set M:", M)
# print("Final set N:", N)
print("selection rate:", cnt/len(S))
print("selection data average:", sum_shap/cnt)
print("selection data average:", sum(S)/len(S))
print("selection data average:", (sum(S)/len(S))/(sum_shap/cnt))
print("selection data average:", (sum_shap/cnt)/max(S))
print("selection data average:", (sum_shap)/sum(S))
