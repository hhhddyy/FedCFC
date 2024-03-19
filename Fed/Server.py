import cvxpy as cp
import numpy as np



def cosine_similarity(x, y):
    # print(x,y,x @ y)
    # print(cp.sum(x * y),np.dot(x,y) )
    # exit()
    return np.dot(x,y)/ (np.linalg.norm(x) * np.linalg.norm(y))
    # return x @ y / (cp.norm(x) * cp.norm(y))


class Server:
    def __init__(self, client_gradients):
        self.client_gradients = client_gradients  # List of gradients from clients
        self.C = len(client_gradients)
        self.phi = cp.Variable(self.client_gradients[0].shape[0])


    def update_vector(self):
        C = self.C
        grad_matrix = np.stack(self.client_gradients)



        # Compute the cosine similarity matrix for the gradients
        A_upper_left = np.array([[cosine_similarity(grad_matrix[i], grad_matrix[j]) for j in range(C)] for i in range(C)])
        A_upper_right = cp.hstack([self.phi @ grad_matrix[i] for i in range(C)])
        A_upper_right=A_upper_right.reshape((-1,1))
        A_lower_left = A_upper_right.T

        # Construct the full matrix A
        A_lower_right = np.array([[1]])
        A_top = cp.hstack([A_upper_left, A_upper_right])
        A_bottom = cp.hstack([A_lower_left, A_lower_right])
        A = cp.vstack([A_top, A_bottom])

        B = np.full((C + 1, C + 1), 1 / (C ** 2))
        B[C,C] = 1
        for i in range(C):
            B[i, C] = -2 / C
            B[C,i] = -2/ C

        # Define the optimization problem to maximize tr(AB)
        objective = cp.Maximize(cp.trace(A @ B))
        constraints = [cp.sum_squares(self.phi) <= self.client_gradients[0].shape[0]**2]


        prob = cp.Problem(objective,constraints)
        prob.solve(solver=cp.SCS)

        # Update the global vector (phi) based on the solution
        #self.phi.value = self.phi.value / np.linalg.norm(self.phi.value)  # Normalize phi
        print(self.phi.value,prob.value)
        return self.phi


# # Example usage
# client_gradients = [np.random.rand(10) for _ in range(5)]  # Example client gradients
# for i in range(5):
#     client_gradients[i] /= np.linalg.norm(client_gradients[i])
#
#
# server = Server(client_gradients)
# global_vector = server.update_vector()
# print(global_vector.value)



