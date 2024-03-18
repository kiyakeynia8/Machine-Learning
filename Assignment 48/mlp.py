import numpy as np
from tqdm import tqdm

class MLP:
    def __init__(self, D_in, H1, H2, D_out, epochs, learning_rate, A_F1, A_F2, A_F_out):
        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.D_out = D_out
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.A_F1 = A_F1
        self.A_F2 = A_F2
        self.A_F_out = A_F_out
        self.W1, self.W2, self.W3 = self.W_rand()
        self.B1, self.B2, self.B3 = self.B_rand()

    def W_rand(self):
        W1 = np.random.randn(self.D_in, self.H1)
        W2 = np.random.randn(self.H1, self.H2)
        W3 = np.random.randn(self.H2, self.D_out)
        return W1, W2, W3
    
    def B_rand(self):
        B1 = np.random.randn(1, self.H1)
        B2 = np.random.randn(1, self.H2)
        B3 = np.random.randn(1, self.D_out)

        return B1, B2, B3
    
    def activation(self, A_F, X):
        if A_F == "sigmoid":
            return 1 / (1 + np.exp(-X))
        elif A_F == "softmax":
            return np.exp(X) / np.sum(np.exp(X))
        
    def forward(self, x):
        # latyer 1
        net1 = x.T @ self.W1 + self.B1
        out1 = self.activation(self.A_F1, net1)

        # latyer 2
        net2 = out1 @ self.W2 + self.B2
        out2 = self.activation(self.A_F2, net2)

        # later 3
        net3 = out2 @ self.W3 + self.B3
        out3 = self.activation(self.A_F_out, net3)

        return out1, out2, out3
    
    def backward(self, out3, out2, out1, x_train, y_train):
        y_pred = out3
        # layer 3
        error = -2 * (y_train - y_pred) # مشتق
        grad_B3 = error # مشتی بایاس = 1
        grad_W3 = out2.T @ error

        # layer 2
        error = error @ self.W3.T * out2 * (1 - out2)
        grad_B2 = error
        grad_W2 = out1.T @ error

        # layer 1
        error = error @ self.W2.T * out1 * (1 - out1)
        grad_B1 = error
        grad_W1 = x_train @ error

        return grad_B1, grad_B2, grad_B3, grad_W1, grad_W2, grad_W3
    
    def updata(self, grad_B1, grad_B2, grad_B3, grad_W1, grad_W2, grad_W3):
        # layer 1
        self.W1 -= self.learning_rate * grad_W1
        self.B1 -= self.learning_rate * grad_B1

        # layer 2
        self.W2 -= self.learning_rate * grad_W2
        self.B2 -= self.learning_rate * grad_B2

        # layer 3
        self.W3 -= self.learning_rate * grad_W3
        self.B3 -= self.learning_rate * grad_B3

    def fit(self, X_train, Y_train):
        LOSS = []
        ACC = []
        for epoch in tqdm(range(self.epochs)):
            Y_pred = []
            # train
            for x, y in zip(X_train, Y_train):
                x = x.reshape(-1, 1)

                ## forward
                out1, out2, out3 = self.forward(x)

                y_pred = out3
                Y_pred.append(y_pred)

                ## backward
                grad_B1, grad_B2, grad_B3, grad_W1, grad_W2, grad_W3 = self.backward(y_pred, out2, out1, x, y)

                ## updata
                self.updata(grad_B1, grad_B2, grad_B3, grad_W1, grad_W2, grad_W3)
            
            Loss, Acc = self.calc_Loss_Acc(Y_pred, Y_train)
            LOSS.append(Loss)
            ACC.append(Acc)

        # return f"Loss: {LOSS}, Acc: {ACC}"
        return LOSS, ACC, Loss, Acc
    
    def calc_Loss_Acc(self, Y_pred, Y_test):
        Y_pred = np.array(Y_pred).reshape(-1, self.D_out)
        Loss = np.mean((Y_pred - Y_test) ** 2)
        Acc = np.sum(np.argmax(Y_test, axis=1) == np.argmax(Y_pred, axis=1)) / len(Y_pred)
        
        return Loss, Acc