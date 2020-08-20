import numpy as np
import math

def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

'''
Computes the sigmoid functions with the given input
@param x The input to the sigmoid function
@return  The output of the sigmoid function given the param as input
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
Makes a prediction given some inputs, weights, and bias
@param X 
        The inputs dim=(features,)
@param w 
        The weights dim=(features,)
@param b 
        The bias
@return The prediction
'''
def predict(X, w, b):
    n = len(X)  # number of dimensions
    X = X.reshape((1, n))
    # print(f"shape of w:{w.shape}, x:{X.shape}, y: {Y.shape}")
    # A = sigmoid(np.dot(X, w) + b)
    z = np.dot(X, w) + b
    yhat = sigmoid(z)

    if(yhat >= 0.5):
        return 1.0
    else:
        return 0.0
    # return yhat

''' 
Calculates the partial derivatives of loss with respect 
to weights and bias. (dw, db)
@param w 
        The weights array. 
        Has the same length as there are features
        dim=(features,)
@param b 
        The bias
@param X 
        The input array                 
        dim=(features,)
@param Y 
        The expected output
@return gradients
         A dictionary containing (dw,db)
         The gradients with respect to weights and bias
@return cost
         The loss function averaged over the number of training examples
'''
def backprop(w, b, X, Y):
    """
    assumes X is a single example i.e., m =1 and has  n dimensions
    """
    n = len(X) # number of dimensions
    X = X.reshape((1, n)) # 1, 8
    Y = Y.reshape((1, 1))
    # w: 8, 1, X: 1, 8, Y: 1, 1
    A = sigmoid(np.dot(X, w) + b)
    # A: 1x1
    m = X.shape[0] # number of examples in the batch
    # m: 1
    loss = -(Y*np.log(A) + (1-Y)*np.log(1-A))
    cost = (1/m) * np.sum(loss)

    dz = A - Y
    db = dz
    dw = np.zeros(w.shape)
    for i in range(len(dw)):
        dw[i] = X[0][i] * dz

    gradients = {"dw": dw,
                 "db": db}
    # print("dw.shape", dw.shape)
    return gradients, cost


'''
Updates the weights and biases
@param w 
        The weights array
        dim=(features,)
@param b
        The bias
@param X
        The input array
        dim=(features,m)
@param Y
        The expected output array
        dim=(m,)
@param iterations
        The number of training iterations
@param alpha
        The learning rate
@return w 
         the updated weights array
@return b 
         the updated bias
'''
def gradient_descent(w, b, X, Y, iterations, alpha):
    features = len(w)
    #print(X.shape)
    for i in range(iterations):
        batch_cost = 0
        cost_history = []
        for k in range(X.shape[0]):
            #print("iteration #: " + str(i))
            gradients, sample_cost = backprop(w, b, X[k], Y[k])

            if not math.isnan(sample_cost) and not math.isinf(sample_cost):
                batch_cost += sample_cost
                cost_history.append(sample_cost)
            else:
                continue

            dw = gradients["dw"]
            db = gradients["db"]

            b -= alpha * db
            # w -= alpha * dw
            for j in range(features):
                w[j] -= alpha * dw[j]

        if(i % 100 == 0):
            print("Cost after iteration #" + str(i) + ": " + str(batch_cost))
        # print("Cost history after iteration #" + str(i) + ": ", cost_history)

    return w, b

if __name__ == '__main__':
    myFile = load_data('pima-indians-diabetes.csv')

    X_train = myFile[:384,:-1]
    Y_train = myFile[:384, -1]

    X_test = myFile[384:,:-1]
    Y_test = myFile[384:, -1]
    features = len(X_train[0])
    w = np.zeros(shape=(features, 1)) * 0.01
    b = 0
    w, b = gradient_descent(w, b, X_train, Y_train, 1000, 0.002)
    test_no = 16
    print("\n" + "weights: " + str(w))
    print("bias: " + str(b))
    print("sample prediction: " + str(predict(X_test[test_no], w, b)), "Actual Result:", Y_test[test_no])
    print("__________________________________________________________")