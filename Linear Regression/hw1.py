import numpy as np
import itertools
np.random.seed(42)

def preprocess(X, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column. 

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The scaled labels.
    """
      
    Xmean = X.mean(axis=0)
    XmaxVal = X.max(axis=0)
    XminVal = X.min(axis=0)

    X = (X - Xmean) / (XmaxVal - XminVal)

    Ymean = y.mean()
    YmaxVal = y.max()
    YminVal = y.min()

    y = (y - Ymean) / (YmaxVal - YminVal)

    return X, y

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.  

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # Use J for the cost.
    Z = np.subtract(X.dot(theta), y)
    Z = np.square(Z)
    m = y.shape[0]
    J = np.sum(Z) / (2*m)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by 
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = [] # Use a python list to save cost in every iteration
   
    m = y.shape[0]
    mult = (alpha / m)
    for i in range(num_iters):
        Z =  np.subtract(X.dot(theta), y)
        theta = np.subtract(theta, np.multiply((Z).dot(X),mult))
        J_history.append(compute_cost(X, y, theta))
 

    return theta, J_history

def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """
    
    x_trans = np.transpose(X)
    x_pinv = np.linalg.inv(x_trans.dot(X))
    x_pinv = x_pinv.dot(x_trans)
    
    pinv_theta = x_pinv.dot(y) 
  
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value is smaller than 1e-8. This function is
    very similar to the gradient descent function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    J_history = [] # Use a python list to save cost in every iteration
    m = y.shape[0]
    mult = (alpha / m)
    for i in range(num_iters):
        Z = np.subtract(X.dot(theta), y)
        theta = np.subtract(theta, np.multiply((Z).dot(X),mult))
        J = compute_cost(X, y, theta)
        if  i > 1 :
            if J_history[i - 1] - J < 1e-8 : break 
        J_history.append(J)
 

    return theta, J_history

def find_best_alpha(X, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary 
    with alpha as the key and the final loss as the value.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training 
    for every value of alpha.
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    np.random.seed(42)
    theta = np.random.random(size=2)
    for a in alphas:
        t, J = efficient_gradient_descent(X, y, theta, a, iterations)
        alpha_dict.update({a:J[-1]})
    
    return alpha_dict

def generate_triplets(X):
    """
    generate all possible sets of three features out of all relevant features
    available from the given dataset X. You might want to use the itertools
    python library.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """
    
    triplets = []
    for triple in itertools.combinations(X.columns,3):
        triplets.append(triple)
    return triplets

def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best 
    minimizes the cost function. For better performance, you should use the 
    efficient implementation of gradient descent. You should first preprocess
    the data and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A dataframe that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    best_triplet = triplets[0]
    np.random.seed(42)
    n_theta = np.random.random(size=4)
    for triplet in triplets:
    
        col0 = df[triplet[0]]
        col0 = np.array(col0)
        col0.shape = (len(col0),1)
        col1 = df[triplet[1]] 
        col1 = np.array(col1)
        col1.shape = (len(col1),1)
        col2 = df[triplet[2]]
        col2 = np.array(col2)
        col2.shape = (len(col2),1)
        X = np.hstack((col0, col1))
        X = np.hstack((X, col2))
        y = np.array(df['price'])
        #preprocess
        X, y = preprocess(X, y)
        #bias trick
        X0 = np.ones((X.shape[0],1))
        X = np.hstack((X0,X))
        s , J_hist = efficient_gradient_descent(X, y, n_theta, alpha, num_iter)
        if triplet == triplets[0] :
            J_best = J_hist[-1]
            best_triplet = triplets[0]
        elif J_best > J_hist[-1] :
            J_best = J_hist[-1]
            best_triplet = triplet

    return best_triplet

   

  