import numpy as np
import math
import pandas as pd
import time

"""
    This Python script was done for the second practical exam of 
    Artificial Intelligence class, the exam consists of 
    creating functions in order to train the algorithmn
    so it can find the optimal w0 and w1 of the data set

    Author: Santiago Cantu
    email: santiago.cantu@udem.edu
    Institution: Universidad de Monterrey
    First created: March 29, 2020
"""

def store_data(url,data):
  """
    function that reads a csv file from a github url
    and then stores it into x and y data for training
    and testing

    Inputs
    :param url = string type
    :param data = string type

    Output
    :return x: numpy matrix
    :return y: numpy array
    :return mean: numpy array
    :return sd: numpy array
    :return w: numpy array
  """
  if(data == "training"):
    # load data from an url
    training_data = pd.read_csv(url)
    #amount of samples and features
    numberSamples, numberFeatures = training_data.shape
    # remove headers from features and separates data in x and y
    x = pd.DataFrame.to_numpy(training_data.iloc[:,0:numberFeatures-1])
    y = pd.DataFrame.to_numpy(training_data.iloc[:,-1]).reshape(numberSamples,1)
    # prints training data
    print_data(training_data,"training")
    # array for means of every feature
    mean = []
    # array for standard deviation of every feature
    sd = []
    # scale features so when returned, the data is already scalated stores x,mean and sd
    x,mean,sd = scale_features(x,mean,sd,"training")
    # prints scaled training data
    print_scaled_data(x,"training")
    # adds ones and transpose the matrix in order to multiply it by w
    x = np.hstack((np.ones((numberSamples,1)),x)).T
    # initializes an array with 0,0 for every feature
    w = initialize_w(x,numberFeatures)
    return x,y,mean,sd,w
  if(data == "testing"):
    #load data from an url
    testing_data = pd.read_csv(url)
    #amount of samples and features
    numberSamples, numberFeatures = testing_data.shape
    # remove headers from features and stores data x
    x = pd.DataFrame.to_numpy(testing_data.iloc[:,0:numberFeatures])
    # prints testing data
    print_data(testing_data,"testing")
    return x

  
def scale_features(x,mean,sd,data):
  """
    function that scalates the x features from the
    training data and testing data with the mean and
    standard deviation

    Input
    :param x: numpy matrix
    :param mean: numpy array
    :param sd: numpy array
    :param data: string type

    Output
    :return x: numpy matrix with scalated values
    :return mean: numpy array of mean
    :return sd: numpy array of standard deviation

  """
  if(data == "training"):
    # scalates data
    for size in range(x.shape[1]):
      x_data = x[:,size]
      m_data = np.mean(x_data)
      sd_data = np.std(x_data)
      mean.append(m_data)
      sd.append(sd_data)
      x[:,size] = (x_data - m_data)/ sd_data
    return x,mean,sd
  if(data == "testing"):
    #scalates testing data and prints it
    x = (x-mean)/sd
    print_scaled_data(x,"testing")
    return x
  
def initialize_w(x,numberFeatures):
  """
    function that initialized an array with 0,0
    values for each of the features

    Input
    :param x: numpy matrix
    :param numberFeatures: int type

    Output
    :return w: numpy array fill with 0,0 for each feature
  """
  # array for w0 and w1 of every feature
  w=[]
  # appends 0,0 for every feature
  for size in range(numberFeatures):
    w.append([0,0])
  # converts array into numpy array
  w = np.asarray(w)
  return w

def gradient_descent(x,y,w,stopping_criteria,learning_rate):
  """
    function that iterates to get the gradient descendent
    until the l2 norm is bigger than the set stopping
    criteria = 0.001

    Input
    :param x: numpy matrix of data
    :param y: numpy array of data
    :param stopping criteria: float type variable
    :param learning rate: float type variable

    Output
    :return w: returns the w array fill with the optimal w0 and w1 for each feature

    """
  # declare a big l2_norm
  l2_norm = 100000
  # size of features and samples
  numberFeatures, numberSamples = x.shape
  # declares a variable to know the numbers of iterations 
  iterations = 0
  while l2_norm > stopping_criteria:
    # calculates the cost function for w0 and w1 for every feature
    cost_function = calculate_gradient_descent(x,y,w,numberSamples,numberFeatures)
    # reshapes the cost function array in order to multiply by a scalar adding 1 columns
    cost_function = np.reshape(cost_function,(numberFeatures,1))
    # calculates the gradient descent with the w0 and w1 of every feature - the learning rate * the cost function
    w = w-learning_rate*cost_function
    # euclidean norm, in order to stop the algorithmn
    l2_norm = calculate_l2_norm(cost_function)
    # variable counting the iterations
    iterations = iterations+1
  return w

def calculate_gradient_descent(x,y,w,numberSamples,numberFeatures):
  """
    function that calculates the hypothesis function and the
    cost function

    Input
    :param x: numpy matrix of data
    :param y: numpy array of data
    :param numberSamples: int type variable of number of samples in the data
    :param numberFeatures: int type variable of the number of features in the data

    Output
    :return cost_function: returns the cost function
  """

  # transpose of y data so it can be substracted
  y = y.T
  # gets the hypothesis function multiplying transpose of W with X
  hypothesis_function = np.matmul(w.T,x)
  # gets the difference betweenthe hypothesis and y data
  difference = np.subtract(hypothesis_function,y)
  # transpose the difference so it can be multiplied
  difference = difference.T
  # gets the cost function of the x axis of the matrix
  cost_function = np.sum(np.matmul(x,difference)/numberSamples, axis=1)

  return cost_function
  
def predict_last_mile(w,x,mean,sd):
  """
    function that predicts the last mile cost
    with the testing data using the trained w's

    Input
    :param w: numpy array with the optimal w0 and w1 for each feature
    :param x: numpy matrix of testing data scalated
    :param mean: numpy array with the mean of training data
    :param sd: numpy array with the standard deviation of training data

    Output
    :return the predicted value
  """
  # number of samples and features
  numberSamples, numberFeatures = x.shape
  # adds a row of 1's 
  x= np.hstack((np.ones((numberSamples,1)),x)).T
  print_last_mile_cost(np.matmul(w.T,x))
  return np.matmul(w.T,x)

def calculate_l2_norm(cost_function):
  """
    function that calculates the l2 norm with the cost function

    Input
    :param cost_function: float type variable

    Output
    :return the l2_norm calculated
  """
  return np.sqrt(np.sum(np.matmul(cost_function.T,cost_function)))


def print_w(w):
  """
    function to print the optimal w

    input
    :param w: numpy array 

    output
    prints the optimal w for each feature
  """
  c = 0
  print('------------------------------------------------')
  print('W parameter')
  print('------------------------------------------------')
  for i in zip(w):
    print('w%s: %s'%(c,i[0][0]))
    c = c + 1

def print_data(sample,data):
  """
    function to print the training and testing data

    input
    :param sample: numpy matrix with data
    :param data: string type variable

    output
    prints the testing and training data
  """
  if(data == "testing"):
    print('------------------------------------------------')
    print('Testing data')
    print('------------------------------------------------')
    print(sample)
  if(data == "training"):
    print('------------------------------------------------')
    print('Training data')
    print('------------------------------------------------')
    print(sample)


def print_scaled_data(scaled_data,data):
  """
    function to print the training and testing data scalated

    input
    :param sample: numpy matrix with data
    :param data: string type variable

    output
    prints the testing and training data scalated
  """
  if(data == "testing"):
    print('------------------------------------------------')
    print('Testing data scaled')
    print('------------------------------------------------')
    print(scaled_data)
  if(data == "training"):
    print('------------------------------------------------')
    print('Training data scaled')
    print('------------------------------------------------')
    print(scaled_data)
  

def print_last_mile_cost(last_mile_cost):
  """
    function to print the predicted last mile cost

    input
    :param last_mile_cost: float type

    output
    prints the predicted last mile cost
  """
  print('------------------------------------------------')
  print('Last-mile cost (predicted)')
  print('------------------------------------------------')
  print(last_mile_cost[0])

def table():
  """
    function that prints the table with the different computing time
    and iterations that took the program to run with different learning
    rates
  """
  print("Learning rate       0.0005  0.001  0.005  0.01  0.05  0.1  0.5")
  print("Computing time        4.19   2.14   0.57  0.26  0.08  541  1714")
  print("Iterations          108781  54389  10875  5436  1085  0.06 0.12")