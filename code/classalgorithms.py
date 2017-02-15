from __future__ import division  # floating point division
import numpy as np
import utilities as utils

import random



class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
            
    def reset(self, parameters):
        self.resetparams(parameters)
        # TODO: set up required variables for learning
        
    # TODO: implement learn and predict functions                  
            
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions                  
           

class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
        
    # TODO: implement learn and predict functions                  

    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)


class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        
    # TODO: implement learn and predict functions                  
           







class RBF_linearRegression(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01, 'beta':0.05,}
        self.reset(parameters)
        self.beta = 1        
    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None



    def transform(self, X, centers):
        datapoint = X
        for i, center in enumerate( centers):
            X[i] = np.exp(np.dot(-self.beta,utils.euclidian(datapoint, center) ))
        
        return X

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
        
        # Ensure ytrain is {-1,1}
        numFeatures = Xtrain.shape[1]
        self.centers = random.sample(Xtrain, numFeatures)
 
                       
        for i, X in enumerate(Xtrain):
            Xtrain[i] = self.transform(X, self.centers)
            
        yt = np.copy(ytrain)
        #yt[yt == 0] = -1
        
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
         
    def predict(self, Xtest):
        for i, X in enumerate(Xtest):
            Xtest[i] = self.transform(X, self.centers)
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest



class RBF():
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.beta = 1        



    def distance(self, X, centers):
        datapoint = X
        for i, center in enumerate( centers):
            X[i] = np.exp(np.dot(-self.beta,utils.euclidian(datapoint, center) ))
        
        return X

    def transform(self, Data):
        """ Learns using the traindata """
        
        
        # Ensure ytrain is {-1,1}
        numFeatures = Data.shape[1]
        self.centers = random.sample(Data, numFeatures)
 
                       
        for i, X in enumerate(Data):
            Data[i] = self.distance(X, self.centers)
            
        
         
        return Data


class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    


    def learn(self, Xtrain, ytrain):
       """ Learns using the traindata """

       # Initial random weights ( Better if initialized using linear regression optimal wieghts)
       #Xless = Xtrain[:,self.params['features']]
       weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)



       # w(t+1) = w(t) + eta * v
       #pone = self.probabilityOfOne(self.weights, Xtrain[i])
       p = utils.sigmoid(np.dot(Xtrain, weights))
       tolerance = 0.1
       #error = utils.crossentropy( Xtrain, ytrain, self.weights)
       error = np.linalg.norm(np.subtract(ytrain, p))
       err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
       while np.abs(error - err) < tolerance:
           P = np.diag(p)
           
           I = np.identity(P.shape[0])
           #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
           #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p))#np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
           weights= weights - (np.dot(Hess_inv, First_Grad))
           p = utils.sigmoid(np.dot(Xtrain, weights))

           # error = utils.crossentropy(Xtrain, ytrain, self.weights)
           err = np.linalg.norm(np.subtract(ytrain,  p))

       self.weights = weights

    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest
        

class RBF_LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.01, 'beta':0.05, 'regularizer':None}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions 

    def probabilityOfOne(self, weights, Xtrain):

        return 1/( 1 + np.exp(np.dot(weights.T, Xtrain))) 


    


    def learn(self, Xtrain, ytrain):
       """ Learns using the traindata """

       # Initial random weights ( Better if initialized using linear regression optimal wieghts)
       #Xless = Xtrain[:,self.params['features']]
       Rbf = RBF()
       Xtrain = Rbf.transform(Xtrain)
       weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)), Xtrain.T),ytrain)



       # w(t+1) = w(t) + eta * v
       #pone = self.probabilityOfOne(self.weights, Xtrain[i])
       p = utils.sigmoid(np.dot(Xtrain, weights))
       tolerance = 0.1
       #error = utils.crossentropy( Xtrain, ytrain, self.weights)
       error = np.linalg.norm(np.subtract(ytrain, p))
       err = np.linalg.norm(np.subtract(ytrain,  p))
       #err = 0
       #soldweights =self.weights
       while np.abs(error - err) < tolerance:
           P = np.diag(p)
           
           I = np.identity(P.shape[0])
           #Hess_inv =-np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.subtract(I,self.P)),Xtrain))
           #Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           Hess_inv=-np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I-P))),Xtrain))
           First_Grad= np.dot(Xtrain.T, np.subtract(ytrain,p))#np.dot(Xtrain.T, np.subtract(ytrain, p))
           #oldweights = self.weights
           weights= weights - (np.dot(Hess_inv, First_Grad))
           p = utils.sigmoid(np.dot(Xtrain, weights))

           # error = utils.crossentropy(Xtrain, ytrain, self.weights)
           err = np.linalg.norm(np.subtract(ytrain,  p))

       self.weights = weights

    def predict(self, Xtest):
        Rbf = RBF()
        Xtest = Rbf.transform(Xtest)
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(ytest)
        return ytest
        





