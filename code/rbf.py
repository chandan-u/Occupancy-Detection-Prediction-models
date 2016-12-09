

import numpy as np
import random



class RBF(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)
        self.beta = 1        
    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None



    def transform(self, X, centers):
        datapoint = X
        for i, center in enumerate( centers):
            X[i] = np.exp(np.dot(-self.beta,np.linalg.norm(datapoint, center) ))
        
        return X

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
        
        # Ensure ytrain is {-1,1}
        numFeatures = Xtrain.shape[1]
        self.centers = random.sample(Xtrain, numFeatures)
 
                       
        for i, X in enumerate(Xtrain):
            Xtrain[i] = self.transform(X, centers)
            
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
 
