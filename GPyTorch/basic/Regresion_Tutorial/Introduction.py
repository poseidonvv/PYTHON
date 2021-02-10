''' GPytorch Regression Tutorial
we demonstrate many of the desing  we'll be modeling the function  Y=SIN(2PIX) + EPSILON  
con EPSILON ~  N(0,0.04)  With 100 training examples
'''
# LIBRARIES
import math
import torch
from torch._C import dtype
import gpytorch
from matplotlib import pyplot as plt
import os
import pandas as pd 
import numpy as np
from datetime import datetime

#################################  PARAMS  ####################################
regression_size = 512
start_point = 0
training_iter = 100
learning_rate = 0.05
list_variables = ['Series.Close']#, 'delta.Time'] # 
jump = 1
# 
if(start_point < 1):
    start_point = 1
#################################  DATA  ###################################
def get_data(start_point, regression_size, jump=1):
    pd_read_csv_config =  {
                    'filepath_or_buffer': "F:\DATA\data_gpytorch/ES 100MIL DATOS FINALES.csv", 
                    'dtype': {
                        'Bar Ending Time':str,
                        'Series.Open':float,
                        'Series.High':float,
                        'Series.Low':float,
                        'Series.Close':float,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                        'Series.Trade':float,
                        # 'Relative Strength Index':float,
                        'Series.Volume':float,
                    },
                    'skiprows': ([_ for _ in range(1,start_point)]),
                    'nrows': int(regression_size),
                    'compression': None, #'gzip',
                    'parse_dates':['Bar Ending Time'],#{'UTC':['UTCDate','UTCTime']},
                    'date_parser': (lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')), #%p#(lambda x, y: datetime.strptime(x+y, '%Y%m%d%H%M%S%f')),
                    'usecols': ['Bar Ending Time', 'Series.Close', 'Series.Open', 'Series.High', 'Series.Low'], 
                    'sep':',', 
                }
    data = pd.read_csv(**pd_read_csv_config)
    data = data.reindex(index= data.index[::-1])
    data.reset_index(drop=False, inplace = True)
    data.drop(["index"], axis=1, inplace = True)
    data.reset_index(drop=False, inplace = True)
    data['delta.Time'] = data['Bar Ending Time'].diff().apply(lambda x : x.total_seconds()).fillna(0)
    data = data.iloc[::int(jump), :]
    data.reset_index(drop=False, inplace = True)
    return data

###############################  PARA JODERTE LA VIDA ##########################
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # FIXME
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # FIXME
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # FIXME
    
    def get_kernel_matrix(self, x, detach=False):
        if(detach):
            return self.covar_module(x).detach().numpy()
        else:
            return self.covar_module(x)
    
    def print_model_hyperparameters(self):
        for param_name, param in self.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')
def train_gauss_model(model, likelihood, optimizer, train_x, train_y):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = optimizer
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # FIXME
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()
        # Get into evaluation (predictive posterior) mode
def predict_gauss_model(model, likelihood, test_x):
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 512).type(torch.FloatTensor)
        return likelihood(model(test_x))
def plot_gauss_model_eval(train_x, train_y, test_x, test_y, show=True):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = test_y.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), test_y.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
    if(show):
        plt.show()
    
        
    
#################################  MAKE THE MODEL  ####################################
for variable in list_variables:
    data_1 = get_data(start_point=start_point, regression_size=regression_size, jump=jump)
    train_x_1 = torch.Tensor(data_1["index"].values)
    train_y_1 = torch.Tensor((data_1[variable].values - np.mean(data_1[variable].values))/np.std(data_1[variable].values))
    test_x_1 = train_x_1
    # initialize likelihood and model
    likelihood_1 = gpytorch.likelihoods.GaussianLikelihood() # FIXME
    model_1 = ExactGPModel(train_x_1, train_y_1, likelihood_1)
    optimizer_1 = torch.optim.Adam([
            {'params': model_1.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=learning_rate) # FIXME
    # this is for running the notebook in our testing framework
    train_gauss_model(model_1, likelihood_1, optimizer_1, train_x_1, train_y_1)
    observed_pred = predict_gauss_model(model_1, likelihood_1, test_x_1) # test with train
    plot_gauss_model_eval(train_x_1, train_y_1, test_x_1, observed_pred, show=False)
    kernel_1 = model_1.get_kernel_matrix(train_x_1, detach=True)
    kernel_1 /= (kernel_1.std()+1e-6)
    print("MODEL_1")
    # print("kernel for data 1:\n", kernel_1, kernel_1.std())
    model_1.print_model_hyperparameters()
    # Dimensiones paramétricas del modelo
    # print("Dimensiones paramétricas del modelo : para el set de variables: (index,{})".format(variable))
    # print(likelihood)
    # print(model)
    #print(model.mean_module)
    #print(model.covar_module)
    data_2 = get_data(start_point=16*200, regression_size=regression_size, jump=jump)
    
    train_x_2 = torch.Tensor(data_2["index"].values)
    train_y_2 = torch.Tensor((data_2[variable].values - np.mean(data_2[variable].values))/np.std(data_2[variable].values))
    test_x_2 = train_x_2 #torch.linspace(0,999,1000)
    likelihood_2 = gpytorch.likelihoods.GaussianLikelihood() # FIXME
    model_2 = ExactGPModel(train_x_2, train_y_2, likelihood_2)
    optimizer_2 = torch.optim.Adam([
            {'params': model_2.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=learning_rate) # FIXME
    # this is for running the notebook in our testing framework
    train_gauss_model(model_2, likelihood_2, optimizer_2, train_x_2, train_y_2)
    observed_pred = predict_gauss_model(model_2, likelihood_2, test_x_2) # test with train
    plot_gauss_model_eval(train_x_2, train_y_2, test_x_2, observed_pred, show=False)
    kernel_2 = model_2.get_kernel_matrix(train_x_2, detach=True)
    kernel_2 /= (kernel_2.std()+1e-6)
    print("MODEL_2")
    # print("kernel for data 2:\n", kernel_2, kernel_2.std())
    model_2.print_model_hyperparameters()

    # Kernel difference
    diff_kernel = kernel_1 - kernel_2
    # diff_kernel /= diff_kernel.std()
    # for (i,j), value in np.ndenumerate(diff_kernel):
    #     print("<{},{}> : {}".format(i,j,value))
    print("kernel difference volume:", np.linalg.det(diff_kernel))
    print("kernel difference max :", diff_kernel.max())
plt.show()


