

from matplotlib import pyplot as plt
import gpytorch
import torch
import os
import urllib.request
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
list_variables = ['Series.Close', 'delta.Time'] # 
# 
if(start_point < 1):
    start_point = 1
#################################  DATA  ####################################
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
                'nrows': int(512),
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
#########################  MODEL ###########################

class SpectralDeltaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_deltas, noise_init=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-11))
        likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
        likelihood.noise = 1e-2

        super(SpectralDeltaGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.SpectralDeltaKernel(
            num_dims=train_x.size(-1),
            num_deltas=num_deltas,
        )
        base_covar_module.initialize_from_data(train_x[0], train_y[0])
        self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#######################
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
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
        # Get into evaluation (predictive posterior) mode
def predict_gauss_model(model, likelihood, test_x):
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 512).type(torch.FloatTensor)
        test_x = train_x
        return likelihood(model(test_x))
def plot_gauss_model_eval(train_x, train_y, test_x, test_y):
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
    plt.show()

#################################  MAKE THE MODEL  ####################################

for variable in list_variables:
    train_x = torch.Tensor(data["index"].values)
    train_y = torch.Tensor(data[variable].values - np.mean(data[variable].values))
    test_x = train_x
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood() # FIXME
    model = SpectralDeltaGP(train_x, train_y, likelihood)
    optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=learning_rate) # FIXME
    # this is for running the notebook in our testing framework
    train_gauss_model(model, likelihood, optimizer, train_x, train_y)
    observed_pred = predict_gauss_model(model, likelihood, test_x) # test with train
    plot_gauss_model_eval(train_x, train_y, test_x, observed_pred)
    # Dimensiones paramétricas del modelo
    print("Dimensiones paramétricas del modelo : para el set de variables: (index,{})".format(variable))
    print(likelihood)
    print(model)
    print(model.mean_module)
    print(model.covar_module)




