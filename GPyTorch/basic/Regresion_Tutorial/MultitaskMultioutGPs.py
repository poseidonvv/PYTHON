######## MULTITASKS / MULTIOUTPUT  GPS WITH EXACT INFERENCE 

import math
import torch
import gpytorch
import pandas as pd 
from matplotlib import pyplot as plt
import os
from datetime import datetime
import numpy as np

##################### DATA ###################################################################
#train_x = torch.linspace(0, 1, 100)
#


#################################  PARAMS  ####################################
     
regression_size = 51
start_point = 1
training_iter = 100
learning_rate = 0.05
list_variables = ['Series.Close', 'delta.Time'] # 
jump = 1
####################   DATA ES VOLUME 100 ############################
if(start_point < 1):
    start_point = 1
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


####################### DEFINE A MULTITASK MODEL ############################################

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



#########################  TRAIN THE MODEL HYPERPARAMETERS ###################################


'''
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
'''

################## MAKE PREDICTIONS WITH THE MODEL ##############################################
data_1 = get_data(start_point=start_point, regression_size=regression_size, jump=jump)
train_x = torch.Tensor(data_1["index"].values)
# torch.stack([torch.Tensor(data_1["index"].values), torch.Tensor(data_1["index"].values)], -1)
train_y = torch.stack([torch.Tensor((data_1[variable].values - np.mean(data_1[variable].values))/np.std(data_1[variable].values)) for variable in list_variables], -1)
print(train_y.shape,train_x.shape)
test_x = train_x
# initialize likelihood and model
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(list_variables))
model = MultitaskGPModel(train_x, train_y, likelihood)
optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=learning_rate) # FIXME

# Set into eval mode
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()



# Make predictions
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    model_out = model(test_x)
    predictions = likelihood(model_out)
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

# Initialize plots
f, axis = plt.subplots(1, len(list_variables), figsize=(8, 3))
for item in range(len(list_variables)):
    # Plot training data as black stars
    axis[item].plot(train_x.detach().numpy(), train_y[:, item].detach().numpy(), 'k*')
    # Predictive mean as blue line
    axis[item].plot(test_x.numpy(), mean[:, item].numpy(), 'b')
    # Shade in confidence
    axis[item].fill_between(test_x.numpy(), lower[:, item].numpy(), upper[:, item].numpy(), alpha=0.5)
    # axis[item].set_ylim([-3, 3])
    axis[item].legend(['Observed Data', 'Mean', 'Confidence'])
    axis[item].set_title('Values (Likelihood) <{}>'.format(list_variables[item]))
plt.show()
########################### MAKE MODEL  ###############################################

   
