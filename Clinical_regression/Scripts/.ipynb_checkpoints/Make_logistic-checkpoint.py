import arviz as az
import pandas as pd
import torch 
from sklearn.model_selection import train_test_split
from pyro.infer import MCMC, NUTS, Predictive,HMC
from pyro import sample
import pyro.distributions as dist 
import pyro  
from pyro import clear_param_store
from sklearn.metrics import roc_auc_score,f1_score
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
def logistic_regression_model(features, target=None):
    # Ensure features is a float tensor
    features = features.float() 
    
    # Determine the number of features and data points dynamically
    num_data = int(features.size(0))
    num_features = int(features.shape[1]) # Corrected index
    weight = pyro.sample("weight", dist.Normal(torch.zeros(num_features), torch.ones(num_features)))
    bias = pyro.sample("bias", dist.Normal(torch.zeros(1), torch.ones(1)).to_event(0))
    logits = torch.matmul(features, weight) + bias
    
    # 4. Define the likelihood in a fully vectorized plate over the data dimension
    with pyro.plate("data", num_data):
        return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=target)

def run_inference(features: torch.Tensor, target: torch.Tensor, N_warmup_steps=4000):
    # Initialise NUTS sampler
    kernel = NUTS(logistic_regression_model)
    # Initialise MCMC class
    mcmc = MCMC(kernel, num_samples = N_warmup_steps, warmup_steps = N_warmup_steps)
    # Run sampling
    mcmc.run(features.float(),target)
    return mcmc,logistic_regression_model
def get_arviz(mcmc,x_training):
    posterior_samples = mcmc.get_samples(500)
    posterior_predictive= Predictive(logistic_regression_model, posterior_samples)(x_training)
    prior = Predictive(logistic_regression_model, num_samples=500)(x_training)
    return az.from_pyro(mcmc,prior=prior,posterior_predictive=posterior_predictive)
def train_model(X_train,y_train,i,N_CV,name,steps=4500):
    mcmc,kernel=run_inference(features=X_train, target=y_train, N_warmup_steps=4000)
    posterior_samples = mcmc.get_samples(500)
    predictive= Predictive(logistic_regression_model, posterior_samples)
    posterior_predictive=predictive((X_train))
    prior = Predictive(logistic_regression_model, num_samples=500)(X_train)
    pyro_data = az.from_pyro(mcmc,prior=prior,posterior_predictive=posterior_predictive)
    az.to_json(pyro_data, name+str(i)+str(N_CV)+".json")
    return mcmc,kernel,predictive

def nested_cross_baysian_logistic(X,y,Number_CV,number_split,name):
    X,y=torch.from_numpy(X.values).float(),torch.tensor(y.values).float()
    roc_test=np.zeros(Number_CV*number_split)
    f1_test=np.zeros(Number_CV*number_split)
    roc_train=np.zeros(Number_CV*number_split)
    f1_train=np.zeros(Number_CV*number_split)
    j=0
    for i, N_CV in enumerate(range(Number_CV)): 
        skf =StratifiedKFold(n_splits=number_split, random_state=None, shuffle=True) # Set random state 
        for i, (train_outer_ix, test_outer_ix) in enumerate(skf.split(X, y)): 
           
            X_train, X_test = X[train_outer_ix, :],X[test_outer_ix, :]
            y_train, y_test = y[train_outer_ix], y[test_outer_ix]
            mcmc,kernel ,predictive=train_model(X_train,y_train,i,N_CV,name)
            roc_train[j]=roc_auc_score(y_train, predictive(X_train)["obs"].mean(axis=0))
            f1_train[j]=f1_score(y_train,kernel(X_train))
            roc_test[j]=roc_auc_score(y_test, predictive(X_test)["obs"].mean(axis=0))
            f1_test[j]=f1_score(y_test,kernel(X_test))
            j=j+1
    performence=pd.DataFrame([roc_train,roc_test,f1_train,f1_test],index=["roc_train","roc_test","f1_train","f1_test"]).T   
    return performence,kernel,mcmc