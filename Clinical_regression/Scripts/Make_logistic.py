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
    # Ensure the input features are converted to a float tensor for calculation
    features = features.float() 
    
    # Get the total number of data points (rows) from the features tensor
    num_data = int(features.size(0))

    # Get the total number of individual features (columns) from the features tensor
    num_features = int(features.shape[1]) # Corrected index

    # Sample weights for each feature from a standard normal distribution
    weight = pyro.sample("weight", dist.Normal(torch.zeros(num_features), torch.ones(num_features)))

    # Sample a bias term and ensure it is treated as a single event dimension
    bias = pyro.sample("bias", dist.Normal(torch.zeros(1), torch.ones(1)).to_event(0))
    # Calculate the logits by performing matrix multiplication of features and weights plus bias
    logits = torch.matmul(features, weight) + bias
    
    # Define the likelihood using a plate to declare independence across the data rows
    with pyro.plate("data", num_data):
        # Sample the observations from a Bernoulli distribution using the calculated logits
        return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=target)

def run_inference(features: torch.Tensor, target: torch.Tensor, N_warmup_steps=4000):
    # Initialise the No-U-Turn Sampler (NUTS) using the defined logistic regression model
    kernel = NUTS(logistic_regression_model)
    
    # Create the MCMC object with the NUTS kernel, setting the number of samples and warmup steps
    mcmc = MCMC(kernel, num_samples = N_warmup_steps, warmup_steps = N_warmup_steps)
    
    # Execute the MCMC sampling process using the features cast to float and the target values
    mcmc.run(features.float(),target)
    return mcmc,logistic_regression_model

def train_model(X_train,y_train,i,N_CV,name,steps=4500):
        # Execute the MCMC inference process using the defined features and targets
    mcmc,kernel=run_inference(features=X_train, target=y_train, N_warmup_steps=4000)
    
    # Extract posterior samples and wrap them in an ArviZ InferenceData object for diagnostics
    posterior_samples = mcmc.get_samples(500)
    
    # Generate prior predictive samples by running the model with 500 random samples
    predictive= Predictive(logistic_regression_model, posterior_samples)
    
    # Generate posterior predictive samples based on the training features
    posterior_predictive=predictive((X_train))
    
    # Generate prior predictive samples by running the model with 500 random samples
    prior = Predictive(logistic_regression_model, num_samples=500)(X_train)
    
    # Convert the MCMC results, prior, and posterior predictive data to an ArviZ object
    pyro_data = az.from_pyro(mcmc,prior=prior,posterior_predictive=posterior_predictive)
    
    # Export the ArviZ InferenceData object to a JSON file with a unique filename
    az.to_json(pyro_data, name+str(i)+str(N_CV)+".json")
    return mcmc,kernel,predictive

def nested_cross_baysian_logistic(X,y,Number_CV,number_split,name):
    # Convert pandas DataFrame and Series to torch tensors for Pyro compatibility
    X,y=torch.from_numpy(X.values).float(),torch.tensor(y.values).float()

    # Initialize arrays to store ROC AUC scores for each split
    roc_test=np.zeros(Number_CV*number_split)

    # Initialize arrays to store F1 scores for each split
    f1_test=np.zeros(Number_CV*number_split)

    # Initialize arrays to store training ROC AUC scores
    roc_train=np.zeros(Number_CV*number_split)

    # Initialize arrays to store training F1 scores
    f1_train=np.zeros(Number_CV*number_split)
    
    # Counter to keep track of the current iteration across nested loops
    j=0
    
    # Outer loop for the number of cross-validation repetitions
    for i, N_CV in enumerate(range(Number_CV)): 

        # Define the StratifiedKFold cross-validator
        skf =StratifiedKFold(n_splits=number_split, random_state=None, shuffle=True) # Set random state 
        
        # Inner loop to iterate through each split within the current cross-validation
        for i, (train_outer_ix, test_outer_ix) in enumerate(skf.split(X, y)): 
           # Split the features into training and testing sets based on indices
            X_train, X_test = X[train_outer_ix, :],X[test_outer_ix, :]

            # Split the target labels into training and testing sets
            y_train, y_test = y[train_outer_ix], y[test_outer_ix]

            # Train the Bayesian model and get MCMC results and predictive objects
            mcmc,kernel ,predictive=train_model(X_train,y_train,i,N_CV,name)

            # Calculate and store ROC AUC for the training set using mean predictions
            roc_train[j]=roc_auc_score(y_train, predictive(X_train)["obs"].mean(axis=0))

            # Calculate and store F1 score for the training set
            f1_train[j]=f1_score(y_train,kernel(X_train))

            # Calculate and store ROC AUC for the test set using mean predictions
            roc_test[j]=roc_auc_score(y_test, predictive(X_test)["obs"].mean(axis=0))

            # Calculate and store F1 score for the test set
            f1_test[j]=f1_score(y_test,kernel(X_test))
            
            # Increment the global counter
            j=j+1
            
             # Compile all performance metrics into a pandas DataFrame for analysis
    performence=pd.DataFrame([roc_train,roc_test,f1_train,f1_test],index=["roc_train","roc_test","f1_train","f1_test"]).T 
    # Return the performance summary, the last used kernel, and the last MCMC object
    return performence,kernel,mcmc