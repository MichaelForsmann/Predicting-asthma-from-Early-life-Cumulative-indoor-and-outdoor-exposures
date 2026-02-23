import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap
from sklearn.metrics import roc_auc_score,f1_score, r2_score,root_mean_squared_error
import numpy as np 
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold,StratifiedKFold,RepeatedKFold,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import randint, uniform

def nested_cross_validation_classification(X, y, Number_of_repeats, Number_of_splits_pr_repeat, parameters):
    """
    Performs nested cross-validation with hyperparameter tuning using RandomizedSearchCV
    and calculates SHAP values for a Random Forest Classifier model.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series or pd.DataFrame): Target variable.
        Number_of_repeats (int): Number of times to repeat the outer cross-validation loop.
        Number_of_splits_pr_repeat (int): Number of splits for both inner and outer cross-validation.
        para (dict): Dictionary of hyperparameter distributions for RandomizedSearchCV.

    Returns:
        tuple: (performance_df, total_shap_values, mean_feature_importances, parameters_list)
    """
    # Initialize arrays and lists to store metrics and results across all CV iterations.
    # The SHAP array is 3D for classification: (n_samples, n_features, n_classes)
    shap_values_per_cv = np.zeros((X.shape[0], X.shape[1], y.unique().shape[0]))
    performance_test = [] # Stores AUROC scores (since scoring="roc_auc" was used in RandomizedSearchCV)
    performance_train = []
    f1_test = []
    f1_train = []
    feature_importances = []
    parameters_tree = []
    j = 0 # Counter for tracking progress

    # Configure the outer cross-validation strategy (Repeated Stratified K-Fold)
    # Stratified ensures class balance is maintained in each split.
    CrossValidation = RepeatedStratifiedKFold(n_splits=Number_of_splits_pr_repeat, n_repeats=Number_of_repeats,random_state=42) # Set random state for reproducibility
        
        ## Loop through each outer fold (defines the unique test sets across all repeats)
    for i, (train_outer_ix, test_outer_ix) in enumerate(CrossValidation.split(X, y)): 
        # Verbose progress tracking
        j = j + 1  
            
        # Split data into outer training and testing sets
        X_train, X_test = X.iloc[train_outer_ix,:], X.iloc[test_outer_ix,:]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
            
            ## Establish inner CV for parameter optimization (Stratified K-Fold)
            # This inner loop finds the best hyperparameters for a specific outer fold's training data.
        cv_inner = StratifiedKFold(n_splits=Number_of_splits_pr_repeat, random_state=i, shuffle=True)
            
            # Configure Randomized Search for Hyperparameter Tuning
        search = RandomizedSearchCV(estimator=RandomForestClassifier(), # Use consistent random state for base estimator
                param_distributions=parameters,n_iter =20 ,
                cv=cv_inner,
                scoring="f1", # Optimize parameters based on Area Under the ROC Curve
                random_state=42, # Set random state for reproducibility of the search process
                n_jobs=1, # Use all available cores
                verbose=1,refit=True
            )
        # Perform the hyperparameter search on the outer training data
        # The best_estimator_ attribute holds the model with optimal hyperparameters, 
        # already fit on the entire X_train data provided to .fit() during the search process.
        search.fit(X_train, y_train)

        # EXTRACT THE FITTED MODEL
        best_model = search.best_estimator_ 

        # Note: The original script called best_model.fit(X_train, y_train) again here, 
        # which is generally redundant for GridSearchCV/RandomizedSearchCV.

        # Use SHAP to explain predictions using the best estimator
        explainer = shap.TreeExplainer(best_model) 
        # Calculate SHAP values for the held-out outer test set
        # For tree classifiers, shap_values() returns a list/array of shape (n_classes, n_samples, n_features)
        # or in the structure used by the original script: (n_samples, n_features, n_classes)
        shap_values = explainer.shap_values(X_test)
        
        # Aggregate SHAP values across repeats. We sum the SHAP values for the corresponding
        # indices in the original dataset (X). The indices align with the test_outer_ix.
        # Note: Depending on the SHAP version/output format, a dimension swap might be needed here. 
        # The original code assumes a direct assignment works: 
        shap_values_per_cv[test_outer_ix, :, :] += shap_values # Stacks the list of (n_samples, n_features) arrays into the correct 3D shape

        # Record performance metrics for this specific outer fold using the best model
        # .score() returns the metric used in the scoring parameter of the search object ("roc_auc" here)
        if y.unique().shape[0]>2:
            performance_test.append(roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr'))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train), multi_class='ovr'))
        else:
            performance_test.append(roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train)[:,1]))
            
        # Calculate F1 score separately, as it was not the primary optimization metric
        y_test_pred = best_model.predict(X_test)
        y_train_pred = best_model.predict(X_train)
        if y.unique().shape[0]<=2:
            f1_test.append(f1_score(y_test, y_test_pred))
            f1_train.append(f1_score(y_train, y_train_pred))
        else:    
            f1_test.append(f1_score(y_test, y_test_pred,average='macro',pos_label=1))
            f1_train.append(f1_score(y_train, y_train_pred,average='macro',pos_label=1))
        
        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(search.best_params_)
        
        print('done %', (j / (Number_of_repeats * Number_of_splits_pr_repeat)))
    
    # Average the aggregated SHAP values by the total number of repeats to get a final mean SHAP value per sample/class
    total_shap = shap_values_per_cv / Number_of_repeats
    
    # Organize performance metrics into a clean DataFrame
    performance_data = [performance_train, performance_test, f1_train, f1_test]
    performance_index = ["AUROC_train", "AUROC_test", "F1_train", "F1_test"]
    performance_df = pd.DataFrame(performance_data, index=performance_index).T
    
    # Organize feature importances into a DataFrame and calculate the mean across all folds
    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree
def nested_cross_validation_Regression(X, y, Number_of_repeats, Number_of_splits, parameters):
    """
    Performs nested cross-validation with hyperparameter tuning using RandomizedSearchCV
    and calculates SHAP values for a Random Forest Regressor model.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series or pd.DataFrame): Target variable.
        Number_of_repeats (int): Number of times to repeat the outer cross-validation loop.
        Number_of_splits (int): Number of splits for both inner and outer cross-validation.
        para (dict): Dictionary of hyperparameter distributions for RandomizedSearchCV.

    Returns:
        tuple: (performance_df, total_shap_values, mean_feature_importances, parameters_list)
    """
    # Initialize arrays and lists to store metrics and results across all CV iterations
    # SHAP values are averaged over repeats for each sample in the original dataset
    shap_values_per_cv = np.zeros((X.shape[0], X.shape[1]))
    root_mean_squared_error_test = []
    root_mean_squared_error_train = []
    R_squared_test = []
    R_squared_train = []
    feature_importances = []
    parameters_tree = []
    j = 0 # Counter for tracking progress

    # Configure the outer cross-validation strategy (Repeated K-Fold)
    # This loop provides the generalization error estimate
    CV = RepeatedKFold(n_splits=Number_of_splits, n_repeats=Number_of_repeats, random_state=42) # Set random state for reproducibility
    
    ## Loop through each outer fold and extract SHAP values 
    for i, (train_outer_ix, test_outer_ix) in enumerate(CV.split(X)): 
        # Verbose progress tracking
        j = j + 1  
        
        # Split data into outer training and testing sets
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
        
        ## Establish inner CV for parameter optimization (K-Fold)
        # This inner loop finds the best hyperparameters for a specific outer fold
        cv_inner = KFold(n_splits=Number_of_splits, random_state=i, shuffle=True)
        
        # Configure Randomized Search for Hyperparameter Tuning
        search = RandomizedSearchCV(
            estimator=RandomForestRegressor(), # Use a consistent random state for the base estimator
            param_distributions=parameters, 
            cv=cv_inner,
            scoring="r2", # Use R-squared for optimization criteria
            random_state=42, # Set random state for reproducibility of the search process
            n_jobs=-1, # Use all available cores
            verbose=1
        )
        # Perform the hyperparameter search on the outer training data
        result = search.fit(X_train, y_train)
        
        # The best_estimator_ attribute holds the model with optimal hyperparameters, 
        # already fit on the entire X_train data from the search process.
        best_model = result.best_estimator_

        # Note: The original script called best_model.fit(X_train, y_train) again here, 
        # which is redundant as RandomizedSearchCV fits the best estimator automatically 
        # on the full data provided to .fit() after finding the best params.

        # Use SHAP to explain predictions using the best estimator
        explainer = shap.TreeExplainer(best_model) 
        # Calculate SHAP values for the held-out outer test set
        shap_values = explainer.shap_values(X_test)
        
        # Aggregate SHAP values across repeats. We sum the SHAP values for the corresponding
        # indices in the original dataset (X)
        shap_values_per_cv[test_outer_ix, :] += shap_values
        
        # Record performance metrics for this specific outer fold using the best model
        R_squared_test.append(best_model.score(X_test, y_test))
        R_squared_train.append(best_model.score(X_train, y_train))
        root_mean_squared_error_test.append(root_mean_squared_error(y_test, best_model.predict(X_test)))
        root_mean_squared_error_train.append(root_mean_squared_error(y_train, best_model.predict(X_train)))
        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(result.best_params_)
        
        print('done %', (j / (Number_of_repeats * Number_of_splits)))
    
    # Average the aggregated SHAP values by the number of repeats to get a final mean SHAP value per sample
    total_shap = shap_values_per_cv / Number_of_repeats
    
    # Organize performance metrics into a clean DataFrame
    performance_data = [R_squared_train, R_squared_test, root_mean_squared_error_train, root_mean_squared_error_test]
    performance_index = ["R_squared_train", "R_squared_test", "root_mean_squared_error_train", "root_mean_squared_error_test"]
    performance_df = pd.DataFrame(performance_data, index=performance_index).T
    
    # Organize feature importances into a DataFrame and calculate the mean across all folds
    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree
def nested_cross_validation_nodata_classification(X, y, Number_of_repeats, Number_of_splits_pr_repeat, parameters):
    """
    Performs nested cross-validation with hyperparameter tuning using RandomizedSearchCV
    and calculates SHAP values for a Random Forest Classifier model.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series or pd.DataFrame): Target variable.
        Number_of_repeats (int): Number of times to repeat the outer cross-validation loop.
        Number_of_splits_pr_repeat (int): Number of splits for both inner and outer cross-validation.
        para (dict): Dictionary of hyperparameter distributions for RandomizedSearchCV.

    Returns:
        tuple: (performance_df, total_shap_values, mean_feature_importances, parameters_list)
    """
    # Initialize arrays and lists to store metrics and results across all CV iterations.
    # The SHAP array is 3D for classification: (n_samples, n_features, n_classes)
    shap_values_per_cv = np.zeros((X.shape[0], X.shape[1], y.unique().shape[0]))
    performance_test = [] # Stores AUROC scores (since scoring="roc_auc" was used in RandomizedSearchCV)
    performance_train = []
    f1_test = []
    f1_train = []
    feature_importances = []
    parameters_tree = []
    j = 0 # Counter for tracking progress

    # Configure the outer cross-validation strategy (Repeated Stratified K-Fold)
    # Stratified ensures class balance is maintained in each split.
    CrossValidation = RepeatedStratifiedKFold(n_splits=Number_of_splits_pr_repeat, n_repeats=Number_of_repeats,random_state=42) # Set random state for reproducibility
        
        ## Loop through each outer fold (defines the unique test sets across all repeats)
    for i, (train_outer_ix, test_outer_ix) in enumerate(CrossValidation.split(X, y)): 
        # Verbose progress tracking
        j = j + 1  
            
        # Split data into outer training and testing sets
        X_train, X_test = X.iloc[train_outer_ix,:], X.iloc[test_outer_ix,:]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
        scaler = StandardScaler()
        scale_house = scaler.fit(X_train)
        X_house=scale_house.transform(X_train)
        imputer = KNNImputer(n_neighbors=10)
        X_scaled_landuse=imputer.fit_transform(X_house)
        X_train=pd.DataFrame(X_scaled_landuse,index=X_train.index,columns=X_train.columns)
        scale_house = scaler.fit(X_test)
        X_house=scale_house.transform(X_test)
        imputer = KNNImputer(n_neighbors=10)
        X_scaled_landuse=imputer.fit_transform(X_house)
        X_test=pd.DataFrame(X_scaled_landuse,index=X_test.index,columns=X_test.columns)   
            ## Establish inner CV for parameter optimization (Stratified K-Fold)
            # This inner loop finds the best hyperparameters for a specific outer fold's training data.
        cv_inner = StratifiedKFold(n_splits=Number_of_splits_pr_repeat, random_state=i, shuffle=True)
            
            # Configure Randomized Search for Hyperparameter Tuning
        search = RandomizedSearchCV(estimator=RandomForestClassifier(), # Use consistent random state for base estimator
                param_distributions=parameters,n_iter =20 ,
                cv=cv_inner,
                scoring="f1", # Optimize parameters based on Area Under the ROC Curve
                random_state=42, # Set random state for reproducibility of the search process
                n_jobs=1, # Use all available cores
                verbose=1,refit=True
            )
        # Perform the hyperparameter search on the outer training data
        # The best_estimator_ attribute holds the model with optimal hyperparameters, 
        # already fit on the entire X_train data provided to .fit() during the search process.
        

        search.fit(X_train, y_train)

        # EXTRACT THE FITTED MODEL
        best_model = search.best_estimator_ 

        # Note: The original script called best_model.fit(X_train, y_train) again here, 
        # which is generally redundant for GridSearchCV/RandomizedSearchCV.

        # Use SHAP to explain predictions using the best estimator
        explainer = shap.TreeExplainer(best_model) 
        # Calculate SHAP values for the held-out outer test set
        # For tree classifiers, shap_values() returns a list/array of shape (n_classes, n_samples, n_features)
        # or in the structure used by the original script: (n_samples, n_features, n_classes)
        shap_values = explainer.shap_values(X_test)
        
        # Aggregate SHAP values across repeats. We sum the SHAP values for the corresponding
        # indices in the original dataset (X). The indices align with the test_outer_ix.
        # Note: Depending on the SHAP version/output format, a dimension swap might be needed here. 
        # The original code assumes a direct assignment works: 
        shap_values_per_cv[test_outer_ix, :, :] += shap_values # Stacks the list of (n_samples, n_features) arrays into the correct 3D shape

        # Record performance metrics for this specific outer fold using the best model
        # .score() returns the metric used in the scoring parameter of the search object ("roc_auc" here)
        if y.unique().shape[0]>2:
            performance_test.append(roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr'))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train), multi_class='ovr'))
        else:
            performance_test.append(roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train)[:,1]))
            
        # Calculate F1 score separately, as it was not the primary optimization metric
        y_test_pred = best_model.predict(X_test)
        y_train_pred = best_model.predict(X_train)
        if y.unique().shape[0]<=2:
            f1_test.append(f1_score(y_test, y_test_pred))
            f1_train.append(f1_score(y_train, y_train_pred))
        else:    
            f1_test.append(f1_score(y_test, y_test_pred,average='macro',pos_label=1))
            f1_train.append(f1_score(y_train, y_train_pred,average='macro',pos_label=1))
        
        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(search.best_params_)
        
        print('done %', (j / (Number_of_repeats * Number_of_splits_pr_repeat)))
    
    # Average the aggregated SHAP values by the total number of repeats to get a final mean SHAP value per sample/class
    total_shap = shap_values_per_cv / Number_of_repeats
    
    # Organize performance metrics into a clean DataFrame
    performance_data = [performance_train, performance_test, f1_train, f1_test]
    performance_index = ["AUROC_train", "AUROC_test", "F1_train", "F1_test"]
    performance_df = pd.DataFrame(performance_data, index=performance_index).T
    
    # Organize feature importances into a DataFrame and calculate the mean across all folds
    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree    