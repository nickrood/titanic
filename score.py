import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):

    for var in config.CATEGORICAL_VARS:
        data[var] = pf.impute_na(data, var, replacement='Missing')

    # extract first letter from cabin
     for var in config.IMPUTATION_DICT
         data[var] = data [var].str[0] 

    # impute NA categorical
    for var in config.CATEGORICAL_VARS:
        data [var] = pf.impute_na(data, var, replacement='Missing'
    
    
    # impute NA numerical
    for var in config.NUMERCAL_TO_IMPUTE:
        data[var]= pf.impute_na(X_train,var,replacement= 'Missing')
    
    
    # Group rare labels
    for var in config.CATEGORICAL_VARS:
    data[var] = pf.remove_rare_labels(data, var, config.FREQUENT_LABELS[var])
    
    # encode variables
    for var in config.CATEGORICAL VARS 
        data = data.copy()
        data = pd.concat([data, pd.get_dummies(data[var], prefix=var, drop_first=drop_first)]
        , axis=axis)
        data.drop(labels=var, axis=1, inplace=True)
        
    # check all dummies were added

    
    # scale variables
    data = pf.scale_features(data[config.FEATURES],
                             config.OUTPUT_SCALER_PATH) 
    
    # make predictions
    predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)

    
    return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)
    
    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        