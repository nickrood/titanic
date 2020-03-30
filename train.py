import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df, config.TARGET)


# get first letter from cabin variable
for var in config.IMPUTATION_DICT:
    X_train[var] = X_train[var].str[0]    


# impute categorical variables
for var in config.CATEGORICAL_VARS:
    X_train[var] = pf.impute_na(X_train, var, replacement='Missing')

# impute numerical variable
X_train[config.NUMERICAL_TO_IMPUTE] = pf.impute_na(X_train,
       config.NUMERICAL_TO_IMPUTE,
       replacement= 'Missing')

# Group rare labels
for var in config.CATEGORICAL_VARS:
    X_train[var] = pf.remove_rare_labels(X_train, var, config.FREQUENT_LABELS[var])


# encode categorical variables
for var in config.CATEGORICAL_VARS: 

    X_train = X_train.copy()
    X_train = pd.concat([X_train, pd.get_dummies(X_train[var], prefix=var, drop_first=drop_first)]
    , axis=axis)
    X_train.drop(labels=var, axis=1, inplace=True)


# check all dummies were added



# train scaler and save
scaler = pf.train_scaler(X_train[config.NUMERICAL_TO_IMPUTE],
                         config.OUTPUT_SCALER_PATH)

# scale train set
X_train = scaler.transform(X_train[config.NUMERICAL_TO_IMPUTE])


# train model and save
pf.train_model(X_train, y_train), config.OUTPUT_MODEL_PATH


print('Finished training')