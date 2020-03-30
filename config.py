# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
IMPUTATION_DICT = 'cabin'


# encoding parameters
FREQUENT_LABELS = {
	'sex' : ['male', 'female'],
	'cabin' : ['B5', 'C22', 'E12', 'D7', 'A36', 'C101', 'nan', 'C62', 'B35', 'A23',
		       'B58', 'D15', 'C6', 'D35', 'C148', 'C97', 'B49', 'C99', 'C52', 'T',
		       'A31', 'C7', 'C103', 'D22', 'E33', 'A21', 'B10', 'B4', 'E40',
		       'B38', 'E24', 'B51', 'B96', 'C46', 'E31', 'E8', 'B61', 'B77', 'A9',
		       'C89', 'A14', 'E58', 'E49', 'E52', 'E45', 'B22', 'B26', 'C85',
		       'E17', 'B71', 'B20', 'A34', 'C86', 'A16', 'A20', 'A18', 'C54',
		       'C45', 'D20', 'A29', 'C95', 'E25', 'C111', 'C23', 'E36', 'D34',
		       'D40', 'B39', 'B41', 'B102', 'C123', 'E63', 'C130', 'B86', 'C92',
		       'A5', 'C51', 'B42', 'C91', 'C125', 'D10', 'B82', 'E50', 'D33',
		       'C83', 'B94', 'D49', 'D45', 'B69', 'B11', 'E46', 'C39', 'B18',
		       'D11', 'C93', 'B28', 'C49', 'B52', 'E60', 'C132', 'B37', 'D21',
		       'D19', 'C124', 'D17', 'B101', 'D28', 'D6', 'D9', 'B80', 'C106',
		       'B79', 'C47', 'D30', 'C90', 'E38', 'C78', 'C30', 'C118', 'D36',
		       'D48', 'D47', 'C105', 'B36', 'B30', 'D43', 'B24', 'C2', 'C65',
		       'B73', 'C104', 'C110', 'C50', 'B3', 'A24', 'A32', 'A11', 'A10',
		       'B57', 'C28', 'E44', 'A26', 'A6', 'A7', 'C31', 'A19', 'B45', 'E34',
		       'B78', 'B50', 'C87', 'C116', 'C55', 'D50', 'E68', 'E67', 'C126',
		       'C68', 'C70', 'C53', 'B19', 'D46', 'D37', 'D26', 'C32', 'C80',
		       'C82', 'C128', 'E39', 'D', 'F4', 'D56', 'F33', 'E101', 'E77', 'F2',
		       'D38', 'F', 'E121', 'E10', 'G6', 'F38'],
    'embarked' : ['S', 'C', 'nan', 'Q'],
    'title' : ['Miss', 'Master', 'Mr', 'Mrs', 'Other']
}


DUMMY_VARIABLES = ['sex', 'cabin', 'embarked', 'title']


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['pclass', 'age', 'sibsp', 'parch', 'fare']