import argparse
import pickle as pickle

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

font = {'size': 14}
matplotlib.rc('font', **font)


def get_data(filename, sep="\t"):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    data: A dataframe containing features to be preprocessed and used for training.
    y: A dataframe containing labels, used for model response.
    """
    data = pd.read_csv(filename, sep=sep, header=0)
    # only keep rows that are +/- 3 standard deviations from the mean
    outlier_mask = (np.abs(stats.zscore(data.select_dtypes(include=np.number))) < 3).all(axis=1)
    data = data[outlier_mask]
    if 'price' in data.columns:
        data['log_price'] = np.log(data['price'])
        y = data.pop('log_price')
        return data, y
    return data


def preprocessing(data):
    """Take cleaned dataframe and preprocess features to prepare for training

    Parameters
    ----------
    data: a dataframe containing features to be preprocessed and used for training.

    Returns
    -------
    X: A dataframe of preprocessed data ready for training
    """
    data.loc[:, ['x', 'y', 'z']] = data[['x', 'y', 'z']].replace(
        0, np.nan).fillna(data[['x', 'y', 'z']].mean())
    data['clarity'] = data['clarity'].map(
        {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8})
    data['cut'] = data['cut'].map({'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5})
    data['color'] = data['color'].map({'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1})
    data['log_volume'] = np.log(data['x'] * data['y'] * data['z'])
    data_model = data[['color', 'cut', 'clarity', 'log_volume', 'table']]
    X = data_model.copy()
    return X


def train(filename, use_tree):
    """Load raw data from a file and train either using linear regression or random forest

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.
    use_tree: if True, use Random Forest Regressor

    Returns
    -------
    model: a model fit using either linear regression or random forest
    """
    X, y = get_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-2)
    lm = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-2)
    if use_tree:
        clf = rf
    else:
        clf = lm
    model = Pipeline(steps=[
        ('preprocessing', FunctionTransformer(preprocessing, validate=False)),
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    summary(np.exp(y_test), np.exp(y_predicted))
    return model


def summary(y_true, y_predicted):
    """return descriptive statistics of predicted and true responses

    Parameters
    ----------
    y_true: the true value of the response
    y_predicted: the predicted value of the response

    Returns
    -------
    summary_output: descriptive statistical metrics
    """
    ax = sns.regplot(y_true, y_predicted)
    ax.set(xlabel='True Price ($)', ylabel='Predicted Price($)')
    plt.show()
    nme = mean_absolute_error(y_true, y_predicted)
    nmse = np.sqrt(mean_squared_error(y_true, y_predicted))
    rs = r2_score(y_true, y_predicted)
    summary_output = f'''
        mean absolute error = {nme : .2f}
        root mean squared error = {nmse : .2f}
        R squared = {rs : .2f}
    '''
    print(summary_output)


def predict(filename, model_input_path):
    """Predict responses based on feature inputs

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.
    model_path: the location of the model
    Returns
    -------
    y_preds: predictions of the responses based on the input model
    """
    X = get_data(filename)

    model = joblib.load(model_input_path)
    y_preds = model.predict(X)
    return np.exp(y_preds)


def parse_arguments():
    """command line arguments

    Parameters
    ----------
    Returns
    -------
    args: arguments used for executing script on the command line
    """
    parser = argparse.ArgumentParser(
        description='Fit a Text Classifier model and save the results.')
    parser.add_argument('--data', help='A tab delimited csv file with input data.')
    parser.add_argument('--model_output_path',
                        help='A file to save the serialized model object to.', default='model.joblib')
    parser.add_argument('mode', help='train or predict model', default='predict')
    parser.add_argument('--output_file', help='where to save the model predictions',
                        default='predictions.txt')
    parser.add_argument('--tree_model', action='store_true',
                        help='if True, use Random Forest Model')
    parser.add_argument('--no_tree_model', action='store_false',
                            help='if false, do not use Random Forest Model')
    parser.add_argument('--model_input_path', help='model to load', default='model.joblib')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        model = train(args.data, args.tree_model)
        joblib.dump(model, args.model_output_path)
    if args.mode == 'predict':
        preds = predict(args.data, args.model_input_path)
        np.savetxt(args.output_file, preds, delimiter="\t")
