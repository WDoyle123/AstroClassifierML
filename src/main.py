import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

from data_handler import get_data_frame, remove_outliers
from models import dnn_model, svm_model
from plotter import model_accuracy_plot, model_error_plot

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from collections import Counter

def main():
    
    # get data
    df = get_data_frame('star_classification.csv')

    # remove outliers from data
    df = remove_outliers(df)

    # turn categorical data into ints
    df['class'] = pd.factorize(df['class'])[0]

    # crop id columns
    columns_to_drop = [col for col in df.columns if '_ID' in col]
    df.drop(columns_to_drop, axis=1, inplace=True)

    # Calculate the correlation matrix
    corr = df.corr()

    # Isolate the correlation with 'class'
    correlation_with_class = corr['class']

    # Identify columns with correlation below the threshold
    threshold = 0.015
    columns_to_drop = correlation_with_class[abs(correlation_with_class) < threshold].index

    # Drop these columns from the DataFrame
    df.drop(columns_to_drop, axis=1, inplace=True)

    # All columns but class
    X = df.drop('class', axis=1)
    
    # how many inputs
    input_dim =(len(list(X.columns)))

    # Target is class
    y = df['class']

    # Balence between objects 
    sm = SMOTE(random_state=220301)
    X, y = sm.fit_resample(X, y)

    # create split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=220301, stratify=y)

    # scale X values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)   

    # Run DNN
    model, history = dnn_model(X_train_scaled, X_test_scaled, y_train, y_test, input_dim)

    # Evaluate the model
    dnn_scores = model.evaluate(X_test_scaled, y_test)
    final_accuracy = dnn_scores[1]

    # Plot DNN accuracy
    model_accuracy_plot(history, final_accuracy)

    # Run SVM
    score, svm_clf = svm_model(X_train_scaled, X_test_scaled, y_train, y_test)
    svm_score = np.mean(score)

    # Plot Class prediction error
    model_error_plot(svm_clf, X_train_scaled, X_test_scaled, y_train, y_test)
    
    print(f'DNN Score: {final_accuracy}\nSVM Score: {svm_score}')

if __name__ == "__main__":
    main()
