from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from yellowbrick.classifier import ClassPredictionError

def model_accuracy_plot(history, final_accuracy):

    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot training & validation accuracy values
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Add a horizontal line for the final accuracy
    ax.axhline(y=final_accuracy, color='r', linestyle='--')

    # Add an annotation
    ax.annotate(f'Final Accuracy: {final_accuracy:.2f}%', 
                 xy=(len(history.history['accuracy'])-1, final_accuracy), 
                 xytext=(len(history.history['accuracy'])/2, final_accuracy+5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right', verticalalignment='top')

    # Save the plot
    plt.savefig('../figures/model_accuracy_plot.png', dpi=300)
    plt.close()

def model_error_plot(model, X_train, X_test, y_train, y_test):

    classes=['GALAXY','STAR','QSO']

    visualiser = ClassPredictionError(model, classes=classes)
    visualiser.fit(X_train, y_train)
    visualiser.score(X_test, y_test)

    visualiser.poof(outpath='../figures/model_error_plot.png', dpi=300)

def log_plot(df, columns):

    colors = ['blue', 'green', 'red', 'magenta']

    # Determine the number of rows and columns for the grid
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))

    # Create a figure and an array of subplots with 3 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    plt.subplots_adjust(hspace=0.5)  

    # Loop through the columns and plot each one in a subplot
    for idx, column in enumerate(columns):
        # Ensure the column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        for i in range(3):
            filtered_data = df[df["class"] == i][column]
            if not filtered_data.empty:
                sns.kdeplot(data=np.log(filtered_data), label=([i]), ax=ax, color=colors[i])

        sns.kdeplot(data=np.log(df[column]), label=["All"], ax=ax, color=colors[3])
        ax.legend()

        ax.set_xlabel(column)

    plt.savefig('../figures/data_log_plot.png', dpi=300)
    plt.close()
