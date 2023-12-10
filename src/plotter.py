from matplotlib import pyplot as plt
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
    plt.savefig('../model_accuracy_plot.png', dpi=300)
    plt.close()

def model_error_plot(model, X_train, X_test, y_train, y_test):

    classes=['GALAXY','STAR','QSO']

    visualiser = ClassPredictionError(model, classes=classes)
    visualiser.fit(X_train, y_train)
    visualiser.score(X_test, y_test)
    visualiser.show()

    visualiser.poof(outpath='../model_error_plot.png', dpi=300)
