import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn import svm
from tensorflow.keras.callbacks import EarlyStopping

def dnn_model(X_train, X_test, y_train, y_test, input_dim):
    
    print('Running DNN Model')

    # Model ~ 97.21% accurate
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='accuracy', patience=5, mode='max', 
                                   restore_best_weights=True, verbose=1, 
                                   baseline=0.95)
    
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=256, callbacks=[early_stopping], verbose=0)

    return model, history

def svm_model(X_train, X_test, y_train, y_test):
 
    print('Running SVM Model')
    # Model ~ 95% Accurate
    svm_clf = svm.SVC(kernel='rbf', C=1, random_state=220301)
    history = svm_clf.fit(X_train, y_train)
    score = svm_clf.score(X_test, y_test)

    return score

