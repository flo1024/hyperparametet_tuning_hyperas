import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation , Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space


np.random.seed(111)

def data():
    from sklearn.decomposition import PCA
    train = pd.read_hdf("train.h5", "train")

    x=train.loc[:, 'x1' : 'x120']
    y=train.y
    PCA=PCA(n_components='mle', copy=True, whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=42)
    PCA.fit(x)
    x=PCA.fit_transform(x, y=None)
    print(x.shape)
    #test=PCA.transform(test)

    y=keras.utils.to_categorical(y,5)
    x = np.expand_dims(x, axis=2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    return x_train, y_train, x_test, y_test

test = pd.read_hdf("test.h5", "test")
test=test.loc[:, 'x1' : 'x120']
x_pred = np.expand_dims(test, axis=2)



#n_timesteps, n_features, n_outputs = x_train.shape[1], 1, 5

def model(x_train, y_train, x_test, y_test):
    model=Sequential()
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Flatten())
    model.add(Dense({{choice([16, 32, 64, 128])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense({{choice([16, 32, 64, 128])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense({{choice([16, 32, 64,128])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense({{choice([8, 16,32,64,128])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size={{choice([128, 256, 512, 1024])}}, epochs=50,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model, space = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials(),
                                          eval_space=True,
                                          return_space=True)
x_train, y_train, x_test, y_test = data()

print("Evalutation of best performing model:")
print(best_model.evaluate(x_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)


y_pred=best_model.predict(x_pred)
y_pred=np.argmax(y_pred, axis=1)

