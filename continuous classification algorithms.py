#Machine learning learning notes

# =========================================================
# 1. Data preparation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("wine dataset/winequality-white.csv", sep=';')

print(df.head())
print('dataframe size: ' + str(df.shape)) 

plt.figure()
plot_num = 0
#for label in list(df)[:-1]: #the last one is the quality
#    plot_num += 1
#    plt.subplot(3,4,plot_num)
#    plt.scatter(df[label], df['quality']) # density returns the proportion of observations per bin
#    plt.title(label)
#    plt.ylabel = ('quality')
#    plt.xlabel = (str(label))

#plt.show()

# train, validation, training
train, validation, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def get_xy(dataframe, y_label, x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None:
        X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels) == 1:
            X = dataframe[x_labels[0]].values.reshape(-1,1)
        else:
            X = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1,1)
    data = np.hstack((X,y))

    return data, X, y

_, X_train_alc, y_train_alc = get_xy(df,'quality',x_labels=["alcohol"])
_, X_val_alc, y_val_alc = get_xy(df,'quality',x_labels=["alcohol"])
_, X_test_alc, y_test_alc = get_xy(df,'quality',x_labels=["alcohol"])

alc_reg = LinearRegression()

alc_reg.fit(X_train_alc, y_train_alc)
print(alc_reg.coef_, alc_reg.intercept_)
print(alc_reg.score(X_test_alc, y_test_alc))

#figure()
plt.scatter(X_train_alc, y_train_alc, label="Data", color="blue")
x = tf.linspace(5,20,100)
plt.plot(x,alc_reg.predict(np.array(x).reshape(-1,1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Quality vs alcohol")
plt.ylabel = "quality"
plt.xlabel = "alcohol"
#plt.show()

# Multiple linear regression
_, X_train, y_train = get_xy(df,'quality',x_labels=df.columns[:-1])
_, X_val, y_val = get_xy(df,'quality',x_labels=df.columns[:-1])
_, X_test, y_test = get_xy(df,'quality',x_labels=df.columns[:-1])

all_reg = LinearRegression()
all_reg.fit(X_train, y_train)
print(all_reg.score(X_test,y_test))

# Simple regression with neural net
def plot_loss(history):
    #figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel = 'Epoch'
    plt.ylabel = 'MSE'
    plt.legend()
    plt.grid(True)

    plt.show()

alc_normaliser = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
alc_normaliser.adapt(X_train_alc.reshape(-1))

alc_nn_model = tf.keras.Sequential([
    alc_normaliser,
    tf.keras.layers.Dense(1)
])

alc_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

history = alc_nn_model.fit(
    X_train_alc.reshape(-1), y_train_alc,
    #verbose = 0,
    epochs = 100,
    validation_data=(X_val_alc,y_val_alc)
)
plot_loss(history)

# Simple regression using neural nets
nn_model = tf.keras.Sequential([
    alc_normaliser,
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'relu')
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
history = nn_model.fit(
    X_train_alc, y_train_alc,
    validation_data = (X_val_alc, y_val_alc),
    #verbose = 0, 
    epochs=100
)
plot_loss(history)