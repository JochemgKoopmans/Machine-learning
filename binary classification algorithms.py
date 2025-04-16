#Machine learning learning notes

# =========================================================
# 1. Data preparation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# ===== Inspect data ======
# --> What values do the columns represent, do columns have names?
# --> Are all columns numerical or do they need to be transformed?

df = pd.read_csv("wine dataset/winequality-red.csv", sep=';')

print(df.head())
print('dataframe size: ' + str(df.shape)) 

# #1599 wine entries and 12 wine variables (3x acidity, sugar, chlorides, SO2, density, pH, sulphates, alcohol, quality)

# --> Is there a (binary) classification column?
print(df['quality'].unique())

df['passing_grade']  = df['quality'] > 5 #(df['quality'] > 5).astype(int)

# --> How are the quality ratings/passing grades distributed?
plt.figure(1)
plt.hist(df['quality']) # or 'passing_grade'
plt.title("Quality ratings")
plt.ylabel("Occurrences")
plt.xlabel("Quality")

# --> Visualise dataset, see if any variable should make for a good classification predictor:
plt.figure(2)
plot_num = 0
for label in list(df)[:-1]: #the last one is the quality
    plot_num += 1
    plt.subplot(3,4,plot_num)
    plt.hist(df[df['passing_grade']==True][label], color='green', label='pass', alpha=.7, density=True) # density returns the proportion of observations per bin
    plt.hist(df[df['passing_grade']==False][label], color='red', label='fail', alpha=.7, density=True)  # alpha sets the transparency of bars
    plt.title(label)
    plt.ylabel = ("probability")
    plt.xlabel = (str(label))
    plt.legend()

#plt.show()

# Visual identifiers of pass-quality wines:
#  - Volatile acidity: less seems better
#  - Citric acid: more seems better
#  - Total sulphur dioxide: less seems better
#  - Density: fail-quality wines are more centralised to the mean
#  - Sulphates: more seems better
#  - Alcohol: More is a strong indication that is not fail-quality

# ===== Train, validation, test datasets =====

train, validation, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
# np.split(dataframe, [indexes where to split])
# df.sample(fraction-to-return=1)

# --> Standardize predictor (column) values to avoid biases towards large values 
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-2]].values #leaves out the class columns
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X) # Standardises the predictor columns in the dataset

    if oversample: #If one class is overrepresented in the training group, that can skew predictions. Oversampling ensures a balanced training set.
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X,np.reshape(y,(len(y),1))))
    return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
validation, X_validation, y_validation = scale_dataset(validation, oversample=False) # Necessary only for neural networks (TBA)
test, X_test, y_test = scale_dataset(test, oversample=False) # False because the classifier must be able to predict future (hence non-oversampled) data

# =========================================================
# 2. Machine classification: K-nearest neighbours
print('K-Nearest Neighbours:')
# The idea is that all your data are individual entries in a large table.
# The data can be plotted in a (high-dimensional) plot, which a label for which group a data point belongs to.
# For any new datapoint, its group can be predicted by checking which groups its nearest neighbouring points belong to.
# Output is non-probabilistic

# --> Especially useful when classes generally group together, 
#       but (due to high-dimensionality) the factors determining group belonging are unclear
# --> Applicable to smaller datasets
# --> Takes more resources (time) for larger datasets
# --> Should be relatively easily expandable to multi-class classification
# --> An interesting addition could be a datapoint 'typicality', that is the distance to its nearest neighbours
# --> Another addition could be the 'certainty', that is which groups the neighbours belong to

# ---------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3) # One heuristic is to use n = sqrt(training set length)
knn_model.fit(X_train, y_train)

y_predKNN = knn_model.predict(X_test)

print(classification_report(y_test, y_predKNN))
#precision: out of all that were predicted as True/False, how many were indeed that?
#recall: out of all that were True/False, how many were recalled as such?

# =========================================================
# 3. Machine classification: Naive Bayes
print('Naive Bayes:')
# The predictor variables are treated as evidence for an observation belonging to a specific class.
# The algorithm calculates the probability of class K among all observations, the probability of evidence x,
#   and the probability of the observed evidence given that it is a member of class K. 
# Together (probability of the evidence, given class K)*(probability of class K)/probability(evidence) is the posterior probability or
#   the probability of an observation being in class K given the evidence. It mathematically reverses the epistemic question of x given K to K given x.
# It calculates the posterior probability (of an observation belonging to class K) for each class and returnst the most likely class.
# Output can be either probabilistic or not

# --> Naive bayes assumes that predictors are uncorrelated (hence naive). Should one correct for correlated predictors?
# --> Increases in internal validity with larger numbers of observations, decreases in validity with more predictors?
# --> Is relatively fast for larger datasets
# --> Is suitable for multi-class predictions

# ---------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
#naive bayes assumes that all predictors are independent (non-correlated)

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_predNB = nb_model.predict(X_test)

print(classification_report(y_test, y_predNB))

# =========================================================
# 4. Machine classification: Logistic regression
print('Logistic regression:')
# Fits a logistic regression to plotted evidence (x-values of an observation) and class belonging (y-value).
# A linear regression is too continuous (straight line) to demarcate class boundaries in terms of their corresponding x-values,
#   so a logistic regression (s-curve) is preferrable.
# A simple logistic regression takes one independent variable (x), multiple logistic regression takes multiple.
# Multiple logistic regression calculates the y-value (used to predict class membership) by adding (weighted) the x-values 
#   for each predictor into a single (0-1, for two classes) y-value. 
# Output is probabilistic

# --> Applies to two classes, not immediately obvious how to expand to multiple classes (look into multinomial or ordinal logistic regression)
# --> Can be sensitive to overfitting if there are relatively few observations (few is when there's less than 10-20*number of predictors)

# ---------------------------------------------------------

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_predLG = lg_model.predict(X_test)

print(classification_report(y_test, y_predLG))

# =========================================================
# 5. Machine classification: Support vector machines
print('Support vector machines:')
# Predictors are axes in an n-dimensional matrix, where each axis represents one predictor.
# The line or (hyper) plane that best separates the classes in this matrix is the decision boundary. 
# The best line or (hyper) plane is chosen such that it maximises the distance between the nearest data points from either class.
# These nearest points are called support vectors. 
# Its output is non-probabilistic

# --> Sensitive to outliers and to class point clouds overlapping (but that should be solvable with virtual support vectors 
#       that are class averages?). Overlapping point clouds (non-linear data) can be solved using kernel tricks (predictor transformations).
# --> Perform better with high-dimensional and unstructured datasets and are less prone to overfitting than logistic regression
# --> Can be computationally expensive compared to logistic regression
# ---------------------------------------------------------

from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_predSVM = svm_model.predict(X_test)

print(classification_report(y_test, y_predSVM))

# =========================================================
# 6. Neural networks
# Predictors serve as inputs (weighted) to an interim summation (neuron), which receives a bias and 
#   serves as further input to a new layer of neurons. In itself, this is equivalent to a linear function.
# What makes it a neural network is that a neuron's output is passed through a (non-linear) activation function, 
#   which takes its original output value and determines the neuron's activation by computing the y-value that corresponds to it
#   in the chosen activation function (e.g. sigmoid s-curve function running 0-1, or any complex non-linear function)

# There are various parameters to change that will affect a neural network's performance. By iterating over different combinations,
#   the best NN can be chosen to predict new values' class membership.

# --> Takes a lot of computer power/time, it does not necessarily perform better than the other classification algorithms
# ---------------------------------------------------------
import tensorflow as tf

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.show()

# Each layer is an entry in the list
#Dense layer: each predictor is connected to all (32) this-layer nodes, activation is its function
def train_model(X_train, y_train, num_nodes, dropout_probability, learning_rate, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes,activation='relu', input_shape=(11,)),
        tf.keras.layers.Dropout(dropout_probability),
        tf.keras.layers.Dense(num_nodes,activation='relu'),
        tf.keras.layers.Dropout(dropout_probability),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy',
                 metrics=['accuracy'])

    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation,y_validation), verbose=0
    )
    return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs = 100
for num_nodes in [16, 32, 64]:
    for dropout_probability in [0, 0.2]:
        for learning_rate in [0.1, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_probability}, learning rate {learning_rate}, batch size {batch_size}")
                model, history = train_model(X_train, y_train, num_nodes, dropout_probability, learning_rate, batch_size, epochs)
                #plot_history(history)
                val_loss = model.evaluate(X_validation, y_validation)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model

y_pred_NN = least_loss_model.predict(X_test)
y_pred_NN = (y_pred_NN > .5).astype(int).reshape(-1,)

print(classification_report(y_test, y_pred_NN))