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
    plt.ylabel = "probability"
    plt.xlabel = label
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
# The data can be plotted in a (high-dimensional) plot, which a label for which group a data point belongs to
# For any new datapoint, its group can be predicted by checking which groups its nearest neighbouring points belong to

# --> Especially useful when classes generally group together, 
#       but (due to high-dimensionality) the factors determining group belonging are unclear
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
# 

# ---------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
#naive bayes assumes that all predictors are independent (non-correlated)

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_predNB = nb_model.predict(X_test)

print(classification_report(y_test, y_predNB))

# =========================================================
# 4. Machine classification: Logistic regression

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_predLG = lg_model.predict(X_test)

print(classification_report(y_test, y_predLG))

# =========================================================
# 5. Machine classification: Support vector machines
from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_predSVM = svm_model.predict(X_test)

print(classification_report(y_test, y_predSVM))

