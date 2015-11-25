
# coding: utf-8

# # Project 2: Supervised Learning
# ### Building a Student Intervention System

# ## 1. Classification vs Regression
# 
# Your goal is to identify students who might need early intervention - which type of supervised machine learning problem is this, classification or regression? Why?

# ## 2. Exploring the Data
# 
# Let's go ahead and read in the student dataset first.
# 
# _To execute a code cell, click inside it and press **Shift+Enter**._

# In[1]:

# Import libraries
import numpy as np
import pandas as pd
from sklearn import utils


# In[2]:

# Read student data
student_data = pd.read_csv("student-data.csv")
df = pd.DataFrame(student_data)
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns


# Now, can you find out the following facts about the dataset?
# - Total number of students
# - Number of students who passed
# - Number of students who failed
# - Graduation rate of the class (%age)
# - Number of features
# 
# _Use the code block below to compute these values. Instructions/steps are marked using **TODO**s._

# In[3]:

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
n_students = len(df.index)
n_features = len(df.columns) - 1   # "passed" isn't a feature right => -1 ?
n_passed = len(df[df['passed'].str.contains("yes")])
n_failed = len(df[df['passed'].str.contains("no")])
grad_rate = ((len(df[df['passed'].str.contains("yes")]) * 100) / len(df.index))
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## 3. Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Let's first separate our data into feature and target columns, and see if any features are non-numeric.<br/>
# **Note**: For this dataset, the last column (`'passed'`) is the target or label we are trying to predict.

# In[4]:

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows


# ### Preprocess feature columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation.

# In[5]:

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Split data into training and test sets
# 
# So far, we have converted all _categorical_ features into numeric values. In this next step, we split the data (both features and corresponding labels) into training and test sets.

# In[15]:

# First, decide how many training vs test samples you want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
X, y = utils.shuffle(X_all, y_all)
X_train = X[0:300]
y_train = y[0:300]
X_test = X[300:]
y_test = y[300:]
print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data


# ## 4. Training and Evaluating Models
# Choose 3 supervised learning models that are available in scikit-learn, and appropriate for this problem. For each model:
# 
# - What is the theoretical O(n) time & space complexity in terms of input size?
# - What are the general applications of this model? What are its strengths and weaknesses?
# - Given what you know about the data so far, why did you choose this model to apply?
# - Fit this model to the training data, try to predict labels (for both training and test sets), and measure the F<sub>1</sub> score. Repeat this process with different training set sizes (100, 200, 300), keeping test set constant.
# 
# Produce a table showing training time, prediction time, F<sub>1</sub> score on training set and F<sub>1</sub> score on test set, for each training set size.
# 
# Note: You need to produce 3 such tables - one for each model.

# In[31]:

# Train model 1 = Support Vector Machine
import time

def train_classifier1(clf1, X_train, y_train):
    print "Training {}...".format(clf1.__class__.__name__)
    start = time.time()
    clf1.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn import svm
clf1 = svm.SVC()

# Fit model to training data
train_classifier1(clf1, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it
# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels1(clf1, features, target):
    print "Predicting labels using {}...".format(clf1.__class__.__name__)
    start = time.time()
    y_pred = clf1.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score1 = predict_labels1(clf1, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score1)
# Predict on test data
print "F1 score for test set: {}".format(predict_labels1(clf1, X_test, y_test))


# In[32]:

# Train and predict using different training set sizes
def train_predict1(clf1, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier1(clf1, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels1(clf1, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels1(clf1, X_test, y_test))

train_predict1(clf1, X_train[0:100], y_train[0:100], X_test, y_test)
train_predict1(clf1, X_train[0:200], y_train[0:200], X_test, y_test)
train_predict1(clf1, X_train[0:300], y_train[0:300], X_test, y_test)

# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant


# In[33]:

# Train model 2 = ADAboost
import time

def train_classifier2(clf2, X_train, y_train):
    print "Training {}...".format(clf2.__class__.__name__)
    start = time.time()
    clf2.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.ensemble import AdaBoostClassifier
clf2 = AdaBoostClassifier()

# Fit model to training data
train_classifier2(clf2, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it
# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels2(clf2, features, target):
    print "Predicting labels using {}...".format(clf2.__class__.__name__)
    start = time.time()
    y_pred = clf2.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score2 = predict_labels2(clf2, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score2)
# Predict on test data
print "F1 score for test set: {}".format(predict_labels2(clf2, X_test, y_test))


# In[34]:

# Train and predict using different training set sizes
def train_predict2(clf2, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier2(clf2, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels2(clf2, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels2(clf2, X_test, y_test))

train_predict2(clf2, X_train[0:100], y_train[0:100], X_test, y_test)
train_predict2(clf2, X_train[0:200], y_train[0:200], X_test, y_test)
train_predict2(clf2, X_train[0:300], y_train[0:300], X_test, y_test)

# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant


# In[35]:

# Train model 5 = KNN
import time

def train_classifier5(clf5, X_train, y_train):
    print "Training {}...".format(clf5.__class__.__name__)
    start = time.time()
    clf5.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.neighbors import KNeighborsClassifier
clf5 = KNeighborsClassifier() 

# Fit model to training data
train_classifier5(clf5, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it
# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels5(clf5, features, target):
    print "Predicting labels using {}...".format(clf5.__class__.__name__)
    start = time.time()
    y_pred = clf5.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score5 = predict_labels5(clf5, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score5)
# Predict on test data
print "F1 score for test set: {}".format(predict_labels5(clf5, X_test, y_test))


# In[36]:

# Train and predict using different training set sizes
def train_predict5(clf5, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier5(clf5, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels5(clf5, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels5(clf5, X_test, y_test))

train_predict5(clf5, X_train[0:100], y_train[0:100], X_test, y_test)
train_predict5(clf5, X_train[0:200], y_train[0:200], X_test, y_test)
train_predict5(clf5, X_train[0:300], y_train[0:300], X_test, y_test)

# TODO: Run the helper function above for desired subsets of training data
# Note: Keep the test set constant


# ## 5. Choosing the Best Model
# 
# - Based on the experiments you performed earlier, in 1-2 paragraphs explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?
# - In 1-2 paragraphs explain to the board of supervisors in layman's terms how the final model chosen is supposed to work (for example if you chose a Decision Tree or Support Vector Machine, how does it make a prediction).
# - Fine-tune the model. Use Gridsearch with at least one important parameter tuned and with at least 3 settings. Use the entire training set for this.
# - What is the model's final F<sub>1</sub> score?

# In[13]:

# TODO: Fine-tune your model and report the best F1 score


# In[39]:

# Final Model SVM
import time

def train_classifier_test(clf_test, X_train, y_train):
    print "Training {}...".format(clf_test.__class__.__name__)
    start = time.time()
    clf_test.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn import svm
from sklearn import grid_search
clt = svm.SVC()

kernel       = ['linear','poly','rbf', 'sigmoid'] 
Cs           = np.logspace(-10, 0, 10)
coef0        = np.logspace(-10, 0, 10)
param_grid   = dict(C=Cs,kernel=kernel,coef0=coef0)
clf_test     = grid_search.GridSearchCV(estimator=clt,param_grid=param_grid, n_jobs=-1)

# Fit model to training data
train_classifier_test(clf_test, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it
# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels_test(clf_test, features, target):
    print "Predicting labels using {}...".format(clf_test.__class__.__name__)
    start = time.time()
    y_pred = clf_test.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score_test = predict_labels_test(clf_test, X_train, y_train)
print "F1 score for training set: {}".format(train_f1_score5)
# Predict on test data
print "F1 score for test set without GridSearch): {}".format(predict_labels1(clf1, X_test, y_test))
print "F1 score for final test set with GridSearch: {}".format(predict_labels_test(clf_test, X_test, y_test))
print  "best parameter choice:", clf_test.best_params_
print  "best score:", clf_test.best_score_


# In[ ]:



