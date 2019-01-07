#!/usr/bin/env python
# coding: utf-8

# ### imports (ML models)

# In[ ]:


print ('Submitted By')
print ('UBITname      = karanman')
print ('Person Number = 50290755')


# In[1]:


# Importing Packages
import pickle
import gzip
import pandas as pd
import numpy as np 
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import matplotlib.pyplot as plt
import itertools


get_ipython().run_line_magic('matplotlib', 'inline')


# ### read mnist data

# In[2]:


#load mnist dataset
def one_hot(_y):
    '''
    given a list of numbers,
    returns a one hot encoded version of the numbers,
        example:
            one_hot(np.array([0,1,2,3,4,5,6,7,8,9]))
        returns:
            array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    '''
    y = np.zeros((_y.shape[0], 10))
    y[np.arange(_y.shape[0]), _y] = 1
    
    return y

# 
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

X_train_r = np.asarray(training_data[0])
_Y_train = np.asarray(training_data[1])
y_train = one_hot(_Y_train)


X_val_r = np.asarray(validation_data[0])
_y_val = np.asarray(validation_data[1])
y_val = one_hot(_y_val)


X_test_r = np.asarray(test_data[0])
_y_test = np.asarray(test_data[1])
y_test = one_hot(_y_test)


# ### data preprocessing

# In[3]:


X_mnist = np.concatenate([X_train_r, X_val_r, X_test_r])
y_mnist = np.concatenate([y_train, y_val, y_test])


# In[4]:


# apply standard scaler
def standard_scaler(inner_x):
    return StandardScaler().fit_transform(inner_x)

X_scaled_mnist = standard_scaler(X_mnist)


# In[5]:


# separate the train and test sets
X_train_mnist = X_scaled_mnist[:50000]
X_val_mnist = X_scaled_mnist[50000:60000]
X_test_mnist = X_scaled_mnist[60000:]

y_train_mnist = y_mnist[:50000]
y_val_mnist = y_mnist[50000:60000]
y_test_mnist = y_mnist[60000:]

# get the label with the highest predicted prrobability
y_train_mnist = y_train_mnist.argmax(axis=1)
y_val_mnist = y_val_mnist.argmax(axis=1)
y_test_mnist = y_test_mnist.argmax(axis=1)


# In[6]:


# print out the datasets' dimensions
print('X_train_mnist shape : {}'.format(X_train_mnist.shape))
print('X_val_mnist shape : {}'.format(X_val_mnist.shape))
print('X_test_mnist shape : {}'.format(X_test_mnist.shape))


# In[7]:


# USPS dataset

# path to USPS dataset
path_to_data = "./USPSdata/Numerals/"

# resize to MNIST scale function
def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

# resize all images
img_list = os.listdir(path_to_data)
sz = (28,28)
validation_usps = []
validation_usps_label = []
for i in range(10):
    label_data = path_to_data + str(i) + '/'
    img_list = os.listdir(label_data)
    for name in img_list:
        if '.png' in name:
            img = cv2.imread(label_data+name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = resize_and_scale(img, sz, 255)
            validation_usps.append(resized_img.flatten())
            validation_usps_label.append(i)


# In[8]:


# convert to numpy array and even batch sizes
validation_usps = np.array(validation_usps)
validation_usps = validation_usps[:-3]
print(validation_usps.shape)
validation_usps_label= np.array(validation_usps_label)
validation_usps_label = validation_usps_label[:-3]
print(validation_usps_label.shape)
validation_usps_label_one_hot = one_hot(validation_usps_label)


# ### confusion matrix helper function

# In[9]:


# confusion matrix method
def plot_confusion_matrix(cm, classes, cm_path, model='Random Forest',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("{} Normalized confusion matrix".format(model))
    else:
        print('{} Confusion matrix'.format(model))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.figure()


# ### Logistic Regression

# In[10]:


def logistic_regression(dataset):
    X_train, y_train, X_val, y_val, X_test, y_test, X_usps, y_usps = dataset

    def softmax(x):
        e = np.exp(x - np.max(x))  # this prevent overflow
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:  
            return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


    class LogisticRegression(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.W = np.zeros((x.shape[1], y.shape[1]))  # initialize Weights 0
            self.b = np.zeros(y.shape[1])          # initialize bias 0
        
        def fit(self, lr=0.1):

            h = softmax(np.dot(self.x, self.W) + self.b)

            d_y = self.y - h
            
            self.W += lr * np.dot(self.x.T, d_y)
            self.b += lr * np.mean(d_y, axis=0)

        def loss(self):
            h = softmax(np.dot(self.x, self.W) + self.b)
            y = self.y
            
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
        def predict(self, x):
            return softmax(np.dot(x, self.W) + self.b)


    ## hyper-parameters
    ### learning rates
    lrs = [0.01, 0.001]
    ### number of iterations
    n_iters = [5, 20, 50, 100, 250]
    
    ## limited grid of hyper-parameters
    ### learning rates
    lrs = [0.01, 0.001]
    ### number of iterations
    n_iters = [5,20,50,100,250]


    def train_logr_model(lr, n_iter, usps=False):


        print('\n--Hyperparameters--')
        print('learning rate : {}'.format(lr))
        print('no of iterations : {}'.format(n_iter))
        
        # standardize by dividing with 255
        model = LogisticRegression(X_train/255, y_train)
        for i in range(n_iter):
            model.fit(lr=lr)

        print('loss : {}'.format(model.loss()))
        
        train_preds = model.predict(X_train)
        acc_train = accuracy_score(y_train, (train_preds>0.5).astype(int))
        
        val_preds = model.predict(X_val)
        acc_val = accuracy_score(y_val, (val_preds>0.5).astype(int))
        
        test_preds = model.predict(X_test)
        acc_test = accuracy_score(y_test, (test_preds>0.5).astype(int))
        
        print('training accuracy : {}'.format(acc_train))
        print('validation accuracy : {}'.format(acc_val))
        print('test accuracy : {}'.format(acc_test))
        if usps:
            usps_preds = model.predict(X_usps)
            acc_usps = accuracy_score(y_usps, (usps_preds>0.5).astype(int))
            return (acc_train, acc_val, acc_test, acc_usps), (test_preds, usps_preds)
        
        return (acc_train, acc_val, acc_test), val_preds


    scores_df = pd.DataFrame()
    for lr in lrs:
        for n_iter in n_iters:
            (acc_train, acc_val, acc_test), val_preds = train_logr_model(lr, n_iter)
            scores = pd.DataFrame([acc_val], index=['acc'], columns=[(lr, n_iter)]).T
            scores_df = pd.concat([scores_df, scores])

    best_params = scores_df['acc'].idxmax()
    lr = best_params[0]
    n_iter = best_params[1]

    print('\n--Training with best hyper-parameters--\n')

    (acc_train, acc_val, acc_test, acc_usps), (test_preds, usps_preds) = train_logr_model(lr, n_iter, usps=True)

    print('training accuracy : {}'.format(acc_train))
    print('validation accuracy : {}'.format(acc_val))
    print('test accuracy : {}'.format(acc_test))
    print('usps accuracy : {}'.format(acc_usps))
    
    return test_preds, usps_preds
    


# In[11]:


dataset = (X_train_r, y_train, X_val_r, y_val, X_test_r, y_test, validation_usps, validation_usps_label_one_hot)
y_preds_logr_mnist, y_preds_logr_usps = logistic_regression(dataset)


# ## convert one-hot encoded values to numbers

# In[12]:


print(y_preds_logr_mnist.shape)
y_preds_logr_mnist = y_preds_logr_mnist.argmax(axis=1)
print(y_preds_logr_mnist.shape)


# In[13]:


print(y_preds_logr_usps.shape)
y_preds_logr_usps = y_preds_logr_usps.argmax(axis=1)
print(y_preds_logr_usps.shape)


# In[14]:


# get unique labels
classes = np.unique(y_test_mnist)

# compute confusion matrix for mnist
cm_mnist = confusion_matrix(y_test_mnist, y_preds_logr_mnist, labels=classes)

# compute confusion matrix for usps
cm_usps = confusion_matrix(validation_usps_label, y_preds_logr_usps, labels=classes)


# In[15]:


# plot mnist confusion matrix
plot_confusion_matrix(cm_mnist, classes, model ='Logistic Regression', cm_path=None)


# In[16]:


# plot usps confusion matrix
plot_confusion_matrix(cm_usps, classes, model ='Logistic Regression', cm_path=None)


# ### Random Forest 

# In[17]:


# grid of hyperparameters
ranf_params = {'n_estimators' :[10, 50, 100],
'max_features' :['auto', 'sqrt', 'log2'],
'max_depth' :[6, 7, 8, None],
'n_jobs' : [-1]}                    


# In[18]:


# # limited grid of hyperparameters
# ranf_params = {'n_estimators' :[10, 50],
# 'max_features' :['auto'],
# 'max_depth' :[None],
# 'n_jobs' : [-1]} 


# In[19]:


# model hyperparameters
ranf_kwarg = {'n_estimators' :10,
'max_features' :'auto',
'max_depth' :None,
'n_jobs' : -1}   


# In[20]:


# predefined validation data to pass to grid search for hyperparameter tuning
ps = PredefinedSplit(test_fold=y_val_mnist)


# In[21]:


# initialize random forest model
ranf = RandomForestClassifier()


# In[22]:


# initialize grid search on the grid of hyperparameters
gsearch1 = GridSearchCV(estimator=ranf,
                        param_grid=ranf_params,
                        scoring='accuracy', n_jobs=-1, cv=ps, verbose=1)
#
# fit the grid search using the training data
gsearch1.fit(X_train_mnist, y_train_mnist)


# In[23]:


# results of hyperparameter tuning on validation data (cross validation)
cv_results = gsearch1.cv_results_
scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')[['params', 'mean_test_score']]
print('grid search results : \n {}'.format(scores_df))


# In[24]:


# set the model kwargs to the grid-searched optimum
print('Best model kwargs: %s', gsearch1.best_params_)
for kwarg in gsearch1.best_params_:
    ranf_kwarg[kwarg] = gsearch1.best_params_[kwarg]

    
ranf = eval(str(ranf))

# set model parameters to grid-searched optimum
ranf.set_params(**ranf_kwarg)


# In[25]:


# train random forest
ranf.fit(X_train_mnist, y_train_mnist)


# In[26]:


# predict on mnist test set
y_preds_ranf_mnist = ranf.predict(X_test_mnist)

# compute accuracy on mnist test set
acc_mnist = accuracy_score(y_preds_ranf_mnist, y_test_mnist)

# predict on usps test set
y_preds_ranf_usps = ranf.predict(validation_usps)

# compute accuracy on usps test set
acc_usps = accuracy_score(validation_usps_label, y_preds_ranf_usps)

# print out test scores
print('Accuracy on MNIST test set : {}'.format(acc_mnist))

print('Accuracy on USPS dataset : {}'.format(acc_usps))


# In[27]:


# get unique labels
classes = np.unique(y_test_mnist)

# compute confusion matrix for mnist
cm_mnist = confusion_matrix(y_test_mnist, y_preds_ranf_mnist, labels=classes)

# compute confusion matrix for usps
cm_usps = confusion_matrix(validation_usps_label, y_preds_ranf_usps, labels=classes)


# In[28]:


# plot mnist confusion matrix
plot_confusion_matrix(cm_mnist, classes, model='Random Forest', cm_path=None)


# In[29]:


# plot usps confusion matrix
plot_confusion_matrix(cm_usps, classes, model='Random Forest', cm_path=None)


# ### SVC PCA

# In[30]:


# no of PCA componets
n_components = 16


# initialize PCA for mnist
pca_mnist = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_mnist)

# initialize PCA for usps
pca_usps = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(validation_usps)



# pca on mnist
X_mnist_pca = pca_mnist.transform(X_mnist)

# pca on usps
X_usps_pca = pca_usps.transform(validation_usps)

# separate the train and test sets
X_train_mnist_pca = X_mnist_pca[:50000]
X_val_mnist_pca = X_mnist_pca[50000:60000]
X_test_mnist_pca = X_mnist_pca[60000:]


# In[31]:


# mnist explained variance
plt.hist(pca_mnist.explained_variance_ratio_, bins=n_components, log=True)
pca_mnist.explained_variance_ratio_.sum()


# In[32]:


# usps explained variance
plt.hist(pca_usps.explained_variance_ratio_, bins=n_components, log=True)
pca_usps.explained_variance_ratio_.sum()


# In[33]:


# grid of hyperparameters
svc_params = {'kernel' : ['rbf', 'linear', 'poly'],
'C' : [1e0, 1e1, 1e2, 1e3],
'gamma' : [1, 10, 100]
               }


# In[34]:


# # limited grid of hyperparameters
# svc_params = {'kernel' : ['linear'],
# 'C' : [1e0, 1e1],
# 'gamma' : [1]
#                }


# In[35]:


# svc hyperparameters
svc_kwarg = {'kernel' : 'linear',
'C' : 1e0,
'gamma' : 1
               }


# In[36]:


# initialize support vector machine classifier
svc = SVC()


# In[37]:


# predefined validation data to pass to grid search for hyperparameter tuning
ps = PredefinedSplit(test_fold=y_val_mnist)


# In[38]:


# initialize grid search
gsearch1 = GridSearchCV(estimator=svc,
                        param_grid=svc_params,
                        scoring='accuracy', n_jobs=-1, cv=ps, verbose=1)
#
# fit the grid search using the training data
gsearch1.fit(X_train_mnist_pca, y_train_mnist)


# In[39]:


# get results and display
cv_results = gsearch1.cv_results_
scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')[['params', 'mean_test_score']]
print('grid search results : \n {}'.format(scores_df))


# In[40]:


# set the model kwargs to the grid-searched optimum
print('Best model kwargs: %s', gsearch1.best_params_)
for kwarg in gsearch1.best_params_:
    svc_kwarg[kwarg] = gsearch1.best_params_[kwarg]
    
svc = eval(str(svc))

# set model parameters to grid-searched optimum
svc.set_params(**svc_kwarg)


# In[41]:


# train SVC
svc.fit(X_train_mnist_pca, y_train_mnist)


# In[42]:


# predict on mnist
y_preds_svc_mnist = svc.predict(X_test_mnist_pca)

# predict on usps
y_preds_svc_usps = svc.predict(X_usps_pca)

# compute accuracy for mnist
acc_mnist = svc.score(X_test_mnist_pca, y_test_mnist)

# compute accuracy for usps
acc_usps = accuracy_score(validation_usps_label, y_preds_svc_usps)


# In[43]:


# display results
print('Accuracy on MNIST test set : {}'.format(acc_mnist))

print('Accuracy on USPS test set : {}'.format(acc_usps))   


# In[44]:


# get the labels
classes = np.unique(y_test_mnist)

# compute mnist confusion matrix
cm_mnist = confusion_matrix(y_test_mnist, y_preds_svc_mnist, labels=classes)

# compute usps confusion matrix
cm_usps = confusion_matrix(validation_usps_label, y_preds_svc_usps, labels=classes)


# In[45]:


# plot mnist confusion matrix
plot_confusion_matrix(cm_mnist, classes, model='Support Vector Classifier', cm_path=None)


# In[46]:


# plot usps confusion matrix
plot_confusion_matrix(cm_usps, classes, model='Support Vector Classifier', cm_path=None)


# ### Neural Network

# #### Imports (Neural Network)

# In[47]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:



# initialize transform (converting to pytorch tensor)
transform = transforms.ToTensor()

# initialize hyperparameter grids
lrs_grid = [1e-6, 1e-4, 1e-3]
neurons_grid = [16, 32, 48, 64, 128, 256]

# # limited grid of hyperparams for testing
# lrs_grid = [1e-6, 1e-4]
# neurons_grid = [16]

# batch size
BATCH_SIZE = 4

# initialize pytorch datasets
trainset_mnist = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

validation_mnist = X_val_mnist.reshape(-1, 1, 28, 28)
valset_mnist = torch.utils.data.TensorDataset(torch.from_numpy(validation_mnist), torch.Tensor(y_val_mnist))
valloader_mnist = torch.utils.data.DataLoader(valset_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

testset_mnist = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

validation_usps = validation_usps.reshape(-1, 1, 28, 28)
testset_usps = torch.utils.data.TensorDataset(torch.from_numpy(validation_usps), torch.Tensor(validation_usps_label))
testloader_usps = torch.utils.data.DataLoader(testset_usps, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# visualize mnist batch
def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

dataiter = iter(trainloader_mnist)
images, labels = dataiter.next()

print('Labels: ', labels)
print('Batch shape: ', images.size())
show_batch(images)


# define the neural network
class SequentialMNIST(nn.Module):
    def __init__(self, no_neurons):
        super(SequentialMNIST, self).__init__()
        self.linear1 = nn.Linear(28*28, no_neurons)
        self.linear2 = nn.Linear(no_neurons, 10)

    def forward(self, x):
        h_relu = F.relu(self.linear1(x.view(BATCH_SIZE, -1)))
        y_pred = self.linear2(h_relu)
        return y_pred

# define the train function for mnist
def train_mnist(model, trainloader_mnist, criterion, optimizer, n_epochs=2):
    for t in range(n_epochs):
        for i, data in enumerate(trainloader_mnist):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # TODO: why?
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels) # Compute the loss
            loss.backward() # Compute the gradient for each variable
            optimizer.step() # Update the weights according to the computed gradient

            if not i % 2000:
                print(t, i, loss.data[0])
    
    return model

# define the test function for mnist
def val_mnist(model, val_mnist):
    outGT = torch.FloatTensor()
    outPRED = torch.FloatTensor()
    
    for i, data in enumerate(valloader_mnist):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        outGT = torch.cat((outGT, labels), 0)

        outPRED = torch.cat((outPRED, outputs), 0)
        
    outPRED = outPRED.data.max(dim=1)[1]
        
    return outPRED.numpy()

# define the test function for mnist
def test_mnist(model, testloader_mnist):
    outGT = torch.LongTensor()
    outPRED = torch.FloatTensor()
    
    for i, data in enumerate(testloader_mnist):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        outGT = torch.cat((outGT, labels), 0)

        outPRED = torch.cat((outPRED, outputs), 0)
        
    outPRED = outPRED.data.max(dim=1)[1]
        
    return outPRED.numpy()

# define the test function for usps
def test_usps(model, testloader_usps):
    outGT = torch.FloatTensor()
    outPRED = torch.FloatTensor()
    
    for i, data in enumerate(testloader_usps):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        outGT = torch.cat((outGT, labels), 0)

        outPRED = torch.cat((outPRED, outputs), 0)
        
    outPRED = outPRED.data.max(dim=1)[1]
        
    return outPRED.numpy()


# In[49]:


# loss function
criterion = nn.CrossEntropyLoss()

# initialize scores dataframe
scores_df = pd.DataFrame()

# loop throuth hyperparameters
for neurons in neurons_grid:
    for lr in lrs_grid:
        
        # initialize model
        model = SequentialMNIST(neurons)
        
        # initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # train model
        model = train_mnist(model, trainloader_mnist, criterion, optimizer, n_epochs=1)
        
        # predict on the mnist validation set
        y_preds_nn_mnist = val_mnist(model, valloader_mnist)
        
        # predict on the usps validation set
        y_preds_nn_usps = test_usps(model, testloader_usps)
        
        # compute accuracy
        acc_mnist = accuracy_score(y_val_mnist, y_preds_nn_mnist)

        acc_usps = accuracy_score(validation_usps_label, y_preds_nn_usps)
        
        # load scores into dataframe
        scores = pd.DataFrame([acc_mnist, acc_usps], index=['acc_mnist', 'acc_usps'], columns=[(neurons, lr)]).T
        scores_df = pd.concat([scores_df, scores])
        

        


# In[50]:


# loop through all scores
scores_df_str = scores_df.copy(deep=True)
scores_df_str.index = scores_df_str.index.map(str)
for i in range(scores_df.shape[0]):
    params = scores_df.index.values[i]
    neurons = params[0]
    lr = params[1]
    
    params_str = scores_df_str.index.values[i]
    acc_mnist= scores_df_str.loc[params_str, 'acc_mnist']
    acc_usps= scores_df_str.loc[params_str, 'acc_usps']
    print('Accuracy with neurons : {} and learning rate {} \n mnist : {} : usps : {}'.format(neurons, lr, acc_mnist, acc_usps))


# In[51]:


best_params = scores_df['acc_mnist'].idxmax()

neurons = best_params[0]
lr = best_params[1]

print('\n--Training with best hyper-parameters--\n')

# initialize neural network
model = SequentialMNIST(neurons)

# initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# train the model
model = train_mnist(model, trainloader_mnist, criterion, optimizer, n_epochs=1)


# In[52]:


# mnist test set predictions
y_preds_nn_mnist = test_mnist(model, testloader_mnist)

# mnist test set accuracy
acc_mnist = accuracy_score(y_test_mnist, y_preds_nn_mnist)

# usps test set predictions
y_preds_nn_usps = test_usps(model, testloader_usps)

# usps test set accuracy
acc_usps = accuracy_score(validation_usps_label, y_preds_nn_usps)


# In[53]:


# print out results
print('Accuracy on test set : {}'.format(acc_mnist))

print('Accuracy on USPS dataset : {}'.format(acc_usps))


# In[54]:


# get the labels
classes = np.unique(y_test_mnist)

# mnist confusion matrix
cm_mnist = confusion_matrix(y_test_mnist, y_preds_svc_mnist, labels=classes)

# usps confusion matrix
cm_usps = confusion_matrix(validation_usps_label, y_preds_nn_usps, labels=classes)


# In[55]:


# plot mnist confusion matrix
plot_confusion_matrix(cm_mnist, classes, model='Neural Network', cm_path=None)


# In[56]:


# plot usps confusion matrix
plot_confusion_matrix(cm_usps, classes, model='Neural Network', cm_path=None)


# ### Ensemble classifier (majority voting)

# In[57]:


# merge labels and all model predictions for mnist
y_pred_mnist = pd.DataFrame({'label' : y_test_mnist})

y_pred_mnist['ranf_mnist'] = y_preds_ranf_mnist.astype(int)
y_pred_mnist['svc_mnist'] = y_preds_svc_mnist.astype(int)
### For logistic regression
y_pred_mnist['logr'] = y_preds_logr_mnist.astype(int)
y_pred_mnist['nn_mnist'] = y_preds_nn_mnist.astype(int)

y_pred_mnist['prediction_mnist'] = 0
y_pred_mnist['prediction_mnist'] = y_pred_mnist.iloc[:, 1:3].mode(axis=1)
y_pred_mnist['prediction_mnist'] = y_pred_mnist['prediction_mnist'].astype(int)


# In[58]:


# merge labels and all model predictions for usps
y_pred_usps = pd.DataFrame({'label_usps' : validation_usps_label})

y_pred_usps['ranf_usps'] = y_preds_ranf_usps.astype(int)
y_pred_usps['svc_usps'] = y_preds_svc_usps.astype(int)
### For logistic regression
y_pred_usps['logr_usps'] = y_preds_logr_usps.astype(int)
y_pred_usps['nn_usps'] = y_preds_nn_usps.astype(int)

y_pred_usps['prediction_usps'] = 0
y_pred_usps['prediction_usps'] = y_pred_usps.iloc[:, 1:3].mode(axis=1)
y_pred_usps['prediction_usps'] = y_pred_usps['prediction_usps'].astype(int)


# In[59]:


# mnist accuracy
acc_mnist = accuracy_score(y_test_mnist, y_pred_mnist['prediction_mnist'])

print('Accuracy of ensemble on MNIST test set : {}'.format(acc_mnist))


# In[60]:


# usps accuracy
acc_usps = accuracy_score(validation_usps_label, y_pred_usps['prediction_usps'])

print('Accuracy of ensemble on USPS dataset : {}'.format(acc_usps))


# In[61]:


# get the labels
classes = np.unique(y_test_mnist)

# mnist confusion matrix
cm_mnist = confusion_matrix(y_test_mnist, y_pred_mnist['prediction_mnist'], labels=classes)


# In[62]:


# plot mnist confusion matrix
plot_confusion_matrix(cm_mnist, classes, model='Ensemble Classifier', cm_path=None)


# In[63]:


# usps confusion matrix
cm_usps = confusion_matrix(validation_usps_label, y_pred_usps['prediction_usps'], labels=classes)


# In[64]:


# plot usps confusion matrix
plot_confusion_matrix(cm_usps, classes, model='Ensemble Classifier', cm_path=None)


# In[ ]:




