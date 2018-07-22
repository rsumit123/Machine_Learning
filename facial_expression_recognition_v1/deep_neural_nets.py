import time
from code1 import *
import pickle
import math
import random
from code import *
import matplotlib
#matplotlib.use('agg',warn=False, force=True)
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from deep_neural_nets_utilities import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



np.random.seed(1)





num_pixels=100
np.random.seed(7)
print('loading data ...')     

#loading data 
orig='/home/sumit/Downloads/train_more_images/'
x_data= create_database('cohn-kanade-images/')
x_data1=np.concatenate((create_database1('/home/sumit/Downloads/train_more_images/downloads/happy human faces',5)[0],create_database1(orig+'downloads/sad human faces',6)[0],create_database1(orig+'downloads/angry adult faces',1)[0],create_database1(orig+'downloads/surprised human faces',7)[0],create_database1(orig+'downloads/disgusted adult faces',3)[0],create_database1(orig+'downloads/neutral faces',2)[0],create_database1(orig+'downloads/scared adult faces',4)[0]))

x_data=np.concatenate((x_data,x_data1))
print(x_data.shape)
print(x_data1.shape)

x_data=x_data.reshape(x_data.shape[0],num_pixels,num_pixels,1)
y_data= create_database('Emotion/')

y_data1=np.concatenate((create_database1('/home/sumit/Downloads/train_more_images/downloads/happy human faces',5)[1],create_database1(orig+'downloads/sad human faces',6)[1],create_database1(orig+'downloads/angry adult faces',1)[1],create_database1(orig+'downloads/surprised human faces',7)[1],create_database1(orig+'downloads/disgusted adult faces',3)[1],create_database1(orig+'downloads/neutral faces',    2)[1],create_database1(orig+'downloads/scared adult faces',4)[1]))

y_data=np.concatenate((y_data,y_data1))
print(y_data.shape)
print(y_data1.shape)
#shuffling data

permutation = np.random.permutation(x_data.shape[0])
#print(permutation)
shuffled_x = x_data[permutation] #shuffle x
shuffled_y= y_data[permutation] #shuffle y

#print(shuffled_x.shape)
#print(shuffled_y.shape)

y_data=y_data.reshape(1,y_data.shape[0]) #reshaping acc to below code
shuffled_y=shuffled_y.reshape(1,y_data.shape[1])
print('Data loaded......')


#train x
train_x_orig=x_data[0:2900,:,:]
shuffled_train_x=shuffled_x[0:2900,:,:]
#print('train x shape',train_x_orig.shape)
#print('shuffled train x shape',shuffled_train_x.shape)

#train y

train_y=y_data[:,0:2900]
shuffled_train_y=shuffled_y[:,0:2900]
#print('train y shape',train_y.shape)
#print('shuffled train y shape',shuffled_train_y.shape)
#print('train_y unique',np.unique(train_y))
#print(' shuffled train_y unique',np.unique(shuffled_train_y))

#test x
test_x_orig=x_data[2900:,:,:]
shuffled_test_x=shuffled_x[2900:,:,:]

#test y
test_y=y_data[:,2900:]
shuffled_test_y=shuffled_y[:,2900:]
#print('test_y shape',test_y.shape)
#print('shuffled_test_y',shuffled_test_y.shape)
#print('test_y unique',np.unique(test_y))
#print('shuffled test_y unique',np.unique(shuffled_test_y))

#equating train_x_orig to shuffled values resp.
train_x_orig=shuffled_train_x
train_y=shuffled_train_y
test_x_orig=shuffled_test_x
test_y=shuffled_test_y


#for showing image
train_x_orig1=train_x_orig.reshape(2900,num_pixels,num_pixels)
index = 1400
plt.imshow(train_x_orig1[index]) #7 surprise,#3 disgust,#5 happy,#1 anger,#6 sadness,#4 fear?,#2 neutral
plt.show()
print(train_y[0,index])
classes=[b'anger',b'neutral',b'disgust',b'fear',b'happy',b'sad',b'surprise']
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]-1].decode("utf-8") +  " picture.")



# About dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 1)")
#print ("train_x_orig shape: " + str(train_x_orig.shape))
#print ("train_y shape: " + str(train_y.shape))
#print ("test_x_orig shape: " + str(test_x_orig.shape))
#print ("test_y shape: " + str(test_y.shape))





# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
train_y=train_y/8. #since there are multiple classes
test_y=test_y/8.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
print('training on '+str(train_x.shape[1]))
print('testing on '+str(test_x.shape[1]))





# CONSTANTS DEFINING THE MODEL 
n_x = 10000    # num_px * num_px * 1
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)




# two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
   
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation
        A1, cache1 =  linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 =  linear_activation_forward(A1, W2, b2, 'sigmoid')
        
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. 
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        
        #Feeding the gradients into a grads dictionary
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            print(cost)
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters




#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 1000, print_cost=True)




#predictions_train = predict(train_x, train_y, parameters)




#predictions_test = predict(test_x, test_y, parameters)




### CONSTANTS ###
layers_dims = [10000,20, 30, 40, 35, 20, 35, 50, 70,1] #  7-layer model

def initialize_adam(parameters) :
    
    L = len(parameters) // 2 # number of layers in the neural network
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
    
    
    return v, s
    
    
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. 
        
        v["dW" + str(l+1)] = beta1 * v["dW"+str(l+1)] + (1-beta1) * grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db"+str(l+1)] + (1-beta1) * grads["db"+str(l+1)]
        

        # Compute bias-corrected first moment estimate. 
        
        v_corrected["dW" + str(l+1)] = v["dW"+str(l+1)]/(1-(beta1**t))
        v_corrected["db" + str(l+1)] = v["db"+str(l+1)]/(1-(beta1**t))
        

        # Moving average of the squared gradients.
        
        s["dW" + str(l+1)] = beta2 * s["dW"+str(l+1)] + (1-beta2) * (grads["dW"+str(l+1)]**2)
        s["db" + str(l+1)] = beta2 * s["db"+str(l+1)] + (1-beta2) * (grads["db"+str(l+1)]**2)
        

        # Compute bias-corrected second raw moment estimate. 
        
        s_corrected["dW" + str(l+1)] =  s["dW"+str(l+1)]/(1-(beta2**t))
        s_corrected["db" + str(l+1)] = s["db"+str(l+1)]/(1-(beta2**t))
        

        # Update parameters. 
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+epsilon)
        

    return parameters, v, s    
    
def initialize_parameters_he(layers_dims):
    
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*(np.sqrt(2/layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        
        
    return parameters


# mini _ batches
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    
    np.random.seed(seed)            # To make the "random" minibatches the same always
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        
        mini_batch_X = shuffled_X[:,-1*(m% mini_batch_size): ]
        mini_batch_Y = shuffled_Y[:,-1*(m% mini_batch_size): ]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.002, mini_batch_size = 64,beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000,print_cost=False):#lr was 0.009
    

    np.random.seed(1)
    L=len(layers_dims)
    seed=10
    t=0
    costs = []                         # keep track of cost
    
    # Parameters initialization
    
    parameters = initialize_parameters_he(layers_dims)
    #with open('filename3.pickle', 'rb') as handle:
        #parameters = pickle.load(handle)
    #v, s = initialize_adam(parameters)
    
    st=time.time()
    
    # Loop (gradient descent)
    for i in range(num_epochs):
        
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            (minibatch_X,minibatch_Y)=minibatch

        # Forward propagation
        
            AL, caches = L_model_forward(minibatch_X, parameters)
        
        
        # Compute cost.
        
            cost = compute_cost(AL, minibatch_Y)
        
    
        # Backward propagation.
        
            grads = L_model_backward(AL, minibatch_Y, caches)
        
 
        # Update parameters.
        
            parameters = update_parameters(parameters, grads, learning_rate)
            t = t + 1 # Adam counter
           # parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,t, learning_rate, beta1, beta2,  epsilon)
        
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
            print("                                                              Time passed = ",time.time()-st)
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs (per hundred)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



#parameters = L_layer_model(train_x, train_y, layers_dims,learning_rate=0.00555, num_epochs = 1500, print_cost = True)
#with open('9_layer_nn_w_added_images(1500iter).pickle', 'wb') as handle:
    #pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('9_layer_nn_w_added_images(1500iter).pickle', 'rb') as handle:
    parameters = pickle.load(handle)


pred_train = predict(train_x, train_y, parameters)


print('*********************************************************')

pred_test = predict(test_x, test_y, parameters)


#print_mislabeled_images(classes, test_x_orig, test_y, pred_test)

def testing_ur_own_dataset(files_dir):
    my_label_y = [0.625,0.25,0.75,0.875,0.625,0.625,0.75,0.875]#7 surprise,#3 disgust,#5 happy,#1 anger,#6 sadness,#4 fear?,#2 neutral
    image_data=[]
    for files in sorted(os.listdir(files_dir)):
        #my_image = "IMG_20180722_094407227.jpg" # change this to the name of your image file 
         


        #fname = "/home/sumit/Downloads/" + my_image
        #print(files)
        image=Image.open(files_dir+'/'+files).convert('L')
        plt.imshow(image)
        #plt.show()
        im=np.array(image.resize((100,100)))
    #image = np.array(ndimage.imread(fname, flatten=False))
        #my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*1,1))
        #my_image = my_image/255.
        image_data.append(im)
    image_data=np.array(image_data)    
    image_data=image_data.reshape(image_data.shape[0],num_pixels,num_pixels,1)
    image_data=image_data.reshape(image_data.shape[0], -1).T #flatten
    image_data=image_data/255.
    
    
    print(image_data.shape)
    my_predicted_image = predict1(image_data, my_label_y, parameters)
     
    print('predictions',my_predicted_image)
    my_predicted_image=np.squeeze(my_predicted_image)
    
    for i in range(image_data.shape[1]):
        print ("y = " + str(my_predicted_image[i]) + ", your L-layer model predicts a \"" +             classes[int(my_predicted_image[i]*8)-1].decode("utf-8") +  "\" picture.")

testing_ur_own_dataset('/home/sumit/Downloads/my_own_images')
