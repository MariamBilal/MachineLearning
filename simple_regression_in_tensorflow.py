#importing packages
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt

#Get the data 
features,prices = load_boston(True)
#(true) will let load_bostan know that we want the dataset features and 
#prices in seperate numpy array.so we can store them in two seperate arrays


#splitting the data into training,validation and testing set
train_features = scale(features[:300]) #train dataset, 0-300 values
train_prices = prices[:300]

valid_features =scale(features[300:400])
valid_prices = prices[300:400]

test_features = scale(features[400:]) #testing dataset,400-end values
test_prices= prices[400:]


w = tf.Variable(tf.truncated_normal([13,1],mean=0.0,stddev=1.0,dtype = tf.float64))
b = tf.Variable(tf.zeros(1, dtype = tf.float64))
#tf.Variable() defines a tensor variable.
#tf.zeros will a Tensor with all elements set to zero.


#defining a function to return predictions and errors
def calc(x,y):
    predictions = tf.add(b,tf.matmul(x,w))
    error = tf.reduce_mean(tf.square(y - predictions))
    return[predictions,error]
#tf.add() will add the values.
#tf.matmul() multiplies the matrices.
#tf.squaer() square each element in the tensorflow passed to it.    
   
 
y,cost = calc(train_features,train_prices)
learning_rate=0.05
epochs =10
points = [[],[]]


init=tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(init)
    for i in list(range(epochs)):
        sess.run(optimizer)
        if i% 10 ==0:
            points[0].append(i+1)
            points[1].append(sess.run(cost))
        
        if i%100==0:
            print(sess.run(cost))
    plt.plot(points[0],points[1],'r--')
    plt.axis([0,epochs,50,600])
    plt.show()
    
    valid_cost = calc(valid_features,valid_prices)[1]
    print('Validation error = ',sess.run(valid_cost),'\n')
    test_cost = calc(test_features,test_prices)[1]
    print('test error = ' , sess.run(test_cost),'\n')

