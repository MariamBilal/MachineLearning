import tensorflow as tf

"""
input layer1 > hidden layer 1(activation functions)>weights>
hiddden layer2(activation function)>weights>output layer.
"""
"""
we passes the data feedforward and compare the output 
to the intended output
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot = True)

"""
Here we have 3 hidden layers --> deep neural network.
Each layer have 500 nodes in it.You can increase or decrease the number 
of nodes according to your choice.
"""
n_nodes_hl1=200
n_nodes_hl2=200
n_nodes_hl3=200
 
# number of classes
n_classes = 5
batch_size=100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

"""
here we are passing our data to the neural network layers.
"""
def neural_network_model(data):
    
    hidden_1_layer ={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer ={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer ={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer ={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}
    #   Jo normal ap k pass data araha hai. direct jo pass kiya ja krha h data function may.
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])
    output = tf.nn.relu(l3)
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    
    #AdamOptimizer have a perameter which have learning rate of about 0.001
    optimizer= tf.train.AdamOptimizer().minimize(cost)
    hm_epochs =10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict = {x:epoch_x ,y:epoch_y})
                epoch_loss +=c
            print('Epoch',epoch,'completed out of',hm_epochs ,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy =tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
                
train_neural_network(x)         