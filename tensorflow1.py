
# hello world with tensorflow
import tensorflow as tf
hello = tf.constant("Hello , tensorflow")
sess= tf.Session()
print(sess.run(hello))



# Multiplication operation on two numbers in tensorflow
num1 = tf.constant(5)
num2= tf.constant(6)
result = tf.multiply(num1,num2)
#print(result)
sess = tf.Session()
print(sess.run(result))