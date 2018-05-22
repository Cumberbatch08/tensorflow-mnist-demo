
# coding: utf-8

# ### 载入数据

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# ### 占位符

# In[3]:


#x = mnist.train.images
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, shape = [-1, 28,28, 1])
y = tf.placeholder(tf.float32, [None, 10])


# ### 权重初始化

# In[4]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ### 卷积  池化层初始化

# In[5]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# ### 第一卷积层

# In[7]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


# ### 第一池化层

# In[8]:


h_pool1 = max_pool_2x2(h_conv1)


# ### 第二卷基层

# In[9]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)


# ### 第二池化层

# In[10]:


h_pool2 = max_pool_2x2(h_conv2)


# ### 全连接层

# In[11]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# ### dropout

# In[12]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# ### 输出层

# In[13]:


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# ### 训练与评估

# In[14]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[15]:


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[16]:


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


# In[ ]:


for i in range(10000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuacy = accuracy.eval(feed_dict={x: batch[0], 
                                                 y: batch[1],
                                                 keep_prob: 1.0})
        ##t.eval() is a shortcut for calling tf.get_default_session().run(t)
        print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict = {x: batch[0], 
                                y: batch[1], 
                                keep_prob: 0.5})

# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={x: mnist.test.images,
                                                   y: mnist.test.labels,
                                                   keep_prob: 1.0})))

