import tensorflow as tf
import numpy as np

tf.set_random_seed(888)  # for reproducibility

# Predicting animal type based on various features

xy = np.loadtxt(r'C:\Users\user\Documents\data10.csv', delimiter=',', dtype= np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)


nb_classes = 2  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 20])

Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot

print("one_hot:", Y_one_hot)

Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

print("reshape one_hot:", Y_one_hot)



W = tf.Variable(tf.random_normal([20, nb_classes]), name='weight')

b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations

# softmax = exp(logits) / reduce_sum(exp(logits), dim)

logits = tf.matmul(X, W) + b

hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,

                                                                 labels=tf.stop_gradient([Y_one_hot])))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9).minimize(cost)

prediction = tf.argmax(hypothesis, 1)

correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):

        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict

    pred = sess.run(prediction, feed_dict={X: x_data})

    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    TP=1 #실제참 예측 참
    FN=1
    TN=1 #실제거짓 예측거짓
    FP= 1 #실제거짓 예측 참
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
        if(y==1):  #참일때
            if(int(p)==y):
                #print("real cancer")
                #print(x_data);
                TP=TP+1
            if(int(p)!=y):
                print("not cancer")
                print(x_data,  y_data);
                FN=FN+1
        if(y==0): #거짓일때
            if(int(p)==y):
                TN=TN+1
            if(int(p)!=y):
                FP=FP+1
    print("Accuracy:", (TP+TN)/(TP+TN+FP+FN)*100)
    print("Sensitivy:", TP/(TP+FN)*100)
    print("Specificity(특이도):",TN/(TN+FP)*100)
    print("Precsion(정밀도):",TP/(TP+FP)*100)



