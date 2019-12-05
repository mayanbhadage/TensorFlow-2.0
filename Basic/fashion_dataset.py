
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# import dataset

data = keras.datasets.fashion_mnist



print (data)



# split data in test and training data
# 80% to 90% of data for training 

(train_images, train_labels),(test_images, test_labels) = data.load_data()


# In[5]:


# print(train_labels[0]) => 0 to 9 labels


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# shrunk down the images pixel from range(0 to 1)



train_images = train_images/255.0
test_images = test_images/255.0




model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), #input layer
    keras.layers.Dense(128,activation="relu"), #Hidden Layer
    keras.layers.Dense(10,activation="softmax")# Output Layer
])




model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])


model.fit(train_images,train_labels,epochs = 5)



test_loss,test_acc = model.evaluate(test_images,test_labels)


print("TEST ACCURACY : ", test_acc)






