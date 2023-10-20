#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

#loading MNIST dataset
from tensorflow.keras.datasets import mnist
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

#visualizing the image in train data
plt.imshow(X_train[0])

#visualizing the first 20 images in the dataset
for i in range(20):
     #subplot
    plt.subplot(5, 5, i+1)
    # plotting pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

print(X_train.shape)
print(X_test.shape)

X_train_flat=X_train.reshape(len(X_train),28*28)
X_test_flat=X_test.reshape(len(X_test),28*28)
#checking the shape after flattening
print(X_train_flat.shape)
print(X_test_flat.shape)

#normalizing the pixel values
X_train_flat=X_train_flat/255
X_test_flat=X_test_flat/255

#importing necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(10,input_shape=(784,),activation='softmax')
)

model.compile(loss='sparse_categorical_crossentropy',optimizer ='adam',metrics=['accuracy'])

model.fit(X_train_flat,y_train,epochs=10)
model.evaluate(X_test_flat,y_test)

y_predict = model.predict(X_test_flat)
y_predict[3] #printing the 3rd index

np.argmax(y_predict[3])
plt.imshow(X_test[3])
y_predict_labels=np.argmax(y_predict,axis=1)
#Confusion matrix
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_predict_labels)
#visualizaing confusion matrix with heatmap
plt.figure(figsize=(10,7))
sns.heatmap(matrix,annot=True,fmt='d')
plt.show()

'''
model2=Sequential()
#adding first layer with 100 neurons
model2.add(Dense(100,input_shape=(784,),activation='relu'))
#second layer with 64 neurons
model2.add(Dense(64,activation='relu'))
#third layer with 32 neurons
model2.add(Dense(32,activation='relu'))
#output layer
model2.add(Dense(10,activation='softmax'))
#compliling the model
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model2.fit(X_train_flat,y_train,epochs=10)
#evaluating the model
model2.evaluate(X_test_flat,y_test)'''