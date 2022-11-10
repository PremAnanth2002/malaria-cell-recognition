# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.
Dataset:CellImage

## Neural Network Model

![image](https://user-images.githubusercontent.com/69795479/194772218-6d3bd5db-3516-4776-bbb9-160a2642d1fe.png)


## DESIGN STEPS

### STEP 1:
Import necessary packages

### STEP 2:
Preprocess the image using data augmentation

### STEP 3:
Fit the model using the augmented images


## PROGRAM
```python
Name:Akash S
Reg No: 212220040005
python3

import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

trainDatagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
testDatagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train=trainDatagen.flow_from_directory("cellimage/cell_images/train",class_mode = 'binary',target_size=(150,150))
test=trainDatagen.flow_from_directory("cellimage/cell_images/test",class_mode = 'binary',target_size=(150,150))

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,activation="relu",padding="same"),
    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(128,3,activation="relu"),
    tf.keras.layers.Conv2D(128,3,activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="accuracy")

model.fit(train,epochs=5,validation_data=test)

pd.DataFrame(model.history.history).plot()

import numpy as np

test_predictions = np.argmax(model.predict(test), axis=1)

confusion_matrix(test.classes,test_predictions)

print(classification_report(test.classes,test_predictions))

import numpy as np

img = tf.keras.preprocessing.image.load_img("cellimage/cell_images/test/uninfected/C100P61ThinF_IMG_20150918_144104_cell_34.png")
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(150,150))
img_28 = img_28/255.0
img_28=tf.expand_dims(img_28, axis=0)

if tf.cast(tf.round(model.predict(img_28))[0][0],tf.int32).numpy()==1:
  print("uninfected")
else:
  print("parasitized")  
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![exp4-1](https://user-images.githubusercontent.com/75235022/194769338-5eb30587-e47f-46d5-8548-84db93679745.png)


### Classification Report
![classi report](https://user-images.githubusercontent.com/75235022/194769377-c4ebb990-dc42-449c-9fca-9fb70346c1cb.png)


### Confusion Matrix
![confus matric](https://user-images.githubusercontent.com/75235022/194769426-9863316b-05d1-4f26-a1bb-9f4c085b66ce.png)


### New Sample Data Prediction

![input and output](https://user-images.githubusercontent.com/75235022/194769463-56ff3bc8-6fc4-466a-ac66-0bbacd09e224.png)


## RESULT
A deep neural network for Malaria infected cell recognition is built
