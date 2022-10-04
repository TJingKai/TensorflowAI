# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import tensorflow modules
import tensorflow as tf
from tensorflow.keras import layers

print("TensorFlow version: {}".format(tf.__version__))

# Reads the excel file and loads it
class_name = ["Safe", "Phishing"]
phishing_train = pd.read_csv("dataset_full.csv")
phishing_train.head()

phishing_features = phishing_train.copy()
phishing_labels = phishing_features.pop("phishing")

phishing_features = np.array(phishing_features)

train, val, test = np.split(phishing_train.sample(frac=1), [int(0.8*len(phishing_train)), int(0.9*len(phishing_train))])
print(train.shape, 'training shape')
print(val.shape, 'validation shape')
print(test.shape, 'test shape')

y_train, x_train = train['phishing'], train.drop('phishing', axis=1)
y_val, x_val = val['phishing'], val.drop('phishing', axis=1)
y_test, x_test = test['phishing'], test.drop('phishing', axis=1)
print(y_train.shape, x_train.shape)

# build model
phishing_model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(phishing_features.shape[1],)),  # input shape 111
    layers.Dense(256, activation='relu'),  # hidden layer 1
    layers.Dense(1, activation='sigmoid')  # output layer
])

# compile model with optimizer
phishing_model.compile(
    # loss = tf.keras.losses.MeanSquaredError(), # MSE Loss Function

    ## note: for binary classification (0 and 1), use binary crossentropy
    loss=tf.keras.losses.BinaryCrossentropy(),  # Binary Crossentropy Loss Function
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Adaptive Momentum Optimization
    metrics=['accuracy']  # accuracy because you guys are noob
)

phishing_model.summary()

# train model
hist = phishing_model.fit(
    phishing_features,
    phishing_labels,
    epochs = 10,
    validation_data=(x_val, y_val)
)

with plt.style.context('seaborn'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(hist.history['loss'], label='loss')
    ax1.plot(hist.history['val_loss'], label='val_loss')
    ax2.plot(hist.history['accuracy'], label='accuracy')
    ax2.plot(hist.history['val_accuracy'], label='val_accuracy')
    ax1.legend()
    ax2.legend()
    plt.show()

#TEST CASES, TO VERIFY A SPECIFIC ONE, CHANGE N in .iloc[N] to row no.
test_case = np.expand_dims(np.array(x_test.iloc[7877]), axis=0)
print("index:", 7877)
print(test_case.shape)

prediction = phishing_model.predict(test_case)
prediction = np.squeeze(prediction)

print("predicted index:", int(np.round(prediction)))
print("predicted class:", class_name[int(np.round(prediction))])
print("actual index", y_test.iloc[7877])
print("actual class", class_name[int(y_test.iloc[7877])])

test_case = np.expand_dims(np.array(x_test.iloc[0]), axis=0)
print("index:", 0)
print(test_case.shape)

prediction = phishing_model.predict(test_case)
prediction = np.squeeze(prediction)

print("predicted index:", int(np.round(prediction)))
print("predicted class:", class_name[int(np.round(prediction))])
print("actual index", y_test.iloc[7877])
print("actual class", class_name[int(y_test.iloc[7877])])

#FOR PRESENTATION!!
test_case = np.expand_dims(np.array(x_test.iloc[400]), axis=0)
print("index:", 400)
print(test_case.shape)

prediction = phishing_model.predict(test_case)
prediction = np.squeeze(prediction)

print("Verifying phishing link:")
print("predicted index:", int(np.round(prediction)))
print("predicted class:", class_name[int(np.round(prediction))])
print("actual index", y_test.iloc[400])
print("actual class", class_name[int(y_test.iloc[400])])