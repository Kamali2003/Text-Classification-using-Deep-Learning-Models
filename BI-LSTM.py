import numpy as np
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.layers import Dropout
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=10000)
maxlen = 200
x_train =pad_sequences(x_train, maxlen=maxlen)
x_test =pad_sequences(x_test, maxlen=maxlen)
y_test = np.array(y_test)
y_train = np.array(y_train)
n_unique_words=10000

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

import matplotlib.pyplot as plt

epochs=10

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, epochs+1), history.history['loss'], label='Training Loss')
plt.plot(np.arange(1, epochs+1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, epochs+1), history.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(1, epochs+1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Predictions
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# R-squared and Adjusted R-squared
r2 = r2_score(y_test, y_pred)
num_samples, num_features = x_test.shape[0], x_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (num_samples - 1) / (num_samples - num_features - 1))

print("Confusion Matrix:\n", conf_matrix)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)

import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred_classes)

# Plot the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
