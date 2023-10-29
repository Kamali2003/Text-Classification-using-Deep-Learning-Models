import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

from keras.datasets import imdb
(train_data,train_targets),(test_data,test_targets)= imdb.load_data(num_words=10000)
db=np.concatenate((train_data,test_data),axis=0)
targets=np.concatenate((train_targets,test_targets),axis=0)

print('categories:',np.unique(targets))
print('No.of unique words:',len(np.unique(np.hstack(db))))

length=[len(i) for i in db]
print('avg review length:',np.mean(length))
print("standard deviation:",round(np.std(length)))

print('label:',targets[0])
print(db[0])

index=imdb.get_word_index()
reverse_index=dict([(value,key) for (key,value) in index.items()])
decoder=' '.join([reverse_index.get(i - 3,'#') for i in db[0]])
print(decoder)

def vectorize(sequences,dimension=10000):
  results=np.zeros((len(sequences), dimension))
  for i,sequence in enumerate(sequences):
      results[i,sequence]=1
  return results

db=vectorize(db)
targets=np.array(targets).astype('float32')

test_x = db[:10000]
test_y = targets[:10000]
train_x = db[10000:]
train_y = targets[10000:]
model=models.Sequential()
model.add(layers.Dense(50,activation='relu',input_shape=(10000, )))
model.add(layers.Dropout(0.3,noise_shape=None,seed=None))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dropout(0.2,noise_shape=None,seed=None))
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_x,train_y,epochs=5,batch_size=500,validation_data=(test_x,test_y))

print('test-accuracy:',np.mean(history.history['val_accuracy']))
epochs=5

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

from sklearn.metrics import precision_score,f1_score,recall_score
y_pred=(model.predict(test_x)>0.5).astype(int)

precision=precision_score(test_y,y_pred)
recall=recall_score(test_y,y_pred)
f1=f1_score(test_y,y_pred)

print(precision)
print(recall)
print(f1)
