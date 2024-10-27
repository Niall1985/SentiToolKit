import json
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import pickle
import numpy as np

load_dotenv()
path_to_training_data = os.getenv('path_to_training_data')
with open(path_to_training_data, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
reviews = df['review'].values
sentiments = df['sentiment'].apply(lambda x: 2 if x == 'positive' else (1 if x == 'neutral' else 0)).values


tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

maxlen = 250
x = pad_sequences(sequences, maxlen=maxlen)
y_one_hot = to_categorical(sentiments, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)


embedding_dim = 100

model = Sequential()
model.add(Input(shape=(maxlen,)))
model.add(Embedding(input_dim=8000, output_dim=embedding_dim, input_length=maxlen, trainable=False))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax', kernel_regularizer=l2(0.0005)))


optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


model.summary()


class_weights = {0: 2.0, 1: 2.0, 2: 1.0}
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, class_weight=class_weights,
                    callbacks=[checkpoint, lr_scheduler])


loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')


with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved successfully as 'tokenizer.pkl'.")


def predict_sentiment(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded)
    predicted_class = prediction.argmax(axis=-1)
    if predicted_class == 2:
        return 'Positive'
    elif predicted_class == 1:
        return 'Neutral'
    else:
        return 'Negative'


model.save('SentiToolKit.keras')


new_sentence = input("Enter the review which you wish to analyze: ")
print(predict_sentiment(new_sentence))

