import json
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input


load_dotenv()
path_to_training_data = os.getenv('path_to_training_data')
with open(path_to_training_data, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
# print(df.head())

reviews = df['review'].values
sentiments = df['sentiment'].apply(lambda x:1 if x=='positive' else 0).values

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

maxlen = 100
x = pad_sequences(sequences, maxlen=maxlen)

x_train, x_test, y_train, y_test = train_test_split(x, sentiments, test_size=0.2, random_state=42)

model = Sequential()
model.add(Input(shape=(maxlen,))) 
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=5, batch_size=64,validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

def predict_sentiment(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded)
    if prediction>=0.5:
        return 'Positive'
    elif prediction>=0.299 and prediction<=4.99:
        return 'Neutral'
    else:
        return 'Negative'
new_sentence = "I dislike this product, it is not at all good"
print(predict_sentiment(new_sentence))


# model.save('SentiToolKit.h5')
