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
from tensorflow.keras.utils import to_categorical
import pickle


load_dotenv()
path_to_training_data = os.getenv('path_to_training_data')
with open(path_to_training_data, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
# print(df.head())

reviews = df['review'].values
sentiments = df['sentiment'].apply(lambda x:2 if x=='positive' else(1 if x=='neutral' else 0)).values

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)


maxlen = 100
x = pad_sequences(sequences, maxlen=maxlen)
y_one_hot = to_categorical(sentiments, num_classes=3)
x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train))
batch_size = 64
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model = Sequential()
model.add(Input(shape=(maxlen,))) 
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=200, batch_size=64,validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

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
    
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved successfully as 'tokenizer.pkl'.")


model.save('SentiToolKit.keras')
new_sentence = input("Enter the review which you wish to analyze: ")
print(predict_sentiment(new_sentence))


