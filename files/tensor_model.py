import json
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import pickle


load_dotenv()
path_to_training_data = os.getenv('path_to_training_data')
with open(path_to_training_data, 'r', encoding='utf-8') as f:
    data = json.load(f)


df = pd.DataFrame(data)
reviews = df['review'].values
sentiments = df['sentiment'].apply(lambda x: 2 if x == 'positive' else (1 if x == 'neutral' else 0)).values


tokenizer = Tokenizer(num_words=5000)  
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)


maxlen = 100
x = pad_sequences(sequences, maxlen=maxlen)
y_one_hot = to_categorical(sentiments, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)

embedding_dim = 50  

model = Sequential([
    Input(shape=(maxlen,)),
    Embedding(input_dim=5000, output_dim=embedding_dim, input_length=maxlen, trainable=True),
    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    LSTM(32, dropout=0.3, recurrent_dropout=0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(3, activation='softmax', kernel_regularizer=l2(0.001))
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)


model.summary()

class_weights = {0: 2.0, 1: 2.0, 2: 1.0}
history = model.fit(
    x_train, y_train, 
    epochs=100, batch_size=64, validation_split=0.2, class_weight=class_weights,
    callbacks=[checkpoint, lr_scheduler]
)


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
    return ['Negative', 'Neutral', 'Positive'][predicted_class[0]]


model.save('SentiToolKit.keras')


new_sentence = input("Enter the review which you wish to analyze: ")
print(predict_sentiment(new_sentence))
