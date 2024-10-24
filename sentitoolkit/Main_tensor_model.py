import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


tokenizer_path = 'tokenizer.pkl'

class SentiToolKit:
    def __init__(self, model_path='SentiToolKit.keras', tokenizer_path=tokenizer_path, maxlen=100, vocab_size=5000):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.tokenizer = None

    
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully from", model_path)

 
        self.load_tokenizer(tokenizer_path)

    def load_tokenizer(self, tokenizer_path):
        """
        Load the tokenizer from a pickle file.
        """
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully from", tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Please ensure it is available.")

    def prepare_text(self, sentence):
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Ensure that the tokenizer is correctly loaded.")
        
   
        sequence = self.tokenizer.texts_to_sequences([sentence])
      
        padded = pad_sequences(sequence, maxlen=self.maxlen)
        return padded

    def __call__(self, sentence):
        """
        Make a prediction for the given sentence by passing it through the model.
        """
        prepared_text = self.prepare_text(sentence)
        
        prediction = self.model.predict(prepared_text)
        predicted_class = prediction.argmax(axis=-1)

        
        if predicted_class == 2:
            return 'Positive'
        elif predicted_class == 1:
            return 'Neutral'
        else:
            return 'Negative'



if __name__ == '__main__':
    toolkit = SentiToolKit(model_path='SentiToolKit.keras', tokenizer_path=tokenizer_path)
    test_sentence = input("Enter a review for sentiment analysis: ")
    result = toolkit(test_sentence)
    print(f"Predicted sentiment: {result}")