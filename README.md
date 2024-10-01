# SentiToolKit

## TensorFlow Model
This repository contains a TensorFlow-based deep learning model for sentiment analysis. The model is designed to classify text into three categories: Positive, Negative, and Neutral sentiments based on the input data. It uses LSTM (Long Short-Term Memory) layers to capture the context of the input text and make predictions.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Data](#training-data)
- [Evaluation](#evaluation)
- [Improvements](#improvements)
- [License](#license)

## Features
- Text preprocessing: tokenization, padding, and sequence conversion.
- LSTM-based architecture to capture temporal dependencies in text data.
- Multi-class classification (Positive, Negative, Neutral).
- Adjustable training parameters: epochs, batch size, validation split.
  
## Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- dotenv (for environment variable management)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Niall1985/SentiToolKit.git
   cd SentiToolKit
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for the path to your training data:
   - Create a `.env` file in the root directory and add the following line:
     ```
     path_to_training_data="path_to_your_json_data"
     ```

## Usage

### Training the Model
1. Ensure that your training data is in JSON format with two columns: "review" (the text) and "sentiment" (positive, negative, neutral).
2. Run the script to train the model:
   ```bash
   python tensor_model.py
   ```
   The model will automatically split your data into training and testing sets, train the model, and display accuracy metrics.

### Predicting Sentiment
You can use the model to predict sentiment for a new sentence:

```python
new_sentence = "The battery life is poor and bad, but the display is superb and the keys are smooth"
print(predict_sentiment(new_sentence))
```

## Model Architecture

- **Embedding Layer**: Maps input words into dense vectors of fixed size.
- **LSTM Layer**: Captures sequential dependencies in the text.
- **Dense Layer with Softmax Activation**: Outputs three probabilities for each class (Positive, Negative, Neutral).

### Code Snippet:

```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Training Data
- The dataset used for training consists of labeled text reviews and their corresponding sentiments. The sentiments are categorized into three classes: Positive, Negative, and Neutral.

### Data Preprocessing:
- Convert the text to lowercase.
- Remove special characters and stop words.
- Tokenize the sentences and pad sequences to a fixed length (100 words).
  
## Evaluation
After training, the model will evaluate the performance on the test set. The evaluation includes accuracy and loss metrics.

### Example Output:
```
Test Accuracy: 87.55%
```

## Improvements
- **Data Augmentation**: Adding more diverse examples for neutral sentiments can improve performance.
- **Hyperparameter Tuning**: Experimenting with different numbers of LSTM units, dropout rates, and batch sizes may lead to better results.
- **Pre-trained Embeddings**: Using pre-trained word embeddings like GloVe or Word2Vec may enhance the model's ability to understand the context of words.
- **Handling Imbalanced Data**: Use techniques like oversampling, undersampling, or SMOTE to handle any imbalances in the sentiment categories.

## License
This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007 - see the [LICENSE](LICENSE) file for details.
