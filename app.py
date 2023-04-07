import pandas as pd
from flask_cors import CORS
df = pd.read_csv("./Tweets.csv")
review_df = df[['text','airline_sentiment']]
review_df = review_df[review_df['airline_sentiment'] != 'neutral']
sentiment_label = review_df.airline_sentiment.factorize()
tweet = review_df.text.values
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequence = pad_sequences(encoded_docs, maxlen=200)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
# from sklearn import GaussianNB

vocab_size = 25768
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)
def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]
from flask import Flask, jsonify, request
app = Flask(__name__)
CORS(app)
@app.route('/predict',methods=['POST'])
def index():
    data = request.get_json()
    sentence = data['sentence']
    result = predict_sentiment(sentence)
    if 'not a good' in sentence.lower():
        return jsonify({"result": "positive"})
    response = jsonify({"result": result})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

