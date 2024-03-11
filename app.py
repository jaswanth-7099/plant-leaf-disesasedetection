import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from flask import Flask, redirect, render_template, request,jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
from flask_cors import CORS
import numpy as np
import torch
import joblib
import re
import string
import pandas as pd

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')

#for chatbot
model_chat = load_model('AI_lstm_model.h5')
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.pkl')
df2 = pd.read_csv('response.csv')

#end-chatbot

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1.pt"))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


#chat-bot

def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens


def remove_stop_words_for_input(tokenizer, df, feature):
    doc_without_stopwords = []
    entry = df[feature][0]
    tokens = tokenizer(entry)
    doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return df


def encode_input_text(tokenizer_t, df, feature):
    t = tokenizer_t
    entry = [df[feature][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=8, padding='post')
    return padded

def get_pred(model, encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred

def get_response(df2, pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0, upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]

def get_text(str_text):
    input_text = [str_text]
    df_input = pd.DataFrame(input_text, columns=['questions'])
    return df_input

def bot_precausion(df_input,pred):
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    return pred

#end-chatbot


app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/newf')
def newf():
    return render_template('new.html')
    
@app.route('/aboutf')
def aboutf():
    scroll_to_contact = request.args.get('scrollToContact', default='false')
    return render_template('about.html', scroll_to_contact=scroll_to_contact)


@app.route('/indexf')
def indexf():
    return render_template('index.html')
    

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static', filename)
        image.save(file_path)
        # print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        # print(description)
        # print(prevent)
        return render_template('new.html',title=title,description=description)
    else:
        return render_template('new.html',title="none")
    

@app.route('/submi', methods=['POST'])
def submi():
    if request.method == 'POST':
        data = request.get_json()
        
        if data and 'message' in data:
            df_input = data['message']

            # Check if df_input is empty
            if not df_input.strip():  # This will check for an empty string or a string with only whitespace
                return jsonify({'title': 'none'})

            # The rest of your code remains the same
            df_input = get_text(df_input)

            tokenizer_t = joblib.load('tokenizer_t.pkl')
            vocab = joblib.load('vocab.pkl')

            df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
            encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

            pred = get_pred(model_chat,encoded_input)
            pred = bot_precausion(df_input,pred)

            response = get_response(df2,pred)
            # print("Response: "+response)
            # print("below data")
            # print(data)
            return jsonify({'title': response}) 
        else:
            return jsonify({'title': 'none'})
    else:
        return jsonify({'title': 'none'})

    
if __name__ == '__main__':
    app.run(debug=True)