from flask import Flask, request, jsonify, render_template
import requests
import json
import logging

from transformers import AutoTokenizer, TFAutoModel
from transformers import GPT2Tokenizer,TFGPT2LMHeadModel
import tensorflow as tf
import pickle

from util import *

app = Flask(__name__)

# biobert_tokenizer = None
# question_extractor_model = None
# gpt2_tokenizer = None
# tf_gpt2_model = None

@app.route('/')
def index():
#   biobert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
#   question_extractor_model=tf.keras.models.load_model('question_extractor_model_2_11')
#   gpt2_tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
#   tf_gpt2_model=TFGPT2LMHeadModel.from_pretrained("./tf_gpt2_model_2_104_10000")

  return render_template('index.html')


@app.route('/send', methods=['POST'])
def send_message():
    message = request.form['message']
    # print (message)

    # json = request.get_json()
    # print (json['message'])

    # Modelos de cosulta medica
    biobert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
    question_extractor_model=tf.keras.models.load_model('question_extractor_model_2_11')
    gpt2_tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    tf_gpt2_model=TFGPT2LMHeadModel.from_pretrained("./tf_gpt2_model_2_104_10000")
    # # Clasificacion de intenciones
    count_vectorizer = pickle.load(open("count_vectorizer.pickle", "rb"))
    decision_tree_classifier = pickle.load(open("decision_tree_classifier.pickle", "rb"))
    rules = pickle.load(open("rules.pickle", "rb"))
    utterances_examples = pickle.load(open("utterances_examples.pickle", "rb"))
    answer, intent = return_answer(message, count_vectorizer, decision_tree_classifier, rules, utterances_examples,
                                   biobert_tokenizer,question_extractor_model,gpt2_tokenizer,tf_gpt2_model) if message != 'Yakarta' else message

    # answer = 'aaaa'

    response_text = { "message":  answer }
    return jsonify(response_text)

if __name__ == '__main__':
   app.run(host='0.0.0.0')