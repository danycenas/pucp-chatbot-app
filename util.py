import tensorflow as tf
import numpy as np
import pandas as pd
import re
import os
import csv
from tqdm import tqdm
import faiss
from nltk.translate.bleu_score import sentence_bleu

from random import choice

import spacy
nlp = spacy.load('en_core_web_sm')

rules = {
    "saludos": "saludos",
    "negacion": "negacion",
    "afirmación": "afirmación",
    "despedida": "despedida"
}

def predict_intent(sentence, count_vectorizer, model):
  vector = count_vectorizer.transform([ generate_bigrams(nlp(sentence)) ])
  
  intent = model.predict(vector)
  return intent[0]

def generate_answer(utterance, utterances_examples):
  answers = utterances_examples[utterance]
  answer = choice(answers)
  return answer

def return_answer(sentence, count_vectorizer, model, rules, utterances_examples,biobert_tokenizer,question_extractor_model,gpt2_tokenizer,tf_gpt2_model):
  intent = predict_intent(sentence, count_vectorizer, model)
  
  if intent == 'consulta_medica':
    answer = final_func_1(sentence,biobert_tokenizer,question_extractor_model,gpt2_tokenizer,tf_gpt2_model)
  else:
    utterance = rules[intent]
    answer = generate_answer(utterance, utterances_examples)

  return answer, intent

def generate_bigrams(sentence):
  bigram_list = []
  for token in sentence:
    if(token.dep_ != 'ROOT' and token.dep_ != 'punct'):
      bigram_list.append(token.head.text+'_'+token.text)
  
  example = sentence.text
  if(len(bigram_list) > 0): example = ' '.join(bigram_list)
  return example

def final_func_1(question,biobert_tokenizer,question_extractor_model,gpt2_tokenizer,tf_gpt2_model):
  answer_len=25
  return give_answer(question,answer_len,biobert_tokenizer,question_extractor_model,gpt2_tokenizer,tf_gpt2_model)

def give_answer(question,answer_len,biobert_tokenizer,question_extractor_model,gpt2_tokenizer,tf_gpt2_model):
  preprocessed_question=preprocess(question)
  question_len=len(preprocessed_question.split(' '))
  truncated_question=preprocessed_question
  if question_len>500:
    truncated_question=' '.join(preprocessed_question.split(' ')[:500])
  encoded_question= biobert_tokenizer.encode(truncated_question)
  max_length=512
  padded_question=tf.keras.preprocessing.sequence.pad_sequences(
      [encoded_question], maxlen=max_length, padding='post')
  question_mask=[[1 if token!=0 else 0 for token in question] for question in padded_question]
  embeddings=question_extractor_model({'question':np.array(padded_question),'question_mask':np.array(question_mask)})
  gpt_input=preparing_gpt_inference_data(truncated_question,embeddings.numpy(),gpt2_tokenizer)
  mask_start = len(gpt_input) - list(gpt_input[::-1]).index(4600) + 1
  input=gpt_input[:mask_start+1]
  if len(input)>(1024-answer_len):
   input=input[-(1024-answer_len):]
  gpt2_output=gpt2_tokenizer.decode(tf_gpt2_model.generate(input_ids=tf.constant([np.array(input)]),max_length=1024,temperature=0.7)[0])
  answer=gpt2_output.rindex('`ANSWER: ')
  return gpt2_output[answer+len('`ANSWER: '):]

def preparing_gpt_inference_data(question,question_embedding,gpt2_tokenizer):
  topk=20
  scores,indices=answer_index.search(
                  question_embedding.astype('float32'), topk)
  q_sub=qa.iloc[indices.reshape(20)]
  
  line = '`QUESTION: %s `ANSWER: ' % (
                        question)
  encoded_len=len(gpt2_tokenizer.encode(line))
  for i in q_sub.iterrows():
    line='`QUESTION: %s `ANSWER: %s ' % (i[1]['question'],i[1]['answer']) + line
    line=line.replace('\n','')
    encoded_len=len(gpt2_tokenizer.encode(line))
    if encoded_len>=1024:
      break
  return gpt2_tokenizer.encode(line)[-1024:]

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase


def preprocess(text):
    # convert all the text into lower letters
    # remove the words betweent brakets ()
    # remove these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
    # replace these spl characters with space: '\u200b', '\xa0', '-', '/'
    
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[$)\?"’.°!;\'€%:,(/]', '', text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    return text