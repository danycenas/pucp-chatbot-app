from random import choice

import spacy
nlp = spacy.load('en')

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