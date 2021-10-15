import pandas as pd
import numpy as np
import json
import os
import pickle


def read_files(dir,train_file,test_file):
    '''
    lectura de input de los archivos entregados por DACON
    '''

    DIR = dir
    train_soruce = os.path.join(DIR, train_file)
    test_source = os.path.join(DIR, test_file)

    with open(train_soruce) as f:
        train_data = json.loads(f.read())
    
    with open(test_source) as f:
        test_data = json.loads(f.read())

    return train_data,test_data


def format_json(train_data,test_data):
    '''
    Convertir los .json a formato DataFrame.
    '''
    train = pd.DataFrame(columns=['uid', 'title', 'region', 'context', 'summary'])
    uid = 1000
    for data in train_data:
        for agenda in data['context'].keys():
            context = ''
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '
            train.loc[uid, 'uid'] = uid
            train.loc[uid, 'title'] = data['title']
            train.loc[uid, 'region'] = data['region']
            train.loc[uid, 'context'] = context[:-1]
            train.loc[uid, 'summary'] = data['label'][agenda]['summary']
            uid += 1

    test = pd.DataFrame(columns=['uid', 'title', 'region', 'context'])
    uid = 2000
    for data in test_data:
        for agenda in data['context'].keys():
            context = ''
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '
            test.loc[uid, 'uid'] = uid
            test.loc[uid, 'title'] = data['title']
            test.loc[uid, 'region'] = data['region']
            test.loc[uid, 'context'] = context[:-1]
            uid += 1

    return train,test

def create_token(train,test):
    '''
    Tokenizar los textos por ' '.
    '''
    source_tokens = []
    for sentence in train.iloc[0:,3]:
        source_tokens.append(sentence.split(' '))

    target_tokens = []
    for sentence in train.iloc[0:,4]:
        target_tokens.append(sentence.split(' '))

    source_tokens_full = []
    source_aux = pd.concat([train.iloc[0:,3],test.iloc[0:,3]], axis=0)

    for sentence in source_aux:
        source_tokens_full.append(sentence.split(' '))

    return source_tokens,target_tokens,source_tokens_full

def build_token_dict(token_list):
  token_dict = {
      '<PAD>': 0,
      '<START>': 1,
      '<END>': 2
  }
  for tokens in token_list:
    for token in tokens:
      if token not in token_dict:
        token_dict[token] = len(token_dict)
  return token_dict


if __name__== '__main__':

    train_json , test_json = read_files("../data/","train.json","test.json")
    train , test = format_json(train_json,test_json)

    train.iloc[0:,3] = train.iloc[0:,3].str.slice(0,2900)
    train.iloc[0:,4] = train.iloc[0:,4]
    test.iloc[0:,3] = test.iloc[0:,3].str.slice(0,2900)

    source_tokens,target_tokens,source_tokens_full = create_token(train,test)

    source_token_dict = build_token_dict(source_tokens)
    source_token_dict_full = build_token_dict(source_tokens_full)
    target_token_dict = build_token_dict(target_tokens)

    #target_token_dict_inv = {v:k for k,v in target_token_dict.items()}

    # Agregar start, end y pad a cada frase del set de entrenamiento
    encoder_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
    decoder_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
    output_tokens = [tokens + ['<END>'] for tokens in target_tokens]

    source_max_len = max(map(len, encoder_tokens))
    target_max_len = max(map(len, decoder_tokens))

    encoder_tokens = [tokens + ['<PAD>']*(source_max_len-len(tokens) if (source_max_len-len(tokens) >= 0) else 0) for tokens in encoder_tokens]
    decoder_tokens = [tokens + ['<PAD>']*(target_max_len-len(tokens) if (target_max_len-len(tokens) >= 0) else 0) for tokens in decoder_tokens]
    output_tokens = [tokens + ['<PAD>']*(target_max_len-len(tokens) if (target_max_len-len(tokens) >= 0) else 0) for tokens in output_tokens ]

    encoder_input = [list(map(lambda x: source_token_dict_full[x], tokens)) for tokens in encoder_tokens]
    decoder_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decoder_tokens]
    output_decoded = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

    with open('../data/source_token_dict.pickle', 'wb') as handle:
        pickle.dump(source_token_dict, handle)
    
    with open('../data/source_token_dict_full.pickle', 'wb') as handle:
        pickle.dump(source_token_dict_full, handle)

    with open('../data/target_token_dict.pickle', 'wb') as handle:
        pickle.dump(target_token_dict, handle)

    with open('../data/encoder_input.pickle', 'wb') as handle:
        pickle.dump(encoder_input, handle)
    
    with open('../data/decoder_input.pickle', 'wb') as handle:
        pickle.dump(decoder_input, handle)
    
    with open('../data/target_token_dict.pickle', 'wb') as handle:
        pickle.dump(target_token_dict, handle)

    with open('../data/output_decoded.pickle', 'wb') as handle:
        pickle.dump(output_decoded, handle)

    test.to_csv('../data/test_processed.csv',index=False)
  