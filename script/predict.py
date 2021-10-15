import pickle
import os
os.environ['TF_KERAS'] = '1'
from keras_transformer import get_model, decode
import pandas as pd

def predict(model,sentence,target_token_dict,source_token_dict_full):
	
    target_token_dict_inv = {v:k for k,v in target_token_dict.items()}
    sentence_tokens = [tokens + ['<END>', '<PAD>'] for tokens in [sentence.split(' ')]]
    tr_input = [list(map(lambda x: source_token_dict_full[x], tokens)) for tokens in sentence_tokens][0]
    decoded = decode(
        model, 
        tr_input, 
        start_token = target_token_dict['<START>'],
        end_token = target_token_dict['<END>'],
        pad_token = target_token_dict['<PAD>']
    )

    return '{}'.format(' '.join(map(lambda x: target_token_dict_inv[x], decoded[1:-1])))

def result(test,target_token_dict,source_token_dict_full):
    '''
    generar archivo para subbimitear.
    '''
    resultado = []
    for i in range(0,test.shape[0]):
        aux = predict(test.iloc[i,3],target_token_dict,source_token_dict_full)
        resultado.append(aux)
        print('summary:{}'.format(test.iloc[i,3]))
        print('summary:{}'.format(aux))

    resultado_df = pd.DataFrame(resultado)
    sub = pd.read_csv('../data/sample_submission.csv')
    sub['summary'] = resultado
    sub.to_csv('../data/output_dacon.csv', index=False)



if __name__ == '__main__':

    with open('../data/source_token_dict.pickle', 'rb') as handle:
        source_token_dict = pickle.load(handle)

    with open('../data/source_token_dict_full.pickle', 'rb') as handle:
        source_token_dict_full = pickle.load(handle)

    with open('../data/target_token_dict.pickle', 'rb') as handle:
        target_token_dict = pickle.load(handle)
    
    test = pd.read_csv('../data/test_processed.csv')

    # Crear la red transformer
    model = get_model(
        token_num = max(len(source_token_dict_full),len(target_token_dict)),
        embed_dim = 64,
        encoder_num = 3,
        decoder_num = 3,
        head_num = 4,
        hidden_dim = 128,
        dropout_rate = 0.1,
        use_same_embed = False,
    )

    model.compile('adam', 'sparse_categorical_crossentropy')
    model.load_weights('../model/model.h5')

    result(test,target_token_dict,source_token_dict_full)