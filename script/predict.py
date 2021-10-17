import pandas as pd
import os
from script.config import load_config,load_file
os.environ['TF_KERAS'] = '1'

from keras_transformer import get_model, decode


def predict(model, sentence, target_token_dict, source_token_dict_full):
    '''
    Servir la prediccion a partir del modelo entrenado.
    '''
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
    sentence_tokens = [tokens + ['<END>', '<PAD>']
                       for tokens in [sentence.split(' ')]]
    tr_input = [list(map(lambda x: source_token_dict_full[x], tokens))
                for tokens in sentence_tokens][0]
    decoded = decode(
        model,
        tr_input,
        start_token=target_token_dict['<START>'],
        end_token=target_token_dict['<END>'],
        pad_token=target_token_dict['<PAD>']
    )

    return '{}'.format(' '.join(map(lambda x: target_token_dict_inv[x], decoded[1:-1])))


def result(model,test, target_token_dict, source_token_dict_full):
    '''
    generar archivo para subbimitear en DACON.
    '''
    resultado = []
    for i in range(0, test.shape[0]):
        _aux = predict(model,test.iloc[i, 3],
                       target_token_dict, source_token_dict_full)
        resultado.append(_aux)
        print('summary:{}'.format(test.iloc[i, 3]))
        print('summary:{}'.format(_aux))

    resultado_df = pd.DataFrame(resultado)
    sub = pd.read_csv(config['data_directory']+config['sample_sub'])
    sub['summary'] = resultado
    sub.to_csv(config['data_directory']+config['output_dacon'], index=False)

if __name__ == '__main__':

    config = load_config("config.yaml")

    source_token_dict_full = load_file(config['data_directory'],config['source_token_dict_full'])
    target_token_dict = load_file(config['data_directory'],config['target_token_dict'])
    test = pd.read_csv(config['data_directory']+config['test_processed'])

    # Crear la red transformer
    model = get_model(
        token_num=max(len(source_token_dict_full), len(target_token_dict)),
        embed_dim = config["embed_dim"],
        encoder_num = config["encoder_num"],
        decoder_num = config["decoder_num"],
        head_num = config["head_num"],
        hidden_dim = config["hidden_dim"],
        dropout_rate = config["dropout_rate"],
        use_same_embed = config["use_same_embed"],
    )

    model.compile(config["opt"], config["metric"])
    model.load_weights(config['model_directory']+config['model'])

    result(model,test, target_token_dict, source_token_dict_full)

