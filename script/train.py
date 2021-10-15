import pandas as pd
import numpy as np
import os
import pickle
import random

os.environ['TF_KERAS'] = '1'
from keras_transformer import get_model, decode

def seed_everything(seed: int = 199030):
    '''
    Definir seed para replicar.
    '''
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':
    
    seed_everything()

    with open('../data/source_token_dict.pickle', 'rb') as handle:
        source_token_dict = pickle.load(handle)

    with open('../data/source_token_dict_full.pickle', 'rb') as handle:
        source_token_dict_full = pickle.load(handle)

    with open('../data/target_token_dict.pickle', 'rb') as handle:
        target_token_dict = pickle.load(handle)

    with open('../data/encoder_input.pickle', 'rb') as handle:
        encoder_input = pickle.load(handle)

    with open('../data/decoder_input.pickle', 'rb') as handle:
        decoder_input = pickle.load(handle)

    with open('../data/output_decoded.pickle', 'rb') as handle:
        output_decoded = pickle.load(handle)

    x = [np.array(encoder_input), np.array(decoder_input)]
    y = np.array(output_decoded)

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

    model.fit(x,y, epochs=80, batch_size=32)
    model.save_weights('../model/model.h5')
