import pandas as pd
import numpy as np
import pickle
import random
import os
import yaml
from config import load_config,CONFIG_PATH,seed_everything,load_dict,load_file
os.environ['TF_KERAS'] = '1'
from keras_transformer import get_model, decode

if __name__ == '__main__':

    seed_everything()
    config = load_config("config.yaml")

    source_token_dict_full = load_file(config['data_directory'],config['source_token_dict_full'])
    target_token_dict = load_file(config['data_directory'],config['target_token_dict'])
    encoder_input = load_file(config['data_directory'],config['encoder_input'])
    decoder_input = load_file(config['data_directory'],config['decoder_input'])
    output_decoded = load_file(config['data_directory'],config['output_decoded'])

    x = [np.array(encoder_input), np.array(decoder_input)]
    y = np.array(output_decoded)

    model = get_model(
        token_num=max(len(source_token_dict_full), len(target_token_dict)),
        embed_dim=config["embed_dim"],
        encoder_num=config["encoder_num"],
        decoder_num=config["decoder_num"],
        head_num=config["head_num"],
        hidden_dim=config["hidden_dim"],
        dropout_rate=config["dropout_rate"],
        use_same_embed=config["use_same_embed"],
    )
    model.compile(config["opt"], config["metric"])

    model.fit(x, y, epochs=config["epochs"], batch_size=config["batch_size"])
    model.save_weights(config['model_directory']+config['model'])
