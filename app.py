from script.predict import predict
from script.config import load_config,load_file
from keras_transformer import get_model
from flask import Flask, render_template, request
import os
import logging

logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

os.environ['TF_KERAS'] = '1'

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    '''
    Recibe el metodo post,levanta los objetos necesarios para generar el resumen. 
    '''

    config = load_config("config.yaml")

    source_token_dict_full = load_file(config['data_directory'],config['source_token_dict_full'])
    target_token_dict = load_file(config['data_directory'],config['target_token_dict'])

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
    model.load_weights(config['model_directory']+config['model'])

    if request.method == 'POST':
        message = request.form['message']
        my_prediction = predict(
            model, message, target_token_dict, source_token_dict_full)
        app.logger.info('Texto recibido : {}'.format(message))
        app.logger.info('Texto resumido : {}'.format(my_prediction))
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':

    app.run(debug=True)
