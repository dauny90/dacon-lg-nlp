from flask import Flask,render_template,url_for,request
import pickle
import os
os.environ['TF_KERAS'] = '1'
from keras_transformer import get_model, decode
from script.predict import predict

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/summarize',methods=['POST'])
def summarize():

	with open('../data/source_token_dict.pickle', 'rb') as handle:
		source_token_dict = pickle.load(handle)

	with open('../data/source_token_dict_full.pickle', 'rb') as handle:
		source_token_dict_full = pickle.load(handle)

	with open('../data/target_token_dict.pickle', 'rb') as handle:
		target_token_dict = pickle.load(handle)

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

	if request.method == 'POST':
		message = request.form['message']
		my_prediction = predict(model,message,target_token_dict,source_token_dict_full)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)