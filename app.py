from flask import Flask, request, jsonify
import numpy as np
from sklearn import preprocessing
import pandas as pd
import pickle

model = pickle.load(open('fake_profil.sav', 'rb'))
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    profile_pic = request.form.get('profile_pic')
    username_length = request.form.get('username_length')
    fullname_words = request.form.get('fullname_words')
    fullname_length = request.form.get('fullname_length')
    name_username = request.form.get('name_username')
    des_length = request.form.get('des_length')
    ext_url = request.form.get('ext_url')
    private = request.form.get('privat')
    posts = request.form.get('posts')
    followers = request.form.get('followers')
    fallows = request.form.get('fallows')

    list = [profile_pic, username_length, fullname_words, fullname_length, name_username, des_length, ext_url, private,
            posts, followers, fallows]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    li = le.fit_transform(list)

    result = model.predict([li])[0]

    return jsonify({'data': str(result)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
