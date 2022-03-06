from flask import Flask, render_template, request, url_for, redirect, session
import sqlite3
import pandas as pd
import time
import numpy as np
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#Create database and table
"""db = sqlite3.connect('database.db')
cur = db.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS Posts (message TEXT, topic TEXT, label TEXT)')
db.commit()

#convert dataframe to database
df = pd.read_pickle('C:/Users/grego/Documents/Modo_bot/JVC.pkl')
df['Label'] = 0
data = [tuple(x) for x in df.values]
cur.executemany('''INSERT INTO Posts(message, topic, label) VALUES(?,?,?)''', data)
db.commit()
cur.execute('SELECT COUNT(*) FROM Posts')
print(cur.fetchall())
"""
app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/user_labelisation')
def get_labels():
    db = sqlite3.connect('database.db')
    cur = db.cursor()
    message = []
    for content in cur.execute('SELECT message, topic, rowid FROM Posts WHERE rowid IN (SELECT rowid FROM Posts WHERE Posts.label = 0 ORDER BY RANDOM() LIMIT 1)'):
        message.append(content)
    message, topic, rowid = message[0]
    session['rowid'] = rowid
    text = "TOPIC: {}\n MESSAGE: {}".format(topic, message)
    return render_template('form.html', topic = topic, message = message)

@app.route('/save_label', methods= ['POST'])
def save_label():
    label = request.form['submit_button']
    rowid = session['rowid']
    db = sqlite3.connect('database.db')
    cur = db.cursor()
    cur.execute('UPDATE Posts SET label = ? WHERE rowid = ?',(label, rowid))
    db.commit()
    return redirect(url_for('get_labels'))

@app.route('/check_labellized_data')
def checking():
    db = sqlite3.connect('database.db')
    cur = db.cursor()
    for content in cur.execute('SELECT message, topic, label FROM Posts WHERE label != 0'):
        print(content)
    return "oui"

@app.route('/user_phrase')
def user_gives_data():
    db = sqlite3.connect('database.db')
    cur = db.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS User_data (message TEXT, label TEXT)')
    db.commit()
    return render_template('user_phrases.html')


@app.route('/add_user_data', methods = ['POST'])
def user_data_to_db():
    phrase = request.form['user_phrase']
    db = sqlite3.connect('database.db')
    cur = db.cursor()
    cur.execute('INSERT INTO User_data (message, label) VALUES (?,?)',(phrase, 'insultant'))
    db.commit()
    return str(phrase.upper) + '\nMERCI!'

@app.route('/model')
def test_model():
    return render_template('user_test.html')

@app.route('/predict', methods = ['POST'])
def model_pred():
    #database as dataframe
    db = sqlite3.connect('database.db')
    user_data = pd.read_sql_query("SELECT * FROM User_data", db)
    scrapped_data = pd.read_sql_query("SELECT * FROM Posts WHERE label != 0 AND label != 'inexploitable' AND label != 'neutre'", db)
    del scrapped_data['topic']
    df = pd.concat([scrapped_data, user_data])
    df = df.reset_index()
    del df['index']

    #dataframe nettoyé avec autant de positifs que négatifs
    positives = df[df['label'] == 'insultant']
    negatives = df[df['label'] == 'pas insultant']
    negatives_sample = negatives.sample(n = len(positives)*2) #on équilibre le nb de positifs et de négatifs
    #negatives_sample = negatives
    df = pd.concat([negatives_sample, positives])
    df = df.reset_index()
    del df['index']

    #labels numériques et corpus d'entrainement
    df = df.replace('pas insultant', 0)
    df = df.replace('insultant', 1)
    corpus = df['message']
    phrase_test = request.form['user_test']
    phrase_test = pd.Series([phrase_test])
    corpus = corpus.append(phrase_test)


    #découpage data/label
    y_train = df['label']
    x_train = df['message']
    x_test = phrase_test

    vec = TfidfVectorizer()
    vocab = vec.fit(corpus)
    x_train_tfidf = vocab.transform(x_train)
    x_test_tfidf = vocab.transform(x_test)
    NB_clf = MultinomialNB().fit(x_train_tfidf, y_train)
    SGD_clf = SGDClassifier().fit(x_train_tfidf, y_train)
    predicted_NB = NB_clf.predict(x_test_tfidf)
    print(predicted_NB)
    print(phrase_test)
    predicted_SGD = SGD_clf.predict(x_test_tfidf)
    print(predicted_SGD)
    if predicted_SGD[0] == 1:
        return "Quelle violence..."
    else: 
        return "Vous êtes très courtois, bravo!"


if __name__ == '__main__':
    app.run()
