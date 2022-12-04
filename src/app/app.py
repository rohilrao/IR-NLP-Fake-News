import os
from flask import Flask, request, render_template
#from lime_explainer import explainer,  METHODS
from lime.lime_text import LimeTextExplainer
import newspaper
from newspaper import Article
import urllib
import joblib


import contractions
import string
#import nltk
#from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#from nltk.corpus import stopwords

'''
Deployment: https://realpython.com/flask-by-example-part-1-project-setup/
'''


def fix_contractions(text):
    return contractions.fix(text)


def stopword_specialchar_removal(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if (word.isalpha() and (word not in stop_words))]
    return " ".join(words)

app = Flask(__name__)
SECRET_KEY = os.urandom(24)

@app.route('/')
@app.route('/result', methods=['POST'])
def index():
    exp = ""
    if request.method == 'POST':
        text = str(request.form['entry'])
        url = urllib.parse.unquote(text)
        article = Article(str(url))
        article.download()
        article.parse()
        article.nlp()
        news = article.text

        news = news.lower()
        news = fix_contractions(news)
        #news = stopword_specialchar_removal(news)

        #news = article.summary

        explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
        clf_method = request.form['classifier']
        nfeatures = int(request.form['nfeatures'])
        if clf_method == "Decision Tree":
            dt_model = joblib.load('dt_pipeline.pkl')
            exp = explainer.explain_instance(news, classifier_fn= dt_model.predict_proba, num_features= nfeatures)
        elif clf_method == "Adaboost":
            adab_model = joblib.load('adab_pipeline.pkl')
            exp = explainer.explain_instance(news, classifier_fn=adab_model.predict_proba, num_features= nfeatures)
        elif clf_method == "Logistic Regression":
            logit_model = joblib.load('lr_pipeline.pkl')
            exp = explainer.explain_instance(news, classifier_fn=logit_model.predict_proba, num_features= nfeatures)
        elif clf_method == "Naive Bayes":
            nb_model = joblib.load('nb_pipeline.pkl')
            exp = explainer.explain_instance(news, classifier_fn=nb_model.predict_proba, num_features= nfeatures)
        elif clf_method == "Random Forest":
            rf_model = joblib.load('rf_pipeline.pkl')
            exp = explainer.explain_instance(news, classifier_fn=rf_model.predict_proba, num_features= nfeatures)
        #model = joblib.load('rf_pipeline.pkl')

        exp = exp.as_html()

        # allmodels = {"dt":dt_model, "rf":rf_model, "adab":adab_model, "logit":logit_model, "nb":nb_model}
        # allpreds = {}
        # for m in allmodels.keys():
        #     if allmodels[m].predict([text]) == 0:
        #         allpreds[m] = "Fake"
        #     elif allmodels[m].predict([text]) == 1:
        #         allpreds[m] = "Real"

        #allpreds = {"dt":dt_model.predict([text]), "rf":rf_model.predict([text]), "adab":adab_model.predict([text]), "logit": logit_model.predict([text]), "nb":nb_model.predict([text])}

        #return render_template('index.html', exp=exp,  entry=news, url = url, allpreds = allpreds, classifier = clf_method)
        return render_template('index.html', exp=exp, entry=news, url=url,  classifier=clf_method)
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = SECRET_KEY
    app.run(debug=True)
