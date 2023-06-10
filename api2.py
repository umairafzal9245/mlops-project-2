from flask import Flask, request, jsonify, redirect
import re
import nltk
import gensim.corpora as corpora
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the saved LDA model
# lda_model = gensim.models.LdaMulticore.load('./lda_model.model')
lda_model = gensim.models.LdaModel.load('./model/lda_model.model')

@app.route('/', methods=['GET'])
def index():
    matched_strings = request.args.get('matched_strings')
    return f"<h1> LDA </h1><p>Topics: {matched_strings}</p>"


def home():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def lda():
    data = request.form['ans']

    # Remove punctuation and convert to lowercase
    data = re.sub('[,\.!?]', '', data)
    data = data.lower()

    # Tokenize the text and remove stopwords
    data_token = word_tokenize(data)
    data_without_sw = [
        word for word in data_token if not word in stopwords.words()]
    data_without_sw = set(data_without_sw)

    # Create dictionary and corpus for LDA
    data_without_sw = [d.split() for d in data_without_sw]
    id2word = corpora.Dictionary(data_without_sw)
    texts = data_without_sw
    corpus = [id2word.doc2bow(text) for text in texts]

    # Get topics from the pre-trained LDA model
    lda_topics = lda_model.show_topics(num_words=8)

    # Preprocess the topics and match with a predefined list
    topics = []
    filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
    for topic in lda_topics:
        topics.append(preprocess_string(topic[1], filters))

    final_topics = []
    for i in range(3):
        final_topics.append(topics[i])

    final_topics = [item for sublist in final_topics for item in sublist]
    final_topics = list(set(final_topics))
    list2 = ['virtual function', 'objects', 'member function', 'pointers', 'friend function',
             'pure virtual function', 'abstract class', 'derived class', 'function overriding']
    matched_strings = []

    for string in final_topics:
        for s in list2:
            if string in s:
                matched_strings.append(s)
                break

    # return jsonify({'topics': matched_strings})
    return redirect(f'/?matched_strings={matched_strings}')


if __name__ == '__main__':
    exit()
    app.run(debug=True)