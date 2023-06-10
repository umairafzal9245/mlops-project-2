

from flask import Flask, jsonify, request, redirect, render_template, make_response
from flask_cors import CORS

# import mysql.connector as connector
# from mysql.connector import errorcode
from nltk.corpus import stopwords
# from flask import Flask, jsonify, request, redirect, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import math
import numpy as np
import Levenshtein
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from db import db_connect, return_score

app = Flask(__name__)
CORS(app)


def return_score(text1, text2):

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    text_list = [text1, text2]
    sentence_embeddings = model.encode(text_list)
    ans = cosine_similarity(
        [sentence_embeddings[0]],
        sentence_embeddings[1:]
    )
    return ans

def score2(text1, text2):
    text_list = [text1, text2]
    # Tokenization of each document
    tokenized_sent = []
    for s in text_list:
        tokenized_sent.append(word_tokenize(s.lower()))

    # Flatten the tokenized sentences into a single list of words
    words = []
    for sentence in tokenized_sent:
        words.extend(sentence)

    # Remove duplicate words
    words = list(set(words))

    # Create vectors for each sentence
    word_vectors = []
    for sentence in tokenized_sent:
        # Initialize a vector of zeros for each sentence
        sentence_vector = np.zeros(len(words))
        # Count the occurrence of each word in the sentence
        for word in sentence:
            sentence_vector[words.index(word)] += 1
        word_vectors.append(sentence_vector)

    # Calculate the cosine similarity between the two sentences
    cosine_score = cosine(word_vectors[0], word_vectors[1])

    print("cosine_score", cosine_score)
    return cosine_score


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# Function to check if user has not repeated a same sentence just to populate the answer.
def check_duplicate_sentences(text):
    sentences = text.split('.')
    n = len(sentences)
    for i in range(n):
        for j in range(i+1, n):
            sim_score = 1 - \
                Levenshtein.distance(
                    sentences[i], sentences[j])/max(len(sentences[i]), len(sentences[j]))
            # print("score=", sim_score)
            if sim_score > 0.9:   # set a threshold for similarity score
                return True
    return False

@app.route('/')
def index():
    score = request.args.get('score')
    return redirect(f'/?score={score}')
    # return f"<h1> Hello World </h1><p>Score: {score}</p>"


@app.route('/quesans', methods=['POST'])
def quesans():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        return response
    
    original_ans = request.form['original_ans']
    user_ans = request.form['user_ans']
    score = return_score(original_ans, user_ans)
    score_2 = score2(original_ans, user_ans)
    score = score[0][0]

    # ---------------------------
    score = score*10
    print("score=", score)
    # -----------------------------
    score_2 = score_2*10
    print("score_2 =", score_2)

    if score > 8.7 and score_2 < 6:
        x = 0.60
        y = 0.40
    elif score < 8.5 and score_2 > 6:
        x = 0.40
        y = 0.60
    else:
        x = 0.5
        y = 0.5

    if score < 4.9:
        score = 0
    elif score >= 5 and score <= 6.5:
        score = math.floor(score-4)
    elif score >= 6.6 and score <= 7.9:
        score = math.floor(score-3)
    elif score >= 8 and score <= 8.5:
        score = math.floor(score-2)
    else:
        score = round(score)

    if score_2 < 3.4:
        score_2 = 0
    elif score_2 >= 3.5 and score <= 4:
        score_2 = math.floor(score_2-2)
    else:
        score_2 = round(score_2)

    print("score VERSION2  =", score)
    print("score_2 VERSION2  =", score_2)
    print("S1--------", score*x)
    print("S2--------", score_2*y)
    total_score = (score*x) + (score_2*y)

    if total_score < 7.5:
        total_score = math.floor(total_score)
    else:
        total_score = round(total_score)

    if check_duplicate_sentences(user_ans) == True:
        total_score = 0

    print("DUPLICATES:", check_duplicate_sentences(user_ans))
    return jsonify({'total_score': total_score})
    # return redirect(f'/?score={total_score}')
    # original_ans = request.form['original_ans']
    # user_ans = request.form['user_ans']
    # score = return_score(original_ans, user_ans)
    # score = score[0][0]
    # score = float(score)
    # # score = round(score * 10)
    # # if score < 0:
    # score = score * 10
    # if score <= 6.5:
    #     score = math.floor(score)
    # else:
    #     score = round(score)
    # return jsonify({'score': score})

    # return score

    # return redirect(f'/?score={score}')



if __name__ == '__main__':
    exit()
    app.run(port=4000)
