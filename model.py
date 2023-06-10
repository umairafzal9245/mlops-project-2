import datetime
import mlflow
from datetime import datetime
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import numpy as np
from sklearn.model_selection import train_test_split
import os 
np.random.seed(2018)

nltk.download('wordnet')


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text, pos='v')


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize(token))
    return result

data = pd.read_csv('./myDataset.csv', on_bad_lines='skip')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


data_text = train_data[['Answers']]
data_text['index'] = data_text.index
documents = data_text

processed_docs = documents['Answers'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


#########################################################################
test_processed_docs = test_data['Answers'].map(preprocess)

test_bow_corpus = [dictionary.doc2bow(doc) for doc in test_processed_docs]

##########################################################################

experiment_name = "Experiment - " + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_id = mlflow.create_experiment(experiment_name)


for idx,topic in enumerate([10]):

    lda_model = gensim.models.LdaModel(bow_corpus, num_topics=topic, id2word=dictionary, passes=2)

    if os.path.exists('./model') == False:
        os.mkdir('./model')

    lda_model.save('./model/lda_model.model')

    coherence_model = gensim.models.CoherenceModel(model=lda_model, corpus=test_bow_corpus, dictionary=dictionary, coherence='u_mass')

    coherence_score = coherence_model.get_coherence()

    run_name = f"num_topics_{topic}_passes_{2}"

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:

        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_param("Topics", topic)

        # Track metrics
        mlflow.log_metric("coherence_score", coherence_score)

        # Track model
        mlflow.sklearn.log_model(lda_model, "lda_model.model")
