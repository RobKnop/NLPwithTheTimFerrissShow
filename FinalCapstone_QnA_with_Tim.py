
#%%
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 400)
pd.set_option('float_format', '{:.2f}'.format)
pd.set_option('isplay.max_colwidth', 300)
import re
import matplotlib.pyplot as plt
import IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import nltk
nltk.download('stopwords')
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns

#%%
# Import raw data
# This data set contains the transcripts of Tim Ferriss Podcasts
# Source: https://tim.blog/2018/09/20/all-transcripts-from-the-tim-ferriss-show/
df = pd.read_csv("raw/transcript-151-3XX.csv")
df.columns=['id', 'title', 'text']
df.head()

# Cleaning the titles
df['title'] = df['title'].str.replace(r"The Tim Ferriss Show Transcripts: ", "")
df['title'] = df['title'].str.replace(r"Transcripts: ", "")
df['title'] = df['title'].str.replace(r"Tim Ferriss Show Transcript: ", "")
df['title'] = df['title'].str.replace(r"Episode ", "")

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"\\n\\n'", "")
    df[text_field] = df[text_field].str.replace(r"\\n", " ")
    df[text_field] = df[text_field].str.replace(r"\\xa0", " ")
    df[text_field] = df[text_field].str.replace(r"xa0", " ")
    return df
df = standardize_text(df, "text")

#df.to_csv("./cleaned/transcripts_cleaned.csv") # Just save it

#%%
regexp_tokenizer = RegexpTokenizer(r'\w+')
word_tokens = df["text"].apply(regexp_tokenizer.tokenize)

# Inspecting our dataset a little more
all_words = [word for tokens in word_tokens for word in tokens]
sentence_lengths = [len(tokens) for tokens in word_tokens]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max episode length is %s" % max(sentence_lengths))
print("Mean episode length is %s" % np.mean(sentence_lengths))

label = df["title"].tolist()   
def plot_LSA(test_data, test_labels, plot=True):
        lsa = PCA(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        if plot:
            x = lsa_scores[:,0]
            y = lsa_scores[:,1]
            plt.scatter(x, y, s=10, alpha=1)
            for i,label in enumerate(test_labels):
                plt.text(x[i], y[i], label, fontsize=14)

#%% [markdown]
## Topic Modeling
#%%
from custom_stopwords import remove_custom_stopwords

df_topic = df.copy()
df_topic['text'] = df_topic['text'].str.lower()

# Creating the tf-idf matrix.
vectorizer = TfidfVectorizer(stop_words=set(stopwords.words('english')), min_df=3, norm='l2')
podcasts_tfidf = vectorizer.fit_transform(remove_custom_stopwords(df_topic, 'text').text)


# Getting the word list.
terms = vectorizer.get_feature_names()

# Number of topics.
ntopics=4

# Linking words to topics
def word_topic(tfidf,solution, wordlist):
    
    # Loading scores for each word on each topic/component.
    words_by_topic=tfidf.T * solution

    # Linking the loadings to the words in an easy-to-read way.
    components=pd.DataFrame(words_by_topic,index=wordlist)
    
    return components

# Extracts the top N words and their loadings for each topic.
def top_words(components, n_top_words):
    topwords=pd.Series()
    for column in range(components.shape[1]):
        # Sort the column so that highest loadings are at the top.
        sortedwords=components.iloc[:,column].sort_values(ascending=False)
        # Choose the N highest loadings.
        chosen=sortedwords[:n_top_words]
        #print(chosen)
        # Combine loading and index into a string.
        for i, data in enumerate(chosen):
            chosenlist=chosen.index.values.tolist()[i] +" "+ str(round(data ,2))
            topwords = topwords.append(pd.Series([chosenlist], index=[column]))
    return(topwords)

# Number of words to look at for each topic.
n_top_words = 10

# LSA
svd = TruncatedSVD(n_components=ntopics)
lsa_pipe = make_pipeline(svd, Normalizer(copy=False))
podcasts_lsa = lsa_pipe.fit_transform(podcasts_tfidf)

components_lsa = word_topic(podcasts_tfidf, podcasts_lsa, terms)

topwords=pd.DataFrame()
topwords['LSA']=top_words(components_lsa, n_top_words) 

# LDA
from sklearn.decomposition import LatentDirichletAllocation as LDA

lda = LDA(n_components=ntopics, 
          doc_topic_prior= 1/ntopics,
          topic_word_prior=1/ntopics,
          learning_decay=0.6, # Convergence rate.
          learning_offset=5.0, # Causes earlier iterations to have less influence on the learning
          max_iter=100, # when to stop even if the model is not converging (to prevent running forever)
          evaluate_every=3, # Do not evaluate perplexity, as it slows training time.
          mean_change_tol=0.001, # Stop updating the document topic distribution in the E-step when mean change is < tol
          max_doc_update_iter=100, # When to stop updating the document topic distribution in the E-step even if tol is not reached
          n_jobs=-1, # Use all available CPUs to speed up processing time.
          verbose=0, # amount of output to give while iterating
          random_state=32
         )

podcasts_lda = lda.fit_transform(podcasts_tfidf) 

components_lda = word_topic(podcasts_tfidf, podcasts_lda, terms)

topwords['LDA']=top_words(components_lda, n_top_words)

# NNMF

from sklearn.decomposition import NMF

nmf = NMF(alpha=0.1, 
          init='nndsvdar', # how starting value are calculated
          l1_ratio=0.5, # Sets whether regularization is L2 (0), L1 (1), or a combination (values between 0 and 1)
          max_iter=400, # when to stop even if the model is not converging (to prevent running forever)
          n_components=ntopics, 
          random_state=65, 
          solver='cd', # Use Coordinate Descent to solve
          tol=0.0001, # model will stop if tfidf-WH <= tol
          verbose=0 # amount of output to give while iterating
         )
podcasts_nmf = nmf.fit_transform(podcasts_tfidf) 

components_nmf = word_topic(podcasts_tfidf, podcasts_nmf, terms)

topwords['NNMF']=top_words(components_nmf, n_top_words)

#Show topics
for topic in range(ntopics):
    print('Topic {}:'.format(topic))
    print(topwords.loc[topic])

#%%
#Load distil BERT Model
nlp = spacy.load('/usr/local/lib/python3.7/site-packages/en_pytt_xlnetbasecased_lg/en_pytt_xlnetbasecased_lg-2.1.1')
n_items = 86

#%%
def get_embeddings_with_spacy(tokenizer, df):
    embeddings = df['text'].apply(lambda x: tokenizer(x).vector)
    return list(embeddings)

spacy_distil_bert_embeddings = get_embeddings_with_spacy(nlp, df[:n_items])

#%%
df_emb_spacy = pd.DataFrame(spacy_distil_bert_embeddings)

#%%
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(df_emb_spacy) 
df_emb_spacy.loc[:,:] = scaled_values

#%%
# Clustering 

# Calculate predicted values.
km = KMeans(n_clusters=4, random_state=42).fit(df_emb_spacy)
y_pred = km.predict(df_emb_spacy)

print('silhouette score: ', metrics.silhouette_score(df_emb_spacy, y_pred, metric='euclidean'))

# 2D
lsa = PCA(n_components=2)
las_results = lsa.fit_transform(df_emb_spacy.values)
las_results = pd.DataFrame(las_results, columns=['x', 'y'])

df_y = pd.DataFrame(y_pred, columns=['y_pred'])
df_y['y_pred'] = df_y['y_pred'].astype(int)

las_results = pd.concat([las_results, df_y], axis=1)

#Plot
sns.set_style("white")
plt.figure(figsize=(10, 13))  
fig = sns.scatterplot(x=las_results['x'], y=las_results['y'], hue=las_results['y_pred'])
for i, txt in enumerate(label[:n_items]):
    fig.annotate(txt, # this is the text
                 (las_results['x'].values[i],las_results['y'].values[i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,3), # distance from text to points (x,y)
                 ha='left', # horizontal alignment can be left, right or center
                 size=3)
fig.set_title("KMeans Cluster - Fig. 1", fontsize=12)
plt.show()


#%% 
#QnA
df['paragraphs'] = df['text'].apply(lambda x: x.split("''Tim Ferriss:") if len(x.split("''Tim Ferriss:")) > 7 else None)

# Define X
df_X = df.drop(columns=['text', 'id']).dropna()
df_X.reset_index(drop=True, inplace=True)

# Generate squad v1.1 json file so you can create annotated question and answers to fine-tune the model and evaluate it
from cdqa.utils.converters import df2squad
json_data = df2squad(df=df_X, squad_version='v1.1', output_dir='.', filename='qna_tim_ferriss')

#%%
from cdqa.pipeline.cdqa_sklearn import QAPipeline

# Load fine-tuned model 
cdqa_pipeline = QAPipeline(model='./cdqa/bert_qa_vCPU-sklearn.joblib', max_answer_length=60)
cdqa_pipeline.fit_retriever(X=df_X)

#%%
prediction = cdqa_pipeline.predict(X='How many deaths have you witnessed?')
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))
print('answer: {}'.format(prediction[0]))

#%%
prediction = cdqa_pipeline.predict(X='What do you mean by the idea of art?')
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))
print('answer: {}'.format(prediction[0]))


#%%
prediction = cdqa_pipeline.predict(X='what would be a good gymnastic strength training goal to have?')
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))
print('answer: {}'.format(prediction[0]))

#%%
# Evaluate QnA system
from cdqa.utils.evaluation import evaluate_pipeline
evaluate_pipeline(cdqa_pipeline, 'cdqa-v1.1-tim_qna.json')

#{'exact_match': 0.0, 'f1': 5.025362668068075}
#{'exact_match': 0.0, 'f1': 5.684362620078064}

#%%
# Fine-tune Bert model with squad v1.1 custom data set of Tim Ferriss questions
import os
import torch
from sklearn.externals import joblib
from cdqa.reader.bertqa_sklearn import BertProcessor, BertQA

train_processor = BertProcessor(do_lower_case=True, is_training=True)
train_examples, train_features = train_processor.fit_transform(X='cdqa-v1.1-tim_qna.json')

reader = BertQA(train_batch_size=12,
                learning_rate=3e-5,
                num_train_epochs=2,
                do_lower_case=True,
                output_dir='models')

reader.fit(X=(train_examples, train_features))

# Output fine-tuned model
reader.model.to('cpu')
reader.device = torch.device('cpu')
joblib.dump(reader, os.path.join(reader.output_dir, 'bert_tim_qa_vCPU.joblib'))

#%%
# Fix issue where the id is missing
import uuid
import json

with open('cdqa-v1.1-xxxx.json') as json_file:
    annotated_dataset = json.load(json_file)

for doc in annotated_dataset['data']:
    for p in doc['paragraphs']:
        for qa in p['qas']:
            qa['id'] = str(uuid.uuid4())

with open('cdqa-v1.1-tim_qna.json', 'w') as outfile:
    json.dump(annotated_dataset, outfile)

#%%
