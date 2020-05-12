# Importing basic libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import string
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import glob
import cufflinks as cf # cufflinks to link plotly to pandas and add the iplot method
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

#Importing Language Model Libraries
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25, BM25L, BM25Plus
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import CoherenceModel
from gensim import corpora
from pprint import pprint
from gensim.models.ldamulticore import LdaMulticore
import os
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel

from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams

additional_stop_words = ['porsche,' 'mercede','comfortsport', 'mercedes','mercedes-benz', 'honda','toyota','audi', 'benz','bentley','lexus',
                  'nissan','volvo','drive','nt','vehicle','infiniti','miles','corvette','come','edmund','lotus','diego','snake',
                 'porsche', 'cayman','bought','year','minute','chicago','car','home', 'think','suv','people','edmunds',
                  'cabriolet','lexuss','japan','husband','baby','range', 'rover','cadillac','cadillacs','michelin','texas',
                   'awsome','one','now', 'take', 'give', 'new','levinson','road','sedan','wife','sport','bang','tank',
                   'truck','lemon','imho','pathfinder','infinity','convertible','allroad','conv','bike','ski','grocery','mclass'
                  ,'hardtop','club','hubby','child','zoom','etc','brain','ashamed','carmax','alpina','rocketship','germany',
                  'autobahn','mercedez','000','great','good','just','ve','like','--','!!','\'','mustang','ive','gt','lt']

stop_words = text.ENGLISH_STOP_WORDS.union(additional_stop_words)

base_path = 'edmundsconsumer-car-ratings-and-reviews/'
reviewDocs = glob.glob(base_path + '*.csv')


result= pd.DataFrame()

for i in reviewDocs:
    temp = pd.read_csv(i, engine='python', index_col=False, encoding = 'utf8')
    result = result.append(temp)

#drop NA
result = result.dropna()
#drop unnamed column
result = result.drop(['Unnamed: 0'], axis = 1)

#split vehicle name column to year, make and model columns
result['year'] = result.Vehicle_Title.str.split(' ').apply(lambda x:x[0])
result['make'] = result.Vehicle_Title.str.split(' ').apply(lambda x:x[1])
result['model'] = result.Vehicle_Title.str.split(' ').apply(lambda x:x[2])

#round Customer Rating to one decimal places
result['Rating'] = result['Rating'].apply(lambda x: round(x, 1))

#concatenate Review Title and Review Text
result['EntireReview'] = result["Review_Title"].map(str) + result["Review"]

result.to_csv('CarReviews.csv',index=False)
carReviews = pd.read_csv('CarReviews.csv')

def lematized_review(text): # text
    rev_text = nlp(text)
    rev_text = ([token.lemma_.lower() for token in rev_text if not token.is_stop and token.text not in stop_words and not token.is_punct and len(token.text) > 3])
    return rev_text

def preprocess(review):
    # converts to lower case
    review = review.lower()
    # removes URLs
    review = review.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b','')
    review = review.replace(r'www\.\S+\.com','')
    # removes user mention
    review = review.replace(r'@\S+', '')
    # removes html tags
    review = review.replace(r'<.*?>', '')
    # removes extra spaces
    review = review.replace(r' +', ' ')
    # removes punctuations
    review = review.replace('[{}]'.format(string.punctuation), '')
    review = review.translate(str.maketrans('','', string.punctuation))
    review = review.strip()
    return review

carReviews['CleanReview'] = carReviews['EntireReview'].apply(lambda x:preprocess(x))
carReviews['ReviewTokens'] = carReviews['CleanReview'].apply(lambda x:lematized_review(x))


#BM25 Okapi to retrieve top N most relevant reviews based on given query
bm25okapi_index = BM25Okapi(list(carReviews.ReviewTokens))
query = ['mileage','car']
n_most_relevant = 5

scores = bm25okapi_index.get_scores(query)
top_n_indices = np.argsort(scores)[::-1][:n_most_relevant]
top_n_results = carReviews.iloc[top_n_indices, :]
top_n_results['score'] = scores[top_n_indices]


#Vader Sentiment Analysis
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score

carReviews['Negative_Score'] = carReviews['CleanReview'].apply(lambda x:sentiment_analyzer_scores(x)['neg'])
carReviews['Positive_Score'] = carReviews['CleanReview'].apply(lambda x:sentiment_analyzer_scores(x)['pos'])
carReviews['Neutral_Score'] = carReviews['CleanReview'].apply(lambda x:sentiment_analyzer_scores(x)['neu'])

carReviews['Vader_Rating'] = carReviews.apply(lambda x:((x.Positive_Score+x.Neutral_Score)*5), axis=1)

print('3 random reviews with the highest Positive sentiment polarity: \n')
pos = carReviews.loc[carReviews.Vader_Rating >= 4.5, ['EntireReview']].sample(3).values
for p in pos:
    print('------>',p[0])

print('3 random reviews with the highest Negative sentiment polarity: \n')
neg = carReviews.loc[carReviews.Vader_Rating <= 2.5, ['EntireReview']].sample(3).values
for n in neg:
    print('------>',n[0])

#LDA Topic Modelling

#Approach 1
reviews = carReviews["ReviewTokens"]
dictionary = corpora.Dictionary(reviews)
#Term document frequency
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews]
#perform LDA
ldamodel = LdaMulticore(corpus= doc_term_matrix, num_topics =8, id2word=dictionary,chunksize=2000, passes=20,per_word_topics=True)

#get highlighted topics
topics = ldamodel.show_topics()
lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False)

#show HTML view
pyLDAvis.save_html(lda_display,open("lda_8_topics.html","w"))

pprint(ldamodel.show_topics(formatted=False))

# Calculate coherence score
def compute_coherence_score(lda_model,reviews):
    coherence = CoherenceModel(lda_model,texts = reviews,dictionary = dictionary ,coherence = "c_v")
    return coherence.get_coherence(),coherence.get_coherence_per_topic()

coh_score,coh_by_topic = compute_coherence_score(ldamodel,reviews)
print(coh_by_topic,coh_score)

#Approach 2

!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
!unzip mallet-2.0.8.zip

def install_java():
  !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk
  os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
  !java -version       #check java version
install_java()

os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
mallet_path = '/content/mallet-2.0.8/bin/mallet'

#create model
ldamallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=8, id2word=dictionary)

pprint(ldamallet.show_topics(formatted=False))
gensimmodel= gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)

#create wrapper for visualization
ldamallet_display = pyLDAvis.gensim.prepare(gensimmodel, doc_term_matrix, dictionary, sort_topics=False)
pyLDAvis.save_html(ldamallet_display,open("ldamallet_8_topics.html","w"))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=reviews, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\n Mallet Coherence Score: ', coherence_ldamallet)

#Generate Tags
def get_reviews_to_process(text):
    tokens = wordpunct_tokenize(text)
    filtered_sentence = []
    for t in tokens:
        if len(t)>1 and t not in stop_words and t not in string.punctuation:
            filtered_sentence.append(t)
    if len(filtered_sentence)>0:
        return filtered_sentence
    return []

def get_bigrams(tokens,n=10):
    bigrm = nltk.bigrams(tokens)
    bigrmLst = [' '.join(t) for t in bigrm]

    freq = 0
    tag_freq = {}
    for j in bigrmLst:
        if j in tag_freq:
            tag_freq[j] +=1
        else:
            tag_freq[j] = 1

    popular_tags = sorted(tag_freq, key = tag_freq.get, reverse = True)

    top_n = popular_tags[:n]
    topTags = [''.join(h) for h in top_n]
    topTagStr=';'.join(topTags)
    return topTagStr

def get_trigrams(tokens,n=10):
    trigrm = nltk.trigrams(tokens)
    trigrmLst = [' '.join(t) for t in trigrm]

    freq = 0
    tag_freq = {}
    for j in trigrmLst:
        if j in tag_freq:
            tag_freq[j] +=1
        else:
            tag_freq[j] = 1

    popular_tags = sorted(tag_freq, key = tag_freq.get, reverse = True)

    top_n = popular_tags[:n]
    topTags = [''.join(h) for h in top_n]
    topTagStr=';'.join(topTags)
    return topTagStr

review_by_model = dict()
for index,row in carReviews.iterrows():
    car = row['make']+' '+row['model']
    if car not in review_by_model.keys():
        review_by_model[car] = ''
    review_by_model[car] += row['CleanReview']+' '

model_bigrams = dict()
model_trigrams = dict()

for i,rev in carReviews.iterrows():
    car = rev['make']+' '+rev['model']
    if car not in model_bigrams.keys():
        model_bigrams[car] = get_bigrams(get_reviews_to_process(review_by_model[car]),10)
        model_trigrams[car] = get_trigrams(get_reviews_to_process(review_by_model[car]),10)
    carReviews.loc[i,'Model_Bigrams'] = model_bigrams[car]
    carReviews.loc[i,'Model_Trigrams'] = model_trigrams[car]
