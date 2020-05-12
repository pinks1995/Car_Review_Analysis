# DS 595: Information Retrieval & Social Web
## Car Review Analysis

### Introduction
The internet has put everything to our fingertips, from buying groceries to researching for a new house. Internet has now become a forum where the end user evaluates every product and service based on other like-minded user's experience about it. And the case is no longer like that in older days when consumer itself used and evaluated the product. Customer Reviews is considered unbiased and true quality of the product.
Car purchase has been simpler in earlier days: decide a budget, look among the limited options you have, take a test drive, understand car features and Ta-dah!! You have a new car. But then you start experiencing some issues with your car within a few months. May be the sales person hid some feature problems about the model.
For the same reason, buyer wants to go through customer reviews posted online about the product to make better decisions for his next automobile purchase.
However, the amount of reviews received for each product is huge and is cumbersome for any individual to be able to go through each of them and evaluate the product. Thus it becomes important for both customer and companies to be able to get a quick summary of the user experience so as to make better decisions of going with the product or not.

### Dataset
We used <a href="https://www.kaggle.com/ankkur13/edmundsconsumer-car-ratings-and-reviews">Kaggle dataset</a> for our analysis.
1. 62 major car brand
2. Columns: Id, Review Date, Author Name, Vehicle Title, Review Title, Review, Rating

### Environment Setup
* Python 3
* Basic Libraries:
  * numpy
  * pandas
  * string
  * matplotlib
  * seaborn
  * re
  * glob
  * cufflinks
  * os
  * pprint
* Language Model Libraries:
  * textblob
  * vaderSentiment
  * nltk
    * nltk.download('wordnet')
    * nltk.download('stopwords')
  * sklearn
  * spacy
    * python -m spacy download en_core_web_sm
  * tqdm
  * rank_bm25
  * pyLDAvis
  * gensim
  

### BM25 Review Search

### Review Sentiment Analysis
Review Text is a free content that allows users to express as much as they want. Rating is a quantitative measure of the goodness of the product or service. But does the rating define the quality the same way as much as the review text does?
It's important to analyze the review content and calculate the rating value based on that.

### LDA Topic Modelling


### Quick Tags
Having access to huge amount of reviews is surely helpful, however going through all reviews is cumbersome. We have generated quick tags (Bigrams & Trigrams) for each Car Model that provide a summary of the model in few words that will help.
