import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import download

class TextPreprocessor:
    def __init__(self,
                 tokenizer=TweetTokenizer(),
                 stopwords_languages=['english','dutch','french'],
                 gender_keywords_path='data/resources/gender/GenderKeywords.csv',
                 age_keywords_path='data/resources/age/AgeKeywords.csv',
                 location_keywords_path='data/resources/location/city_names.csv'):
        '''
        Preprocess the text to the correct format for subsequent tasks.
        Params:
            tokenizer (object): nltk tokenizer used to tokenize the text. 
            stopwords_languages (list): list of languages for which stopwords should be downloaded.
            gender_keywords_path (str): path to the file containing gender keywords.
            age_keywords_path (str): path to the file containing age keywords.
            location_keywords_path (str): path to the file containing location keywords.
        '''
        self.tokenizer = tokenizer
        #Stopwords
        download('stopwords',quiet=True)
        self.sw = []
        for l in stopwords_languages:
            self.sw += stopwords.words(l)
        #Keywords (add keywords for age and location) (find efficient way to code it)
        gender = pd.read_csv(gender_keywords_path).fillna(' ')
        age = pd.read_csv(age_keywords_path).fillna(' ')
        location = pd.read_csv(location_keywords_path).fillna(' ')
        self.keywords = list(np.ravel(gender.values))+ list(np.ravel(age.values))+ list(np.ravel(location.applymap(lambda s:s.lower() if type(s) == str else s).values)) 
        self.keywords = [k for k in self.keywords if k != ' ' ]
 
    def remove_punctuation(self,tweet):
        #Remove all punctuation except # and @
        punctuation = "!\"$%&'()*+,-?.../:;<=>[\]^_`{|}~"  
        tweet = [''.join([char for char in word if char not in punctuation])  for word in tweet]
        return tweet

    def remove_stopwords(self,tweet,):
        #Remove stopwords
        tweet = [word for word in tweet if word not in self.sw]
        return tweet

    def remove_hyperlinks(self,tweet):
        #Remove hyperlinks from the profile description
        tweet = [word for word in tweet if 'http' not in word]
        return tweet

    def remove_digits(self,tweet):
        #Remove all numbers from the profile description
        tweet = [word for word in tweet if not word.isdigit()]
        return tweet

    def remove_snorkel_keywords(self,tweet):
        #Remove all keywords from the profile description
        tweet = [word for word in tweet if word not in self.keywords]
        return tweet

    def preprocess(self,
                   corpus,
                   remove_keywords=True):
        '''
        Apply a complete preprocessing pipeline to a text corpus.
        Args:
            corpus (pd.Series): corpus to be preprocessed.
            remove_keywords (bool): if True, removes snorkel keywords from the text corpus.
        Returns:
            corpus (list): list of processed strings.
        '''
        corpus = corpus.fillna('')
        corpus = corpus.str.lower()
        corpus = [self.tokenizer.tokenize(tweet) for tweet in corpus]
        corpus = [self.remove_punctuation(tweet) for tweet in corpus]
        corpus = [self.remove_stopwords(tweet) for tweet in corpus]
        corpus = [self.remove_hyperlinks(tweet) for tweet in corpus] 
        corpus = [self.remove_digits(tweet) for tweet in corpus]
        if remove_keywords:
            corpus = [self.remove_snorkel_keywords(tweet) for tweet in corpus]
        corpus = [' '.join(tweet) for tweet in corpus]
        return corpus
