#Import packages
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import  CountVectorizer
from preprocessing.text_preprocessing import TextPreprocessor
from utils.datasplitter import get_verified_users
import requests
from top2vec import Top2Vec
from utils.data_collection import DataLoader
from tqdm import tqdm
import os


def retrieve_url_website(row, timeout = 1): 
    """
    Retrieve the correct url link from a shorten Twitter version.
    Args:
        row (str): the twitter text.
        timeout (int): waiting time before a time out error is launched.
    Returns:
        url_str (str): the complete and correct url link.
    """
    url_list = []
    url_tokens = [t  for t in row.split(' ') if 'http' in t]
    for url in url_tokens:
        try:
            r =  requests.get(url, timeout = timeout)
            url_list.append(r.url)
        except:  #No results with the 
            pass
    url_str= ' '.join(url_list)
    return url_str


def word_count_features(corpus_train,
                        corpus_test=None,
                        column='description',
                        vectorizer=CountVectorizer(token_pattern=r'[^\s]+', strip_accents = 'unicode')
                       ):
    '''
    Create a dataframe of word count features.
    Args:
        corpus_train (list): list of texts used as train set.
        corpus_test (list): list of texts used as test set. Ignored if no test set.
        column (str): the column corresponding to the corpus, used to the name the generated features. Either 'description' or 'text'.
        vectorizer (object): the vectorizer applied on the corpus. Can be any existing scikit-learn vectorizer (CountVectorizer,TfIdfVectorizer).
    Returns:
        train_output (pd.DataFrame): DataFrame containing the word count features as columns for the train set users
        test_output (pd.DataFrame): DataFrame containing the word count features as columns for the test set users. None if no test set.
    '''

    train_features =  vectorizer.fit_transform(corpus_train)
    test_features = vectorizer.transform(corpus_test) if corpus_test else None
    feature_names = [column + '  ' + w for w in vectorizer.get_feature_names_out().tolist()]
    train_output = pd.DataFrame(data=train_features.toarray(),columns=feature_names)
    test_output = pd.DataFrame(data=test_features.toarray(),columns=feature_names) if corpus_test else None
    print('BoW data '+column +' collected')
    return train_output, test_output


def topic_features(corpus_train,
                    corpus_test=None,
                    min_count_top2vec=50):
    '''
    Create a dataframe of Top2Vec topic features.
    Args:
        corpus_train (list): list of texts used as train set.
        corpus_test (list): list of texts used as test set. Ignored if no test set.
        min_count_top2vec (int):  Ignores all words with total frequency lower than this. For smaller corpora a smaller min_count will be necessary.
    Returns:
        train_topic (pd.DataFrame): DataFrame containing the topic features as columns for the train set users
        test_topic (pd.DataFrame): DataFrame containing the topic features as columns for the test set users. None if no test set.
    '''
    train_topic = pd.DataFrame()
    test_topic = pd.DataFrame()
    if len(corpus_train) <= 100:
    #Check that the corpus is sufficiently large
        print('Insuficient number of users to compute topic features with Top2Vec')
        return train_topic, test_topic
    model = Top2Vec(corpus_train,  
                workers = 8,
                speed = 'learn',
                keep_documents = False,
                verbose=False,
                min_count=min_count_top2vec)
    if not os.path.isdir('models'):
        os.makedirs('models')
    model.save('models/top2vec')
    if corpus_test is not None:
        model.add_documents(corpus_test)
    #First extract the size of each topic (the number of documents in the topic)
    topic_sizes, topic_nums = model.get_topic_sizes() 
    for i in topic_nums:  #For each topic create a feature
        _ , document_ids = model.search_documents_by_topic(topic_num=i, 
                                                                  num_docs= topic_sizes[i])
        scores_dict = {} #Create a dict with as key and 1 as value
        for j in document_ids:
            scores_dict[j] = 1
        for u in range(len(corpus_train)):  
            if u not in scores_dict.keys():
                scores_dict[u] = 0
        topic_feature = 'Topic ' + str(i)  
        train_topic[topic_feature] = [scores_dict[u] for u in range(len(corpus_train))] 
    if corpus_test is not None:
        for i in topic_nums:  #For each topic create a feature
            _ , document_ids = model.search_documents_by_topic(topic_num=i, 
                                                                    num_docs= topic_sizes[i])
            scores_dict = {} #Create a dict with as key and 1 as value
            for j in document_ids:
                scores_dict[j] = 1
            for u in range(len(corpus_train),len(corpus_train+corpus_test)):  
                if u not in scores_dict.keys():
                    scores_dict[u] = 0
            topic_feature = 'Topic ' + str(i)
            test_topic[topic_feature] = [scores_dict[u] for u in range(len(corpus_train),len(corpus_train+corpus_test))]
    print('topic data collected')
    return train_topic, test_topic


def celebrity_follower_feature(train_df,
                               test_df=pd.DataFrame(),
                               mentioned_user_df=pd.DataFrame(),
                               bearer_token=None,  
                               belgium_filter=True,
                               min_followers=10000,
                               max_followers=200000,
                               sleep=60   
                               ):
    '''
    Create a dataframe of celebrity_followers features.
    Args:
        train_df (pd.DataFrame): the training set.
        test_df (pd.DataFrame): the test  set. If no test set, an empty dataframe will represent it.
        mentioned_user_df (pd.DataFrame): dataframe containing the mentioned users. If not provided, an empty dataframe will represent it.
        bearer_token (str): the bearer token to access the Twitter API.
        belgium_filter (bool): if true, apply filters to remove non-belgian users from the celebrities list.
        min_followers (int): lower followers count threshold for celebrities to include.
        max_followers (int): upper followers count threshold for celebrities to include.
        sleep (int): waiting time between two API calls.  60 is recommended for large celebrities list to avoid exceeding the API rate limit.
    Returns:
        train_celebrity (pd.DataFrame): DataFrame containing the celebrity features as columns for the train set users
        test_celebrity (pd.DataFrame): DataFrame containing the celebrity features as columns for the test set users. None if no test set 

    '''
    #Collect all verified users
    verified = get_verified_users(pd.concat([train_df,test_df]),mentioned_user_df) if not test_df.empty else get_verified_users(train_df,mentioned_user_df)
    #Filter verified accounts that belong to people living in Belgium
    if belgium_filter:
        location_data = pd.read_csv("data/resources/location/city_names.csv")
        Antwerpen = set(location_data["Antwerpen"].str.lower())
        Limburg = set(location_data["Limburg"].str.lower())
        Oost_Vlaanderen = set(location_data["Oost_Vlaanderen"].str.lower())
        West_Vlaanderen = set(location_data["West_Vlaanderen"].str.lower())
        Vlaams_Brabant = set(location_data["Vlaams_Brabant"].str.lower())
        Brussel = set(location_data["Brussel"].str.lower())
        Nederlands = set(location_data["Nederland"].str.lower())
        be_keywords = {'belgie', 'belgium','belgique','belg','belge','belgian','vlaanderen','vlaams','flemish'}.union(Antwerpen, Limburg, Oost_Vlaanderen, West_Vlaanderen, Vlaams_Brabant, Brussel)                              
        nl_keywords = {'nederland','nederlande','netherlands'}.union(Nederlands)
        #This lambda function checks if at least one element of set b is also in set a
        any_in = lambda a, b : any(i in b for i in a)
        #Users with a keyword relative to belgium but no keywords relative to the Netherlands
        be_mask = verified.apply(lambda row : False if  any_in(nl_keywords, row["location_profile"].lower().split() +row["description"].lower().split()) 
                                        else  True if any_in(be_keywords, row["location_profile"].lower().split() + row["description"].lower().split())                                                                   
                                        else False, axis=1) 
        verified = verified[be_mask]

    #Remove verified users with too many or too few followers
    celebrities = verified[(verified["followers_count"] >= min_followers)  & (verified["followers_count"]<= max_followers)].reset_index(drop=True)
    #There is no pre-available file with the followers data, need to collect it
    #Load the followers data
    dl = DataLoader(bearer_token)
    screen_names=celebrities['screen_name'].to_list()
    celeb_dict = {}
    for name in screen_names:
        celeb_dict[name] = dl.retrieve_all_followers(name,sleep=sleep) #matrix with a list per users

    #Create the feature matrix
    train_celebrity=pd.DataFrame()
    test_celebrity=pd.DataFrame()
    for k in tqdm(celeb_dict.keys()):
        train_celebrity[k] = train_df['user_id'].apply(lambda row : 1 if row in celeb_dict[k] else 0).values
        test_celebrity[k] = test_df['user_id'].apply(lambda row : 1 if row in celeb_dict[k] else 0).values if not test_df.empty else None 
    print('celebrity data collected')
    return train_celebrity, test_celebrity


def tweet_metadata_features(tweet_df):
    '''
    Create a dataframe of tweet metadata features.
    Args:
        tweet_df (pd.DataFrame): the dataframe containing the tweet data.
    Returns:
        tweet_metadata (pd.DataFrame): a dataframe containing the tweet metadata.
    '''
    tweet_metadata=pd.DataFrame()
    tweet_metadata['text_length'] = tweet_df['text'].apply(lambda row : len(row))
    tweet_metadata['created_at_hour'] = tweet_df['created_at'].apply(lambda row : datetime.strptime(row[:-6],"%Y-%m-%d %H:%M:%S").hour)
    tweet_metadata['day_of_the_week'] = tweet_df['created_at'].apply(lambda row : datetime.strptime(row[:-6],"%Y-%m-%d %H:%M:%S").strftime('%A'))
    tweet_metadata['day_period'] = tweet_metadata['created_at_hour'].apply(lambda row : 'Morning' if row in [5,6,7,8,9,10,11,12] 
                                                                            else 'Afternoon' if row in [13,14,15,16,17,18,19,20] 
                                                                            else 'Night')
    print('tweet metadata collected')
    return tweet_metadata


def user_metadata_features(user_df,
                      current_year=2022):                                                                       
    '''
    Create a dataframe of user metadata features.
    Args:
        user_df (pd.DataFrame): the dataframe containing the user data.
        current_year (int): used to compute the account age, relatively to the current year.
    Returns:
        user_metadata (pd.DataFrame): a dataframe containing the user metadata.
    '''
    user_metadata=pd.DataFrame()
    user_metadata['account_age'] = user_df['account_created_at'].apply(lambda row : current_year - int(row[0:4]))
    #URL features
    url_str = user_df['description'].fillna('').apply(lambda row : retrieve_url_website(row))                                                         
    user_metadata['BE_domain'] = url_str.fillna('').apply(lambda row : 1 if '.be' in row else 0)
    user_metadata['NL_domain'] = url_str.fillna('').apply(lambda row : 1 if '.nl' in row else 0)
    user_metadata['instagram_url'] = url_str.fillna('').apply(lambda row : 1 if 'www.instagram.com' in row else 0)
    user_metadata['facebook_url'] = url_str.fillna('').apply(lambda row : 1 if 'facebook.com' in row else 0)
    user_metadata['discord_url'] = url_str.fillna('').apply(lambda row : 1 if 'discord.com' in row else 0)
    user_metadata['linkedin_url'] = url_str.fillna('').apply(lambda row :1 if 'linkedin.com' in row else 0)
    user_metadata['youtube_url'] = url_str.fillna('').apply(lambda row : 1 if 'www.youtube.com' in row else 0)
    user_metadata['twitch_url'] = url_str.fillna('').apply(lambda row :1 if 'twitch.tv' in row else 0)   
    print('user metadata collected')
    return  user_metadata


def aggregate_tweet(tweet_df):
    '''
    Create tweet features by aggregating the tweet dataframe according to user ids.
    Args:
        tweet_df (pd.DataFrame): the dataframe containing the tweet data.
    Returns:
        aggregated_tweet_df (pd.DataFrame): a dataframe containing the aggregated tweet data
    '''
    aggregation_dict = {'source': lambda x: x.value_counts().index[0],  #most frequent source value
                    'day_period': lambda x: x.value_counts().index[0],  #Most frequent period of the day
                    'day_of_the_week': lambda x: x.value_counts().index[0], #Most frequent day of the week
                    'reply_settings':lambda x: x.value_counts().index[0], #most frequent reply settings
                    'possibly_sensitive': 'sum',   #count the number of possibly sensitive tweets written by the user (boolean)
                    'tweet_mentions': lambda x : ','.join(x),   #string with all mentions
                    'text': lambda x : '  <--->  '.join(x), #string of all tweets. <---> separates tweets
                    'tweet_id': 'count'  #count the number of distinct tweets
                    }

    col_new_names = {'tweet_id':'tweet_count_dataset',
                 'source':'main_source',
                 'day_of_the_week':'favorite_day',
                 'day_period':'favorite_period',
                 'reply_settings':'main_reply_settings',
                 }

    aggregated_tweet_df = tweet_df.groupby("user_id").agg(aggregation_dict).rename(columns = col_new_names)
    #This feature has 69 categories. We reduce  to 6 main categories + "Other" category
    aggregated_tweet_df['main_source'] = aggregated_tweet_df['main_source'].apply(lambda row : row if row in ['Twitter for iPhone', 'Twitter for Android', 
                                                                                            'Instagram', 'Foursquare','Twitter Web Client', 
                                                                                            'Twitter for iPad'] else 'Other')
    return aggregated_tweet_df


def create_feature_matrix(tweet_df, train_df, test_df=pd.DataFrame(), mentioned_user_df=None, bearer_token = None,
                          belgium_filter=True,min_followers=10000,max_followers=200000, min_count_top2vec=50,
                          features=['bow','topic','metadata','celebrity']):
    '''
    Create a feature matrix.
    Args:
        tweet_df (pd.DataFrame): the dataframe containing the tweet data.
        train_df (pd.DataFrame): the training set.
        test_df (pd.DataFrame): the test  set. If no test set, an empty dataframe will represent it.
        mentioned_user_df (pd.DataFrame): dataframe containing the mentioned users. If not provided, an empty dataframe will represent it.
        bearer_token (str): the bearer token to access the Twitter API.
        belgium_filter (bool): if true, apply filters to remove non-belgian users from the celebrities list.
        min_followers (int): lower followers count threshold for celebrities to include.
        max_followers (int): upper followers count threshold for celebrities to include.
        min_count_top2vec (int):  Ignores all words with total frequency lower than this. For smaller corpora a smaller min_count will be necessary.
        features (list): list of feature types to include. Possible feature types : 'bow','topic','metadata','celebrity'. 
                         bow and metadata features can be collected in all circumstances. topic features require a dataset larger than 100 users.
                         celebrity features are accessible for researchers with a valid bearer token to access the Twitter API.
        Returns:
            feature_matrix_train (pd.DataFrame): the feature matrix for the train set.
            feature_matrix_test (pd.DataFrame): the feature matrix for the test set.
    '''

    feature_matrix_train = train_df['user_id'].reset_index(drop=True)
    if not test_df.empty:
        feature_matrix_test = test_df['user_id'].reset_index(drop=True)

   
    if 'bow' in features:
        preprocessor = TextPreprocessor()
        corpus_train_desc = preprocessor.preprocess(train_df['description'])
        corpus_test_desc = preprocessor.preprocess(test_df['description']) if not test_df.empty else None
        profile_bow_train, profile_bow_test = word_count_features(corpus_train_desc,corpus_test_desc)
        feature_matrix_train = pd.concat([feature_matrix_train,profile_bow_train],axis=1)
        feature_matrix_test = pd.concat([feature_matrix_test,profile_bow_test],axis=1) if not test_df.empty else None

    #Metadata
    if 'metadata' in features:
        user_metadata_train = user_metadata_features(train_df)
        user_metadata_test = user_metadata_features(test_df) if not test_df.empty else None
        feature_matrix_train = pd.concat([feature_matrix_train,user_metadata_train.reset_index(drop=True)],axis=1)
        feature_matrix_test = pd.concat([feature_matrix_test,user_metadata_test.reset_index(drop=True)],axis=1) if not test_df.empty else None
        tweet_metadata = tweet_metadata_features(tweet_df) 
        aggregated_tweet = aggregate_tweet(pd.concat([tweet_df,tweet_metadata],axis=1))
    else:
        aggregated_tweet=aggregate_tweet(tweet_df)
    
    #Split aggregated tweets in train and test set and remove unnecessary columns
    aggregated_tweet_train = aggregated_tweet[aggregated_tweet.index.isin(feature_matrix_train.user_id)]
    aggregated_tweet_test = aggregated_tweet[aggregated_tweet.index.isin(feature_matrix_test.user_id)] if not test_df.empty else None
    feature_matrix_train = pd.concat([feature_matrix_train,aggregated_tweet_train.reset_index(drop=True)],axis=1)
    feature_matrix_test = pd.concat([feature_matrix_test,aggregated_tweet_test.reset_index(drop=True)],axis=1) if not test_df.empty else None
    

    if 'bow' in features:
        preprocessor = TextPreprocessor()
        corpus_tweet_train = preprocessor.preprocess(aggregated_tweet_train['text'],remove_keywords=False)
        corpus_tweet_test = preprocessor.preprocess(aggregated_tweet_test['text'],remove_keywords=False) if not test_df.empty else None
        tweet_bow_train, tweet_bow_test = word_count_features(corpus_tweet_train,corpus_tweet_test,column='text')
        feature_matrix_train = pd.concat([feature_matrix_train,tweet_bow_train],axis=1)
        feature_matrix_test = pd.concat([feature_matrix_test,tweet_bow_test],axis=1) if not test_df.empty else None
    
    if 'topic' in features:
        preprocessor = TextPreprocessor()
        corpus_tweet_train = preprocessor.preprocess(aggregated_tweet_train['text'],remove_keywords=False)
        corpus_tweet_test = preprocessor.preprocess(aggregated_tweet_test['text'],remove_keywords=False) if not test_df.empty else None
        tweet_topic_train, tweet_topic_test = topic_features(corpus_tweet_train,corpus_tweet_test,min_count_top2vec=min_count_top2vec)
        feature_matrix_train = pd.concat([feature_matrix_train,tweet_topic_train],axis=1)
        feature_matrix_test = pd.concat([feature_matrix_test,tweet_topic_test],axis=1) if not test_df.empty else None


    if 'celebrity' in features:
        celebrity_train, celebrity_test =  celebrity_follower_feature(train_df,test_df,mentioned_user_df,
                                                                      bearer_token,belgium_filter,
                                                                     min_followers,max_followers)
        feature_matrix_train = pd.concat([feature_matrix_train,celebrity_train],axis=1)
        feature_matrix_test = pd.concat([feature_matrix_test,celebrity_test],axis=1) if not test_df.empty else None
    
    feature_matrix_train = feature_matrix_train.drop(columns=['tweet_mentions','text','tweet_count_dataset'])
    feature_matrix_test = feature_matrix_test.drop(columns=['tweet_mentions','text','tweet_count_dataset']) if not test_df.empty else None
    if not test_df.empty:
        return feature_matrix_train, feature_matrix_test
    else :
        return  feature_matrix_train
