'''
This data collection script is largely inspired from the excellent tutorial of Andrew Edward : 
https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a
'''

#Import packages
import requests
import json
import time
import dateutil.parser
import pandas as pd
from tqdm import tqdm

class DataLoader:
    '''
    Collects data from Twitter Academic API endpoints and provides them as json or csv files. Requires a bearer token for authentication.
    '''
    def __init__(self,
                 bearer_token):
        self.bearer_token = bearer_token
        self.headers = {"Authorization": "Bearer {}".format(self.bearer_token)}
        
    def create_tweet_query(self,
                     start_date='2020-01-01 T00:00:00.000Z', 
                     end_date='2020-02-01 T00:00:00.000Z',
                     keyword="place_country:BE has:geo lang:nl",
                     max_results = 500):
        '''
        Initialize query parameters for the Twitter Full Archive Search endpoint.
        Args:
            start_date (str): the start time of the period. It needs to be a valid timestamp.
            end_data (str): the end time of the period. It needs to be a valid timestamp.
            keyword (str): the query parameters to refine the tweet search. See the Twitter API documentation for more information on queries.
            max_results (int): maximum number of tweets to retrieve for the desired period
        Returns:
            query_params (dict): dictionary containing all parameters for the Archive Search query.            
        '''
        query_params = {'query': keyword,
                        'start_time': start_date,
                        'end_time': end_date,
                        'max_results': max_results,
                        'expansions': 'author_id,in_reply_to_user_id,geo.place_id,referenced_tweets.id',
                        'tweet.fields': 'id,text,author_id,context_annotations,geo,created_at,lang,public_metrics,entities,reply_settings,possibly_sensitive,source',
                        'user.fields': 'id,name,username,created_at,description,location,public_metrics,verified,entities,profile_image_url',
                        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                        'next_token': {}}
        return query_params

    def create_follower_query(self,
                              screen_name,
                              cursor=None):
        '''
        Initialize query parameters for the Twitter followers ids API endpoint.
        Args:
            screen_name (str): screenname of the user for which the followers ids need to be retrieved.
            cursor (int): cursor to the next batch of followers ids.
        Returns:
            query_params (dict): dictionary containing all parameters for the Archive Search query.            
        '''
        query_params = {'screen_name': screen_name,
                    'cursor': cursor}
        return query_params

    def connect_to_endpoint(self,
                            search_api,
                            query_params, 
                            next_token = None):
        '''
        Establish connection with the Twitter API endpoint.
        '''
        query_params['next_token'] = next_token   
        response = requests.request("GET", search_api, headers = self.headers, params = query_params)
        print("Endpoint Response Code: " + str(response.status_code))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


    def retrieve_tweet(self,
                       start_list, 
                       end_list,
                       keyword="place_country:BE has:geo lang:nl",  
                       tweet_per_period = 10000):
        '''
        Retrieve tweets from the Full Archive Search, following a specific query, for a given number of tweets per time periods.
        Args:
            start_list (list): list containing the start times for each time period. Needs to have the same length as end_list.
            end_list (list): list containing the end times for each time period. Needs to have the same length as start_list.
            keyword (str):  the query parameters to refine the tweet search. See the Twitter API documentation for more information on queries.
            tweet_per_period (int): the maximum number of tweets to retrieve per time period.
        Returns:
            tweet_list (list): list containing all the Twitter response objects.
        '''
        tweets_list = []  
        #Define the search API : Twitter Full Archive Search
        search_api = "https://api.twitter.com/2/tweets/search/all"
        #Total number of tweets we collected from the loop
        total_tweets = 0
        #Ensures that we collect less than or equal to the desired number of tweets
        #Define the number of results per query. The API accepts a maximum of 500 tweets per query.
        if tweet_per_period >= 500:
            max_results=500
        else:
            max_results=tweet_per_period
        for i in range(0,len(start_list)):
            count = 0 # Counting tweets per time period
            #Max tweets per period is set a bit lower than the real objective
            flag = True
            next_token = None
       
            while flag:
                # Check if max_count reached
                if count >= tweet_per_period:
                    break
                print("-------------------")
                print("Token: ", next_token)
                query = self.create_tweet_query(start_list[i],end_list[i],keyword,max_results)
                json_response = self.connect_to_endpoint(search_api, query, next_token)
                result_count = json_response['meta']['result_count']

                if 'next_token' in json_response['meta']:
                    # Save the token to use for next call
                    next_token = json_response['meta']['next_token']
                    print("Next Token: ", next_token)
                    if result_count is not None and result_count > 0 and next_token is not None:
                        print("Start Date: ", start_list[i])
                        tweets_list.append(json_response)
                        count += result_count
                        total_tweets += result_count
                        print("Total # of Tweets added: ", total_tweets)
                        print("-------------------")
                        time.sleep(3)

                # If no next token exists
                else:
                    if result_count is not None and result_count > 0:
                        print("-------------------")
                        print("Start Date: ", start_list[i])
                        tweets_list.append(json_response)
                        count += result_count
                        total_tweets += result_count
                        print("Total # of Tweets added: ", total_tweets)
                        print("-------------------")
                        time.sleep(3)

                    #Since this is the final request, turn flag to false to move to the next time period.
                    flag = False
                    next_token = None
                time.sleep(3)
        print("Total number of results: ", total_tweets)
        
        return tweets_list

    
    def retrieve_all_followers(self,screen_name,sleep=60):
        '''
        Retrieve all the followers of a given user.
        Args:
            screen_name (str): the screenname of the user.
            sleep (int): waiting time between two queries to avoid exceeding the API call limits.
        Returns:
            followers (list): list of all followers ids.
        '''
        search_api = "https://api.twitter.com/1.1/followers/ids.json"
        followers = []
        flag = True
        next_cursor = None
        count = 0
        while flag:
            print("-------------------")
            print("Cursor: ", next_cursor)
            query = self.create_follower_query(screen_name,next_cursor)
            json_response = self.connect_to_endpoint(search_api,query, next_cursor)
            result_count = len(json_response['ids'])

            if json_response['next_cursor'] != 0 :
                # Save the cursor to use for next call
                next_cursor = json_response['next_cursor']
                print("Next cursor: ", next_cursor)
                if result_count is not None and result_count > 0 and next_cursor is not 0:
                    followers += json_response["ids"] #concatenate list
                    count += result_count
                    print("Total # of followers added : %s"%result_count)
                    print("-------------------")
                    time.sleep(sleep)
            # If no next cursor exists
            else:
                if result_count is not None and result_count > 0:
                    print("-------------------")
                    followers += json_response["ids"] 
                    count += result_count
                    print("Total # of followers added : %s"%result_count)
                    print("-------------------")
                    time.sleep(sleep)

                #Since this is the final request, turn flag to false to move to the next time period.
                flag = False
            
        print("Total number of results: ", count)
        return followers


    def save_json(self,
             tweets,
             output_path='data/raw_tweets/tweet_dataset.json'):
        '''
        Save the Twitter response as a json file.
        '''

        with open(output_path, 'w') as f:
            json.dump(tweets,f)

    def load_json(self,
             file_path='data/raw_tweets/tweet_dataset.json'):
        '''
        Load a Twitter response json file.
        '''
        with open(file_path) as json_file:
            return json.load(json_file)
   
    def to_dataframe(self,
                     tweets):
        '''
        Convert the json raw data to a tabular Pandas DataFrame.
        Args:
            data (json):raw data collected from the Twitter API
        Returns:
            tweet_df (pd.DataFrame) : DataFrame with tweet text and metadata
            location_df (pd.DataFrame) : DataFrame with tweet geolocation data
            user_df (pd.DataFrame) : DataFrame with user metadata
       '''
    #tweet
        tweet_id = []
        user_id_t  = []
        location_id_t = []
        created_at_t = []
        language =  []
        source = []
        retweet_count = []
        quote_count = []
        like_count = []
        reply_count = []
        possibly_sensitive =  []
        reply_settings = []
        text = []
        tweet_mentions= []

        #location
        location_id_l =  []
        country = []
        place_type = []
        name_l = []
        longitude = []
        latitude = []

        #user
        user_id_u  = []
        created_at_u = []
        name_u = []
        screen_name = []
        description = []
        verified  = []
        profile_image_URL = []
        user_location = []
        profile_mentions = []
        followers_count = []
        following_count = []
        tweet_count = []
        listed_count = []

        for i in tqdm(range(len(tweets))):
            #split content
            tweet_content = tweets[i]['data']
            tweet_location = tweets[i]['includes']['places']
            user_profile = tweets[i]['includes']['users']
            #tweet
            for t in range(len(tweet_content)) : 

                tweet_id.append(tweet_content[t]['id'])
                user_id_t.append(tweet_content[t]['author_id'])
                if 'geo' in tweet_content[t].keys():
                    location_id_t.append(tweet_content[t]['geo']['place_id'] )
                else:
                    location_id_t.append('No geotag')
                created_at_t.append(dateutil.parser.parse(tweet_content[t]['created_at']))
                language.append(tweet_content[t]['lang'] )
                source.append(tweet_content[t]['source'])
                text.append(tweet_content[t]['text'] )
                possibly_sensitive.append(tweet_content[t]['possibly_sensitive'])
                reply_settings.append(tweet_content[t]['reply_settings'])
                like_count.append(tweet_content[t]['public_metrics']['like_count'])
                reply_count.append(tweet_content[t]['public_metrics']['reply_count'])
                quote_count.append(tweet_content[t]['public_metrics']['quote_count'])
                retweet_count.append(tweet_content[t]['public_metrics']['retweet_count'])       
                mention_str = ''
                if 'entities' in tweet_content[t].keys():
                    if 'mentions' in tweet_content[t]['entities'].keys():
                        mention_str += ''.join(str(tweet_content[t]['entities']['mentions'][e]['username']) +',' 
                                                for e in range(len(tweet_content[t]['entities']['mentions'])))
                    else:
                        mention_str += 'No mention,'
                else:
                    mention_str += 'No mention,'
                
                mention_str = mention_str[:-1]   #Remove the last comma

                tweet_mentions.append(mention_str)
                                
                #location
            for l in range(len(tweet_location)):
                location_id_l.append(tweet_location[l]['id'])
                country.append(tweet_location[l]['country'])
                place_type.append(tweet_location[l]['place_type'])
                name_l.append(tweet_location[l]['name'])
                longitude.append((tweet_location[l]['geo']['bbox'][1]+tweet_location[l]['geo']['bbox'][3])/2)
                latitude.append((tweet_location[l]['geo']['bbox'][0]+tweet_location[l]['geo']['bbox'][2])/2)

                #user
            for u in range(len(user_profile)):
                user_id_u.append(user_profile[u]["id"] )
                created_at_u.append(dateutil.parser.parse(user_profile[u]['created_at']))
                name_u.append(user_profile[u]["name"] )
                screen_name.append(user_profile[u]["username"])
                description.append(user_profile[u]["description"] )
                verified.append(user_profile[u]["verified"] )
                profile_image_URL.append(user_profile[u]["profile_image_url"])
                followers_count.append(user_profile[u]["public_metrics"]["followers_count"])
                following_count.append(user_profile[u]["public_metrics"]["following_count"])
                tweet_count.append(user_profile[u]['public_metrics']['tweet_count'])
                listed_count.append(user_profile[u]['public_metrics']['listed_count'])
                if "location" in user_profile[u].keys():
                    user_location.append(user_profile[u]["location"])
                else:
                    user_location.append(None)
                mentions_str = ''
                if 'entities' in user_profile[u].keys():
                    if 'description' in user_profile[u]['entities'].keys():
                        if 'mentions' in user_profile[u]['entities']['description'].keys():
                            mentions_str += ''.join(str(user_profile[u]['entities']['description']['mentions'][e]['username']) +','
                                                        for e in range(len(user_profile[u]['entities']['description']['mentions'])))
                        else : 
                            mentions_str += 'No mention,'
                    else:
                        mentions_str += 'No mention,'
                else:
                        mentions_str += 'No mention,'
                        
                mention_str = mention_str[:-1]
                profile_mentions.append(mentions_str)
              
        #Create the DataFrames
        tweet_df = pd.DataFrame({"tweet_id":tweet_id,
                                "user_id":user_id_t,
                                "location_id":location_id_t,
                                "created_at": created_at_t,
                                "language":language,
                                "tweet_mentions":tweet_mentions,
                                "reply_settings":reply_settings,
                                'possibly_sensitive':possibly_sensitive,
                                "like_count":like_count,
                                "retweet_count":retweet_count,
                                "quote_count":quote_count,
                                "reply_count":reply_count,
                                "source":source,
                                "text":text}
                            )

        location_df = pd.DataFrame({"location_id":location_id_l,
                                "country":country,
                                "place_type":place_type,
                                "location_geo": name_l,
                                "longitude":longitude,
                                "latitude":latitude})

        user_df = pd.DataFrame({"user_id":user_id_u,
                                "account_created_at":created_at_u,
                                "name":name_u,
                                "screen_name":screen_name,
                                "description":description,
                                "profile_image_url":profile_image_URL,
                                "location_profile":user_location,
                                "profile_mentions": profile_mentions,
                                "followers_count":followers_count,
                                "following_count":following_count,
                                "listed_count":listed_count,
                                "tweet_count":tweet_count,
                                "verified":verified})
        
        #Remove all duplicates
        user_df = user_df.drop_duplicates(subset=['user_id'], 
                                          keep='first', inplace=False, ignore_index=False)
        location_df = location_df.drop_duplicates(subset= ['location_id'],
                                                      keep='first', inplace=False, ignore_index=False)
        #Merge the information from tweet and location dataframes
        tweet_df = tweet_df.merge(location_df,on='location_id',how='left')
        return tweet_df, user_df
