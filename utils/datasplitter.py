#Import packages
import pandas as pd

def active_mentioned_users_split(user_df, tweet_df):
    '''
    Split a user dataframe into two dataframes containing the active users, those who wrote tweets, and the mentioned users.
    '''
    id_list = tweet_df["user_id"].unique()
    active_user_df = user_df[user_df["user_id"].isin(id_list)]
    mentioned_user_df = user_df[~user_df["user_id"].isin(id_list)]
    return active_user_df, mentioned_user_df

def get_verified_users(user_df,mentioned_user_df):  
    '''
    Get verified users from the active and mentioned users dataframes.
    '''
    verified_active = user_df[user_df["verified"]].fillna("Missing")
    verified_mentioned = mentioned_user_df[mentioned_user_df["verified"]].fillna("Missing")
    verified = pd.concat([verified_active,verified_mentioned])
    return verified