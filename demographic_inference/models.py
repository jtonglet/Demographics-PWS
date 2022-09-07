#Import packages
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
import numpy as np
import pickle
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import  MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

       
class GenerativeModel:
    '''
    Params:
        cardinality (int): number of target classes (e.g. 2 for gender, 4 for age, ...)
        lfs (list): list of base lfs associated to the model
        model (LabelModel): the model itself with its weights.
        seed (int): seed for reproducibility
    '''

    def __init__(self,
                 cardinality,
                 lfs,
                 model=None,
                 seed=42):
        self.cardinality = cardinality
        self.lfs = lfs
        self.model = model
        self.seed = seed

    def apply_lfs(self,user_df):
        '''
        Apply the labeling functions to a user dataframe.
        Args:
            user_df (pd.DataFrame): the user dataframe
        Returns:
            L (np.array): the label matrix.
        '''
        applier = PandasLFApplier(lfs=self.lfs)
        L = applier.apply(df=user_df.fillna(''))
        return L
    
    def fit(self,L):
        '''
        Fit the generative model on a label matrix.
        '''
        self.model = LabelModel(cardinality=self.cardinality,verbose=True)
        self.model.fit(L_train=L, n_epochs=500, log_freq=100, seed=self.seed)
    
    def predict(self,L):
        '''
        Predict the labels for a label matrix.
        '''
        preds = self.model.predict(L)
        return preds   

    def extend(self,L,discriminative_predictions):
        '''
        Add the predictions of a discriminative model to a label matrix.
        Args:
            L (np.array): the base label matrix.
            discriminative_predictions (np.array): the predictions of the discriminative model.
        Returns:
            L_extended (np.array): the extended label matrix.
        '''
        reshaped_preds = np.reshape(discriminative_predictions,(len(discriminative_predictions),1))
        L_extended = np.hstack((L,reshaped_preds))
        return L_extended


class MajorityModel:
    '''
    Params:
        lfs (list): list of base lfs associated to the model
    '''
    def __init__(self,
                lfs):
        self.lfs = lfs

    def apply_lfs(self,user_df):
        '''
        Apply the labeling functions to a user dataframe.
        Args:
            user_df (pd.DataFrame): the user dataframe
        Returns:
            L (np.array): the label matrix.
        '''                    
        applier = PandasLFApplier(lfs=self.lfs)
        L = applier.apply(df=user_df)
        return L
        
    def predict(self,L):
        '''
        Make predictions for the label matrix, using the mode of each column.
        '''
        self.model = MajorityLabelVoter()
        preds = self.model.predict(L)
        return preds  

    def extend(self,L,discriminative_predictions):
        '''
        Add the predictions of a discriminative model to a label matrix.
        Args:
            L (np.array): the base label matrix.
            discriminative_predictions (np.array): the predictions of the discriminative model.
        Returns:
            L_extended (np.array): the extended label matrix.
        '''
        reshaped_preds = np.reshape(discriminative_predictions,(len(discriminative_predictions),1))
        L_extended = np.hstack((L,reshaped_preds))
        return L_extended 

class DiscriminativeModel:
    '''
    Params:
        base_model (object): the classifier to include in the prediction pipeline. e.g. logistic regression, random forest, xgboost, ...
        pipe (sklearn.Pipeline): pipeline object containing the classifier and the preprocessing steps.
    '''
    def __init__(self,
                 base_model=LogisticRegression(),
                 pipe = None
                 ):
        self.base_model = base_model
        self.pipe = pipe

    def fit(self,
            X,
            y,
            scaler = MinMaxScaler(),
            cat_encoder = OneHotEncoder(handle_unknown='ignore'),           
            cat_features=[ 'main_source', 'favorite_period', 'favorite_day','main_reply_settings'],
            ):
        '''
        Args:
            X (pd.DataFrame): the feature matrix. It should only include features to be used in the classification.
            y (np.array): the target labels
            scaler (object): scaler applied on the numeric features. By default MinMaxScaler.
            cat_encoder (object): categorical encoder applied on the categorical features. By default OneHotEncoder.
            cat_features (list): the list of categorical features. Used implicitely to define the list of numeric features as well.
        '''
        numeric_features =  X.columns.drop(cat_features).to_list()
        preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_features),  
            ("cat", cat_encoder, cat_features)     
        ],
        remainder = 'passthrough'       
        )

        #Pipeline objects
        self.pipe =  Pipeline(steps=[("preprocessor", preprocessor), 
                                ("classifier", self.base_model)])
        self.pipe.fit(X,y)
    

    def predict(self,X):
        '''
        Predict the labels.
        '''
        predictions = self.pipe.predict(X)
        return predictions
    
    def save(self,output_path):
        '''
        Save the model's pipeline
        '''
        pickle.dump(self.pipe, open(output_path, 'wb'))

    def load(self,path):
        '''
        Load the model's pipeline
        '''
        self.pipe = pickle.load(open(path,'rb'))

    def cross_validate(self,X,y,cv=5,scoring=['accuracy','f1_macro']):
        '''
        Apply cross-validation on the dataset.
        Args:
            X (pd.DataFrame): the feature matrix. It should only include features to be used in the classification.
            y (np.array): the target labels
            cv (int): number of folds for cross-validation
            scoring (list): list of metrics to use for scoring.
        Returns:
            score_dict (dict): evaluation metrics dictionnary.
        '''

        score_dict = cross_validate(self.pipe,X,y,cv,scoring) 
        for k in score_dict.keys():
            if k not in ['fit_time','score_time']:
                print('Metric %s : %s'%(k,np.mean(score_dict[k])))   
        return score_dict    