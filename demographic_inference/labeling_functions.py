#Import packages
import pandas as pd
import re 
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from utils.mapping import *
from preprocessing.text_preprocessing import TextPreprocessor
from demographic_inference.third_party_lf import deep_face_classification,  CLIP_classification
import clip
import torch

############################################
## Generic functions to create custom LFs ##
############################################

#Generic function to create keyword LFs
def keyword_lookup(x,keywords, label,remove_url=True):
    if remove_url:
      preprocessor = TextPreprocessor()
      tokens = preprocessor.tokenizer.tokenize(x['description'])
      description = ' '.join(preprocessor.remove_hyperlinks(tokens))
    else:
      description = x['description']
    if any(word in description.lower().split(' ') for word in keywords):
        return label
    return ABSTAIN
  
def make_keyword_lf(keywords, label = ABSTAIN):
    return LabelingFunction(  
        name = f"keyword_{label}",
        f = keyword_lookup,
        resources = dict(keywords = keywords, label = label),
    )

def make_keyword_lf_zip(keywords, label = ABSTAIN):
  return LabelingFunction(
    name = f"zip_{label}",
    f = keyword_lookup,
    resources = dict(keywords = keywords, label = label),
  )

#Generic function to create name LFs
def name_lookup(x,names, label):
  x['first_name'] = x['name'].split(' ')[0].lower()
  if x['first_name'] in names:
    return label
  return ABSTAIN

def make_first_name_lf(names_set,label=ABSTAIN):
  return LabelingFunction(
    name = f"name_{label}",
    f = name_lookup,
    resources = dict(names=names_set,label=label)
  )

def location_profile_lookup(x,locations,label):
  # This lambda function checks if at least one element of set b is also in set a
  any_in = lambda a, b : any(i in b for i in a)
  delimiters = ", ",","," ","..."
  regexPattern = '|'.join(map(re.escape, delimiters))
  if any_in(locations, re.split(regexPattern,x['location_profile'].lower())):
    return label
  return ABSTAIN

def make_location_profile_lf(locations,label=ABSTAIN):
    return LabelingFunction(
    name = f"location_{label}",
    f = location_profile_lookup,
    resources = dict(locations=locations,label=label)
  )


def CLIP_profile_image_lookup(x,tokens,labels):
  #Set cuda config
  device = "cuda" if torch.cuda.is_available() else "cpu"
  #Load CLIP pre-trained model
  model, preprocess = clip.load('ViT-B/32', device)
  #Define text tokens
  if torch.cuda.is_available():
    text_tokens = clip.tokenize(tokens).cuda()
  else :
    text_tokens = clip.tokenize(tokens)
  pred = CLIP_classification(x['profile_image_url'], text_tokens,preprocess,model)
  for label in labels:
    if pred==label:
      return label
  return ABSTAIN

def make_CLIP_lf(tokens,labels):
  return LabelingFunction(
    name = "CLIP_lf",
    f = CLIP_profile_image_lookup,
    resources = dict(tokens=tokens,labels=labels)
  )



###################################################################
## LFs specific to the experiment conducted in Flanders, Belgium ##
###################################################################


#Load Resources
#Name Dictionary
names = pd.read_csv("data/resources/gender/names.csv")
female_names = [n.lower() for n in names["Female"] if type(n) == str]
male_names = [n.lower() for n in names["Male"]  if type(n) == str]
female_names_set = set(female_names)
male_names_set = set(male_names)
bigender_names_set = female_names_set.intersection(male_names_set)
#Remove all bigender names from the male and female names set
male_names_set -= bigender_names_set
female_names_set -= bigender_names_set
#Keywords
#Gender
gender_keywords = pd.read_csv("data/resources/gender/GenderKeywords.csv")
#Age 
age_keywords = pd.read_csv("data/resources/age/AgeKeywords.csv")
#Location
city_names = pd.read_csv('data/resources/location/city_names.csv', sep = ',')
zipcodes = pd.read_csv("data/resources/location/zipcodes.csv",dtype=str)


#Gender Task : 2 classes
#Keywords in description field
male_keywords = make_keyword_lf(keywords = gender_keywords['Male'].dropna().to_list(), label = MALE)
female_keywords = make_keyword_lf(keywords = gender_keywords['Female'].dropna().to_list(), label = FEMALE)
#Names 
male_first_name = make_first_name_lf(names_set=male_names_set,label=MALE)
female_first_name = make_first_name_lf(names_set=female_names_set,label=FEMALE)
#CLIP
#The order the tokens must correspond to the order of the demographic categories
# e.g. the token for female must come in index 0, and for male in index 1
CLIP_lf = make_CLIP_lf(tokens= ["a woman", "a man", "an object"],labels=[FEMALE,MALE])
#VGG-Face (DeepFace library) predictions are more rigid and return either Man or Woman as label
@labeling_function()
def gender_deepface(x):
  pred = deep_face_classification(x['profile_image_url'])
  return MALE if pred == 'Man' else FEMALE if pred=='Woman' else ABSTAIN


#Age  Task : 4 classes 
#Age keywords in description field
fourtiesabove_keywords = make_keyword_lf(keywords = age_keywords['40+'].dropna().to_list() , label = FOURTIESABOVE)
thirties_keywords = make_keyword_lf(keywords = age_keywords['30-39'].dropna().to_list(), label = THIRTIES)
twenties_keywords = make_keyword_lf(keywords = age_keywords['19-29'].dropna().to_list(), label = TWENTIES)
minor_keywords = make_keyword_lf(keywords = age_keywords['-18'].dropna().to_list(), label = MINOR)
#RegEx
#How to define good regexs is very dependent to the population and language studied. Hence we do not provide a LF template but examples instead.
#LF for minor 
#included ages 13-18: 1[3-8]
@labeling_function()
def regex_minor_jaar(row):
  return MINOR if re.search(r"([^0-9]1[3-8]\s*(y\/?o?)\s)|(\b1[3-8]\s*-?(i?)(jaar|jaren|years?|lentes|jarige?|años))", row['description']) else ABSTAIN
@labeling_function()
def regex_minor_begin(row):
  return MINOR if re.search(r"(^\s*1[3-8]\s*[^0-9\./])|(^°?\s*200[4-9]\s*)|(^°?\s*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9]\s*)", row['description']) else ABSTAIN
@labeling_function()
def regex_minor_only(row):
  return MINOR if re.search(r"(^\s*1[3-8]\s*$)|(^\s*200[4-9]\s*$)", row['description']) else ABSTAIN
@labeling_function()
def regex_minor_delimiterage(row):
  return MINOR if re.search(r"([^0-9]\|\s?1[3-8]\s?\|[^0-9])|([^0-9]\.\s?1[3-8]\s?\.[^0-9])|([^0-9],\s?1[3-8]\s?,[^0-9])|([^0-9]-\s?1[3-8]\s?-[^0-9])|([^0-9]•\s?1[3-8]\s?•[^0-9])", row['description']) else ABSTAIN
@labeling_function()
def regex_minor_delimiterbd(row):
  return MINOR if re.search(r"(\|\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9]\s?\|)|(\.\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9]\s?\.)|(,\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9]\s?,)|(-\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9]\s?-)|(•\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9]\s?•)", row['description']) else ABSTAIN
@labeling_function()
def regex_minor_born(row):
  return MINOR if re.search(r"(\b(?i)(born(\sin)?|anno|est.?|circa|ca\.?|sedert|°)\s*'?0[4-9]\s)|(^\s*(?i)(since|sinds)\s*'?0[4-9]\s)", row['description']) else ABSTAIN
@labeling_function()
def regex_minor_bornfull(row):
  return MINOR if re.search(r"(\b(?i)(born(\son)?|anno|est.?|circa|ca\.?|sedert|since|°|birthday)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9])|(^\s*(?i)(since|sinds)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?200[4-9])", row['description']) else ABSTAIN
#LF for twenties 
#included ages 19-29
@labeling_function()
def regex_twenties_jaar(row):
  return TWENTIES if re.search(r"([^0-9](19|2[0-9])\s*(y\/?o?)\s)|(\b(19|2[0-9])\s*-?(i?)(jaar|jaren|years?|lentes|jarige?|años))", row['description']) else ABSTAIN
@labeling_function()
def regex_twenties_begin(row):
  return TWENTIES if re.search(r"(^\s*(19|2[0-9])\s*[^0-9])|(^°?\s*(200[0-3]|199[3-9])\s*)|(^°?\s*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9])\s*)", row['description']) else ABSTAIN
@labeling_function()
def regex_twenties_only(row):
  return TWENTIES if re.search(r"(^\s*(19|2[0-9])\s*$)|(^\s*(200[0-3]|199[3-9])\s*$)", row['description']) else ABSTAIN
@labeling_function()
def regex_twenties_delimiterage(row):
  return TWENTIES if re.search(r"([^0-9]\|\s?(19|2[0-9])\s?\|[^0-9])|([^0-9]\.\s?(19|2[0-9])\s?\.[^0-9])|([^0-9],\s?(19|2[0-9])\s?,[^0-9])|([^0-9]-\s?(19|2[0-9])\s?-[^0-9])|([^0-9]•\s?(19|2[0-9])\s?•[^0-9])", row['description']) else ABSTAIN
@labeling_function()
def regex_twenties_delimiterbd(row):
  return TWENTIES if re.search(r"(\|\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9])\s?\|)|(\.\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9])\s?\.)|(,\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9])\s?,)|(-\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9])\s?-)|(•\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9])\s?•)", row['description']) else ABSTAIN
@labeling_function()
def regex_twenties_born(row):
  return TWENTIES if re.search(r"(\b(?i)(born(\sin)?|anno|est.?|circa|ca\.?|sedert|°)\s*(((19|')?9[3-9])|((20|')?0[0-3]))\s)|(^\s*(?i)(since|sinds)\s*(((19|')?9[3-9])|((20|')?0[0-3]))\s)", row['description']) else ABSTAIN
@labeling_function()
def regex_twenties_bornfull(row):
  return TWENTIES if re.search(r"(\b(?i)(born(\son)?|anno|est.?|circa|ca\.?|sedert|since|°|birthday)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9]))|(^\s*(?i)(since|sinds)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(200[0-3]|199[3-9]))", row['description']) else ABSTAIN
#LF for thirties
#included ages 30-39
@labeling_function()
def regex_thirties_jaar(row):
  return THIRTIES if re.search(r"([^0-9]3[0-9]\s*(y\/?o?)\s)|(\b(3[0-9])\s*-?(i?)(jaar|jaren|years?|lentes|jarige?|años))", row['description']) else ABSTAIN
@labeling_function()
def regex_thirties_begin(row):
  return THIRTIES if re.search(r"(^\s*3[0-9]\s*[^0-9])|(^°?\s*(198[3-9]|199[0-2])\s*)|(^°?\s*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2])\s*)", row['description']) else ABSTAIN
@labeling_function()
def regex_thirties_only(row):
  return THIRTIES if re.search(r"(^\s*3[0-9]\s*$)|(^\s*(198[3-9]|199[0-2])\s*$)", row['description']) else ABSTAIN
@labeling_function()
def regex_thirties_delimiterage(row):
  return THIRTIES if re.search(r"([^0-9]\|\s?3[0-9]\s?\|[^0-9])|([^0-9]\.\s?3[0-9]\s?\.[^0-9])|([^0-9],\s?3[0-9]\s?,[^0-9])|([^0-9]-\s?3[0-9]\s?-[^0-9])|([^0-9]•\s?3[0-9]\s?•[^0-9])", row['description']) else ABSTAIN
@labeling_function()
def regex_thirties_delimiterbd(row):
  return THIRTIES if re.search(r"(\|\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2])\s?\|)|(\.\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2])\s?\.)|(,\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2])\s?,)|(-\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2])\s?-)|(•\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2])\s?•)", row['description']) else ABSTAIN
@labeling_function()
def regex_thirties_born(row):
  return THIRTIES if re.search(r"(\b(?i)(born(\sin)?|anno|est.?|circa|ca\.?|sedert|°)\s*(19|')?(8[3-9]|9[0-2])\s)|(^\s*(?i)(since|sinds)\s*(19|')?(8[3-9]|9[0-2])\s)", row['description']) else ABSTAIN
@labeling_function()
def regex_thirties_bornfull(row):
  return THIRTIES if re.search(r"(\b(?i)(born(\son)?|anno|est.?|circa|ca\.?|sedert|since|°|birthday)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2]))|(^\s*(?i)(since|sinds)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?(198[3-9]|199[0-2]))", row['description']) else ABSTAIN
#LF for 40+
#included ages 40-80
@labeling_function()
def regex_fourtiesabove_jaar(row):
  return FOURTIESABOVE if re.search(r"([^0-9]([4-7][0-9]|80)\s*(y\/?o?)\s)|(\b([4-7][0-9]|80)\s*-?(i?)(jaar|jaren|years?|lentes|jarige?|años))", row['description']) else ABSTAIN
@labeling_function()
def regex_fourtiesabove_begin(row):
  return FOURTIESABOVE if re.search(r"(^\s*([4-7][0-9]|80)\s*[^0-9])|(^°?\s*19(4[2-9]|[5-7][0-9]|8[0-2])\s*)|(^°?\s*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2])\s*)", row['description']) else ABSTAIN
@labeling_function()
def regex_fourtiesabove_only(row):
  return FOURTIESABOVE if re.search(r"(^\s*([4-7][0-9]|80)\s*$)|(^\s*19(4[2-9]|[5-7][0-9]|8[0-2])\s*$)", row['description']) else ABSTAIN
@labeling_function()
def regex_fourtiesabove_delimiterage(row):
  return FOURTIESABOVE if re.search(r"([^0-9]\|\s?([4-7][0-9]|80)\s?\|[^0-9])|([^0-9]\.\s?([4-7][0-9]|80)\s?\.[^0-9])|([^0-9],\s?([4-7][0-9]|80)\s?,[^0-9])|([^0-9]-\s?([4-7][0-9]|80)\s?-[^0-9])|([^0-9]•\s?([4-7][0-9]|80)\s?•[^0-9])", row['description']) else ABSTAIN
@labeling_function()
def regex_fourtiesabove_delimiterbd(row):
  return FOURTIESABOVE if re.search(r"(\|\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2])\s?\|)|(\.\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2])\s?\.)|(,\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2])\s?,)|(-\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2])\s?-)|(•\s?[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2])\s?•)", row['description']) else ABSTAIN
@labeling_function()
def regex_fourtiesabove_born(row):
  return FOURTIESABOVE if re.search(r"(\b(?i)(born(\sin)?|anno|est.?|circa|ca\.?|sedert|°)\s*(19|')?(4[2-9]|[5-7][0-9]|8[0-2])\s)|(^\s*(?i)(since|sinds)\s*(19|')?(4[2-9]|[5-7][0-9]|8[0-2])\s)", row['description']) else ABSTAIN
@labeling_function()
def regex_fourtiesabove_bornfull(row):
  return FOURTIESABOVE if re.search(r"(\b(?i)(born(\son)?|anno|est.?|circa|ca\.?|sedert|since|°|birthday)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2]))|(^\s*(?i)(since|sinds)\s*\W*[0-3][0-9]\s?/\s?[0-1][0-9]\s?/\s?19(4[2-9]|[5-7][0-9]|8[0-2]))", row['description']) else ABSTAIN


#Location Task : 7 classes
#All location LFs are based on templates
#Location in location_profile field
antwerpen_location = make_location_profile_lf(locations = city_names['Antwerpen'].dropna().str.lower(), label = ANTWERPEN)
limburg_location = make_location_profile_lf(locations = city_names['Limburg'].dropna().str.lower(), label = LIMBURG)
oostvl_location = make_location_profile_lf(locations = city_names['Oost_Vlaanderen'].dropna().str.lower(), label = OOST_VLAANDEREN)
westvl_location = make_location_profile_lf(locations = city_names['West_Vlaanderen'].dropna().str.lower(), label = WEST_VLAANDEREN)
vlbrabant_location = make_location_profile_lf(locations = city_names['Vlaams_Brabant'].dropna().str.lower(), label = VLAAMS_BRABANT)
brussel_wallonie_location = make_location_profile_lf(locations = city_names['Brussel'].dropna().str.lower().to_list() + city_names['Wallonie'].dropna().str.lower().to_list(), label = BRUSSEL_WALLONIE)
other_location = make_location_profile_lf(locations = city_names['Nederland'].dropna().str.lower().to_list() + city_names['Other'].dropna().str.lower().to_list(), label = OTHER)
#Location in description field
antwerpen_profile = make_keyword_lf(keywords = city_names['Antwerpen'].dropna().str.lower(), label = ANTWERPEN)
limburg_profile = make_keyword_lf(keywords = city_names['Limburg'].dropna().str.lower(), label = LIMBURG)
oostvl_profile = make_keyword_lf(keywords = city_names['Oost_Vlaanderen'].dropna().str.lower(), label = OOST_VLAANDEREN)
westvl_profile = make_keyword_lf(keywords = city_names['West_Vlaanderen'].dropna().str.lower(), label = WEST_VLAANDEREN)
vlbrabant_profile = make_keyword_lf(keywords = city_names['Vlaams_Brabant'].dropna().str.lower(), label = VLAAMS_BRABANT)
brussel_wallonie_profile = make_keyword_lf(keywords = city_names['Brussel'].dropna().str.lower().to_list() + city_names['Wallonie'].dropna().str.lower().to_list(), label = BRUSSEL_WALLONIE)
other_profile = make_keyword_lf(keywords = city_names['Nederland'].dropna().str.lower().to_list() + city_names['Other'].dropna().str.lower().to_list(), label = OTHER)
#ZIP in description field
antwerpen_zip = make_keyword_lf_zip(keywords = zipcodes['Antwerpen'].dropna(), label = ANTWERPEN)
limburg_zip = make_keyword_lf_zip(keywords = zipcodes['Limburg'].dropna(), label = LIMBURG)
oostvl_zip = make_keyword_lf_zip(keywords = zipcodes['Oost_Vlaanderen'].dropna(), label = OOST_VLAANDEREN)
westvl_zip = make_keyword_lf_zip(keywords = zipcodes['West_Vlaanderen'].dropna(), label = WEST_VLAANDEREN)
vlbrabant_zip = make_keyword_lf_zip(keywords = zipcodes['Vlaams_Brabant'].dropna(), label = VLAAMS_BRABANT)
brussel_wallonie_zip = make_keyword_lf_zip(keywords = zipcodes['Brussel'].dropna().to_list() + zipcodes['Wallonie'].dropna().to_list(), label = BRUSSEL_WALLONIE)
nederland_zip = make_keyword_lf_zip(keywords = zipcodes['Nederland'].dropna(), label = OTHER)