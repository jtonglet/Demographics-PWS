#Import packages
import numpy as np
from PIL import Image
import requests
from deepface import DeepFace
import torch 

def deep_face_classification(url):
    '''
    Gender classification from profile image with VGG-Face.
    Args : 
        url (str): the url to the profile image
    Returns:
        prediction (str):  the gender predicted by VGG-Face.            
    '''
    if '_normal' in url:
      #If available set the image to an higher resolution
      url = url.replace('_normal', '_400x400')
    try:
      request = requests.get(url, stream = True)
      im = Image.open(request.raw)
      im_np = np.asarray(im)
      obj = DeepFace.analyze(img_path = im_np, actions = ['gender'],prog_bar=True)
      prediction = obj['gender'] 
    except:
      prediction = 'invalid_url or no face detected'
    return prediction


def CLIP_classification(url,
                        text_tokens,
                        preprocess,
                        model):
    '''
    Predict the user's gender using CLIP.
    Args:
        url : url link to the user's profile picture in 400x400
        text_tokens : the class labels
        preprocess : the CLIP model preprocessing function
        model : the CLIP model
    Returns :
        probs : a list with the probabilities assigned to each text token
    '''
    if '_normal' in url:
      url = url.replace('_normal', '_400x400')
    try:
      request = requests.get(url, stream = True)
      im = Image.open(request.raw)
      if torch.cuda.is_available():
        im_preprocessed = preprocess(im).unsqueeze(0).cuda()
      else:
        im_preprocessed = preprocess(im).unsqueeze(0)        
      with torch.no_grad():
        image_features  = model.encode_image(im_preprocessed)
        text_features = model.encode_text(text_tokens)
      logits_per_image, logits_per_text = model(im_preprocessed, text_tokens)
      pred = np.argmax(logits_per_image.softmax(dim=-1).cpu().detach().numpy())
    except:
      pred = 'invalid_url'
    return pred