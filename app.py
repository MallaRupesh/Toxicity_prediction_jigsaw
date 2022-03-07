
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from bs4 import BeautifulSoup
# from werkzeug import filename
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from joblib import dump, load
import time
import itertools
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import streamlit as stl


def final_func_1(x):
    
    '''Final function 1 handling all the cleaning and loading of tokenizer , vectorizer and model and predicting '''
    #Using regex and string manipulation to clean the data 


    def clean(data):
        #https://www.kaggle.com/andrej0marinchenko/jigsaw-ensemble-0-86/notebook
        data = data.str.replace('https?://\S+|www\.\S+', ' social medium ')      

        data = data.str.lower()
        data = data.str.replace("4", "a") 
        data = data.str.replace("2", "l")
        data = data.str.replace("5", "s") 
        data = data.str.replace("1", "i") 
        data = data.str.replace("!", "i") 
        data = data.str.replace("|", "i") 
        data = data.str.replace("0", "o") 
        data = data.str.replace("8", "ate") 
        data = data.str.replace("3", "e") 
        data = data.str.replace("9", "g")
        data = data.str.replace("6", "g")
        data = data.str.replace("@", "a")
        data = data.str.replace("$", "s")
        data = data.str.replace("l3", "b") 
        data = data.str.replace("7", "t") 
        data = data.str.replace("7", "+") 

        data = data.str.replace("#ofc", " of fuckin course ")
        data = data.str.replace("fggt", " faggot ")
        data = data.str.replace("your", " your ")
        data = data.str.replace("self", " self ")
        data = data.str.replace("cuntbag", " cunt bag ")
        data = data.str.replace("fartchina", " fart china ")    
        data = data.str.replace("youi", " you i ")
        data = data.str.replace("cunti", " cunt i ")
        data = data.str.replace("sucki", " suck i ")
        data = data.str.replace("pagedelete", " page delete ")
        data = data.str.replace("cuntsi", " cuntsi ")
        data = data.str.replace("i'm", " i am ")
        data = data.str.replace("offuck", " of fuck ")
        data = data.str.replace("centraliststupid", " central ist stupid ")
        data = data.str.replace("hitleri", " hitler i ")
        data = data.str.replace("i've", " i have ")
        data = data.str.replace("i'll", " sick ")
        data = data.str.replace("fuck", " fuck ")
        data = data.str.replace("f u c k", " fuck ")
        data = data.str.replace("shit", " shit ")
        data = data.str.replace("bunksteve", " bunk steve ")
        data = data.str.replace('wikipedia', ' social medium ')
        data = data.str.replace("faggot", " faggot ")
        data = data.str.replace("delanoy", " delanoy ")
        data = data.str.replace("jewish", " jewish ")
        data = data.str.replace("sexsex", " sex ")
        data = data.str.replace("allii", " all ii ")
        data = data.str.replace("i'd", " i had ")
        data = data.str.replace("'s", " is ")
        data = data.str.replace("youbollocks", " you bollocks ")
        data = data.str.replace("dick", " dick ")
        data = data.str.replace("cuntsi", " cuntsi ")
        data = data.str.replace("mothjer", " mother ")
        data = data.str.replace("cuntfranks", " cunt ")
        data = data.str.replace("ullmann", " jewish ")
        data = data.str.replace("mr.", " mister ")
        data = data.str.replace("aidsaids", " aids ")
        data = data.str.replace("njgw", " nigger ")
        data = data.str.replace("wiki", " social medium ")
        data = data.str.replace("administrator", " admin ")
        data = data.str.replace("gamaliel", " jewish ")
        data = data.str.replace("rvv", " vanadalism ")
        data = data.str.replace("admins", " admin ")
        data = data.str.replace("pensnsnniensnsn", " penis ")
        data = data.str.replace("pneis", " penis ")
        data = data.str.replace("pennnis", " penis ")
        data = data.str.replace("pov.", " point of view ")
        data = data.str.replace("vandalising", " vandalism ")
        data = data.str.replace("cock", " dick ")
        data = data.str.replace("asshole", " asshole ")
        data = data.str.replace("youi", " you ")
        data = data.str.replace("afd", " all fucking day ")
        data = data.str.replace("sockpuppets", " sockpuppetry ")
        data = data.str.replace("iiprick", " iprick ")
        data = data.str.replace("penisi", " penis ")
        data = data.str.replace("warrior", " warrior ")
        data = data.str.replace("loil", " laughing out insanely loud ")
        data = data.str.replace("vandalise", " vanadalism ")
        data = data.str.replace("helli", " helli ")
        data = data.str.replace("lunchablesi", " lunchablesi ")
        data = data.str.replace("special", " special ")
        data = data.str.replace("ilol", " i lol ")
        data = data.str.replace(r'\b[uU]\b', 'you')
        data = data.str.replace(r"what's", "what is ")
        data = data.str.replace(r"\'s", " is ")
        data = data.str.replace(r"\'ve", " have ")
        data = data.str.replace(r"can't", "cannot ")
        data = data.str.replace(r"n't", " not ")
        data = data.str.replace(r"i'm", "i am ")
        data = data.str.replace(r"\'re", " are ")
        data = data.str.replace(r"\'d", " would ")
        data = data.str.replace(r"\'ll", " will ")
        data = data.str.replace(r"\'scuse", " excuse ")
        data = data.str.replace('\s+', ' ')  
        data = data.str.replace(r'(.)\1+', r'\1\1') 
        data = data.str.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '')


        data = data.str.replace(r"what's", "what is ")    
        data = data.str.replace(r"\'ve", " have ")
        data = data.str.replace(r"can't", "cannot ")
        data = data.str.replace(r"n't", " not ")
        data = data.str.replace(r"i'm", "i am ")
        data = data.str.replace(r"\'re", " are ")
        data = data.str.replace(r"\'d", " would ")
        data = data.str.replace(r"\'ll", " will ")
        data = data.str.replace(r"\'scuse", " excuse ")
        data = data.str.replace(r"\'s", " ")

        # Clean some punctutations
        data = data.str.replace('\n', ' \n ')
        data = data.str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
        # Replace repeating characters more than 3 times to length of 3
        data = data.str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')   
        # Add space around repeating characters
        data = data.str.replace(r'([*!?\']+)',r' \1 ')    
        # patterns with repeating characters 
        data = data.str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
        data = data.str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
        data = data.str.replace(r'[ ]{2,}',' ').str.strip()   
        data = data.str.replace(r'[ ]{2,}',' ').str.strip()   
#         data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        return data

    
    # Loading the pre trained bert tokenizer 
#     with open('tokenizer_new.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)

 
#     # Loading the pre trained tfidf vectorizer
#     # def dummy_fun(doc):
#     #     return doc



#     with open('vectorizer_new.pickle', 'rb') as handle:
        
#         vectorizer = pickle.load(handle)

#     # Loading the model
# #     regressor = pickle.load(open("/Users/rupesh/Downloads/Toxicity /finalized_model.sav", 'rb'))
#     with open('regressor_new.pickle', 'rb') as handle:
        
#         regressor = pickle.load(handle)

#     cleaning the text 
    x = clean(x)

    # tokenized_comments = tokenizer(x.tolist())['input_ids']

    # comments_tr = vectorizer.fit_transform(tokenized_comments)


    # preds = regressor.predict(comments_tr)

    with open('pipeline_new.pickle', 'rb') as handle:
        
        pipeline= pickle.load(handle)

    preds = pipeline.predict(x)  

    
    return preds




    
st.title("Predicting Toxicity of comments")
st.header('This app is created to predict how toxic a comment is')

text1 = st.text_area('Enter text')

text1=pd.Series(text1)
if(stl.button('Submit')):
	output =final_func_1(text1)

	# output = predict(text_news)
	# output = str(output)
	# st.success(f"The output score is : {output}")
	# print(float(output))
	stl.text("The level of toxicity :")
	my_bar = st.progress(float(output))
	st.success(f"The output score is : {output}")
	# for percent_complete in range(0,1):
	#      # time.sleep(0.1)
	#      my_bar.progress(percent_complete + 1)
	
        


