from stopwords import get_stopwords
import os
import json
import pandas as pd
from fuzzywuzzy import process

import typing
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.model import Metadata

threshold = 70
new_entities = []
data1_data_path = "data/nlu/lookups/data2.txt"
data1_file = os.path.abspath(data1_data_path)
data1_df = pd.read_csv(data1_file,header=None)
data1_lookup = data1_df[0].tolist()

data2_data_path = "data/nlu/lookups/data1.txt"
data2_file = os.path.abspath(data2_data_path)
data2_df = pd.read_csv(data2_file,header=None)
data2_lookup = data2_df[0].tolist()

STOP_WORDS = get_stopwords("en")
intent = 'expected' #message.get('intent')
intent_name = intent
tokens = ['How']
data1_intents = ['expected_hospitalization_in_data1','select_data1']

data2_intents = ['top']

question_intents = data1_intents + data2_intents

if intent_name in data2_intents:
    data2_df = pd.DataFrame()
    for token in tokens:
        text = token #token.text
        text = text.lower()
        if text not in STOP_WORDS:
            fuzzy_results = process.extract(
                                 text, 
                                 data2_lookup, 
                                 limit=1)
            result = fuzzy_results[0][0]
            confidence = fuzzy_results[0][1]
            if confidence >= threshold:
                data2_dict = {}
                data2_dict['entity'] = 'data2'
                data2_dict['start'] = token.start
                data2_dict['end'] = token.end
                data2_dict['value'] = str(result)
                data2_dict['confidence'] = ((confidence*1.0)/100)
                data2_dict['extractor'] = 'name'
                temp_df = pd.DataFrame([data2_dict])
                data2_df = data2_df.append(temp_df,ignore_index=True)
                
    if(len(data2_df)>0):
        data2_df = data2_df.sort_values(by=['confidence'],ascending=False,ignore_index=True)
        data2_df = data2_df.head(1)
        data2_json_dict=json.loads(data2_df.to_json(orient='split'))
        del data2_json_dict['index']
        new_entities.append(data2_json_dict)
        
if intent_name in data1_intents:
    data1_df = pd.DataFrame()
    for token in tokens:
        text = token #token.text
        text = text.lower()
        if text not in STOP_WORDS:
            fuzzy_results = process.extract(
                                 text, 
                                 data1_lookup, 
                                 limit=1)
            result = fuzzy_results[0][0]
            confidence = fuzzy_results[0][1]
            if confidence >= threshold:
                data1_dict = {}
                data1_dict['entity'] = 'data1'
                data1_dict['start'] = 1
                data1_dict['end'] = 1
                data1_dict['value'] = str(result)
                data1_dict['confidence'] = ((confidence*1.0)/100)
                data1_dict['extractor'] = 'name'
                temp_df = pd.DataFrame([data1_dict])
                data1_df = data1_df.append(temp_df,ignore_index=True)
                
    if(len(data1_df)>0):
        data1_df = data1_df.sort_values(by=['confidence'],ascending=False,ignore_index=True)
        data1_df = data1_df.head(1)
        
        data1_json_dict=data1_df.to_dict('records')[0]
        new_entities.append(data1_json_dict)