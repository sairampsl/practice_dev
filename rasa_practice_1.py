
from stopwords import get_stopwords
import os
import pandas as pd
from fuzzywuzzy import process

import typing
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.model import Metadata


name = "FuzzyExtractor"
provides = ["entities"]
requires = ["tokens"]
defaults = {}
language_list  =["en"]
threshold = 90

    
STOP_WORDS = get_stopwords("en")

#entities = list(message.get('entities'))

data_path = "data/nlu/lookups/region.txt"

file= os.path.abspath(data_path)
data = pd.read_csv(file,header=None)
lookup_data = data[0].tolist()

#tokens = message.get('tokens')
entities = []
token_text = "region"
if token_text not in STOP_WORDS:
    fuzzy_results = process.extract(
                             token_text, 
                             lookup_data, 
                             processor=lambda a: a['value'] 
                                 if isinstance(a, dict) else a, 
                             limit=1)

    for result, confidence in fuzzy_results:
        if confidence >= threshold:
            entities.append({
                "start": 1,
                "end": 1,
                "value": token_text,
                "fuzzy_value": result,
                "confidence": confidence,
                "entity": result
            })

result = fuzzy_results[0][0]
confidence = fuzzy_results[0][1]

new_entities = []
for ent in new_entities:
    print(ent)
            
print(entities)