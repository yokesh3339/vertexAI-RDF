import uvicorn
import os,sys
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI,Request,Response



app=FastAPI(title="Vertex AI Predictions")
AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')
AIP_STORAGE_URI = os.environ.get('AIP_STORAGE_URI')


class ClassPrediction(BaseModel):
    DocType: str
    Confidence: float

class ClassPredictions(BaseModel):
    prediction: List[ClassPrediction]




############################################# Classification Model Packages ###############################################
import numpy as np
import pandas as pd
import nltk
root=os.path.dirname(os.path.abspath(__file__))
nltk_dir=os.path.join(root,"nltk_data")
print((nltk.data.path))
#nltk.data.path.append(nltk_dir)
nltk.data.path=[nltk_dir]
print((nltk.data.path))

############################################# Classification Preprocessing Code ###############################################

def clean(text):
    word_lem=WordNetLemmatizer()
    tokens=word_tokenize(text)
    lower=[word.lower() for word in tokens if len(word)>2 and word.isalpha()]
    lemmatized_text=[word_lem.lemmatize(word) for word in lower]
    return lemmatized_text
def vectorize(data,tfidf_vect_fit):
    x_tfidf=tfidf_vect_fit.transform(data)
    x_tfidf_df=pd.DataFrame(x_tfidf.todense(),columns=tfidf_vect_fit.get_feature_names())
    return x_tfidf_df
print(__name__)
import joblib
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer,word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
import joblib



############################################# Download Model From GCS ###############################################

if AIP_STORAGE_URI:
    #run command to download Model Directires from GCS
    pass

model=joblib.load("model.pkl")
tfidf_vect_fit=joblib.load("tfidf.pkl")



@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}

@app.post(AIP_PREDICT_ROUTE, response_model=ClassPredictions, response_model_exclude_unset=True)
async def predict(request:Request):
    body=await request.json()
    instances=body["instances"]
    print(body)
    out=[]
    ############## Prediction ###############
    ocr_df=pd.DataFrame({"data":[instances]},dtype=str)
    pred_value=model.predict(vectorize(ocr_df["data"].values.astype('U'),tfidf_vect_fit))
    print(pred_value[0])
    pred_conf=max(model.predict_proba(vectorize(ocr_df["data"].values.astype('U'),tfidf_vect_fit))[0])
    print(pred_conf)
    out.append(ClassPrediction(DocType=pred_value[0],Confidence=pred_conf))
    print(out)

    return ClassPredictions(prediction=out)

if __name__ == "__main__":
  import warnings

# Filter out the DeprecationWarning
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  uvicorn.run(app, host="0.0.0.0",port=8080)

