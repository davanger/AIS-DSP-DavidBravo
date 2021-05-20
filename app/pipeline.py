# from app.training import evaluate, train
# from app.preprocessing import preprocess
import pickle
import pandas as pd
import numpy as np
import app.preprocessing
import app.training

target_column = "SalePrice"

def save_model(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))

def load_model(filepath):
    loaded_model = pickle.load(open(filepath, 'rb'))
    return loaded_model

def create_submission(Ids, prediction, data_dir):
    submission_df = pd.DataFrame()
    submission_df['Id'] = Ids
    submission_df['SalePrice'] = prediction
    submission_df.to_csv(data_dir / "submission.csv",index=False)

def training_pipeline(dataframe):
    target = dataframe[target_column]
    preprocessed_dataframe, one_hot_encoder = app.preprocessing.preprocess(target_column=target_column, dataframe=dataframe)
    model, data_split = app.training.train(preprocessed_dataframe, target, train_on_full_dataset=True)
    app.training.evaluate(model, data_split)
    return (model, one_hot_encoder)
    
def inference_pipeline(dataframe, model_path, encoder, data_dir):
    preprocessed_dataframe = dataframe.copy()
    Ids = dataframe.index
    model = load_model(model_path)
    preprocessed_dataframe = app.preprocessing.preprocess(target_column=target_column, dataframe=preprocessed_dataframe, encoder=encoder)[0]
    create_submission(Ids, model.predict(preprocessed_dataframe), data_dir)
    