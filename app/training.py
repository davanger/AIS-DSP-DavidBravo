from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

def ridge_reg(x_train, y_train, alpha=0.5, normalize=True):
    model = Ridge(alpha=alpha, normalize=normalize)
    model.fit(x_train, y_train)
    
    return model

def build_model(regression_fn,                
                target_col, 
#                 names_of_x_cols, 
                dataframe, 
                test_frac=0.2,
                show_plot_Y=False,
                show_plot_scatter=False,
                train_on_full_dataset=False):
    
    y = target_col
    X = dataframe
    
    if not train_on_full_dataset:
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_frac)
    else:
        x_train, x_val, y_train, y_val  = X, X, y, y

    model = regression_fn(x_train, y_train)
    
    return {
            'model': model,
            'data_split': (x_train, x_val, y_train, y_val)
           }     

def train(preprocessed_dataframe: pd.DataFrame, target, train_on_full_dataset=False):
    preprocessed_dataframe = preprocessed_dataframe.copy()
    preprocessed_dataframe.head()
    
    model_data = build_model(regression_fn=ridge_reg,
                        target_col=target,
                        dataframe=preprocessed_dataframe,
                        train_on_full_dataset=train_on_full_dataset,
                        test_frac=0.2,
                        show_plot_Y=True
                        )
    model = model_data["model"]
    data_split = model_data["data_split"]
    
    return (model, data_split)

def evaluate(model, data_split, show_plot_Y=False, show_plot_scatter=False):
    x_train, x_val, y_train, y_val = data_split
    y_pred = model.predict(x_val)
    
    rms = mean_squared_error(np.log(y_val), np.log(y_pred), squared=False)
    
    print("Training_score : " , model.score(x_train, y_train))
    print("Test_score : ", r2_score(y_val, y_pred))
    print("Log RMSE : ", rms)


    if show_plot_Y == True:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plt.plot(y_pred, label='Predicted')
        plt.plot(y_val.values, label='Actual')
        
        plt.ylabel(name_of_y_col)

        plt.legend()
        plt.show()

    if show_plot_scatter == True:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plt.scatter(x_val, y_val)
        plt.plot(x_val, y_pred, 'r')
        
        plt.legend(['Predicted line','Observed data'])
        plt.show()
    
    return {
            'training_score': model.score(x_train, y_train),
            'test_score': r2_score(y_val, y_pred),
            'training_log_rmse': rms,
           }
