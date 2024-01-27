import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
import pickle

from Read_TrainingData import Read_TrainingData

# stock_symbols = list(metadata.keys())

def Vol_Pred(stock_symbols):
    # Create a dictionary to hold all models for each stock symbol
    models_by_symbol = {}

    # Create Sample Data via Read_TrainingData
    Sample_Data = Read_TrainingData(stock_symbols)

    # Train model for each stock (each pickle file represents 1 stock (which consist of summation of i = 1 to 250 models))
    for symbol in stock_symbols:
        models_by_symbol[symbol] = {}
    
        model_filename = f'models/{symbol}.pkl'
    
        # If the model file already exists,
        if os.path.exists(model_filename): continue  # Skip to the next symbol
    
        for start_time in Sample_Data[symbol]:
            models_by_symbol[symbol][start_time] = {}
        
            for end_time in Sample_Data[symbol][start_time]:
                train_data = np.array(Sample_Data[symbol][start_time][end_time])
                trainx = train_data[:, :2]
                trainy = train_data[:, 2]

                model = LinearRegression().fit(trainx, trainy)
                models_by_symbol[symbol][start_time][end_time] = model

        # Save all models for the symbol to a single pickle file (only if any were trained)
        if any(models_by_symbol[symbol].values()):  # Check if any models were trained
            with open(f'models/{symbol}.pkl', 'wb') as model_file:
                pickle.dump(models_by_symbol[symbol], model_file)

# models/symbol.pkl
# {
#     (10,30) : {
#         (22,30) : model
#     }
# }