import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import pickle

from Read_TrainingData import Read_TrainingData

# stock_symbols = list(metadata.keys())

def Vol_Pred(stock_symbols):
    models = {symbol: {} for symbol in stock_symbols}
    # Now, train models for each time frame and each stock
    Sample_Data = Read_TrainingData(stock_symbols)
    for symbol in stock_symbols:
        for start_time in Sample_Data[symbol]:
            for end_time in Sample_Data[symbol][start_time]:
                model_filename = f'models/{symbol}_{start_time}_{end_time}.pkl'

                # Check if the model file already exists
                if not os.path.exists(model_filename):  # Model hasn't been created yet
                    train_data = np.array(Sample_Data[symbol][start_time][end_time])
                    trainx = train_data[:, :2]  # ALL rows but only first 2 columns as features
                    trainy = train_data[:, 2]   # ALL rows but only the third column as target variable

                    if start_time not in models[symbol]:  # Check if the model for start_time exists for the symbol
                        models[symbol][start_time] = {}

                    # Train and save the model for the symbol and time frame if it doesn't exist
                    models[symbol][start_time][end_time] = LinearRegression().fit(trainx, trainy)
                    with open(model_filename, 'wb') as model_file:
                        pickle.dump(models[symbol][start_time][end_time], model_file)
                else:
                    # Model is already existed in the models folder, so we skip the training
                    pass