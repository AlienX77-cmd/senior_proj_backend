import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-GUI backend

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions, plotting

import numpy as np 
import pandas as pd
from datetime import datetime, timedelta

from io import BytesIO
import base64

import threading
import time

from Volume_Pred import Vol_Pred
from Adaptive_OE import Adapt_OE
# from Adaptive_rabbit import Adapt_Rabbit

app = Flask(__name__, static_folder='static')

# Enable CORS
CORS(app)

metadata = {}

# # Function to run in the background thread
# def execution(metadata):
#     metadata = Adapt_OE(metadata)
#     return metadata

# # Thread target function
# def start_execution_thread():
#     # Pass the global metadata to the execution function
#     global metadata
#     # Start the Adapt_Rabbit function in a separate thread
#     thread = threading.Thread(target=execution, args=(metadata,))
#     thread.daemon = True  # This ensures the thread will be closed when the main process exits
#     thread.start()

# @app.before_first_request
# def before_first_request():
#     # This will start the execution thread when the first request hits the Flask server
#     start_execution_thread()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mia Khalifa"})

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio_optimization():
    # global metadata  # Add this line to use the global metadata variable
    try:
        # Get data from POST request
        data = request.get_json()  # This will parse the incoming JSON payload
        
        # Validate the incoming JSON data
        if 'tickers' not in data or 'inputValue' not in data:
            return jsonify({"error": "Missing data in payload"}), 400

        tickers = data['tickers']
        constraint = data['constraint']
        user_value = data['inputValue']
        volume = data['totalVolume']
        duration_str = data['duration']  # e.g., '3h'
        duration_hours = int(duration_str.replace('h', ''))  # Remove the 'h' and convert to int

        user_value = float(user_value/100)
        volume = int(volume)

        # Add .BK to each stock's ticker
        for i in range(len(tickers)): tickers[i] += ".BK"
        
        # Retrieve the stock data via yFinance API
        ohlc = yf.download(tickers, period="max")
        adj_prices = ohlc["Adj Close"].dropna()

        # Calculate the covariance matrix
        S = risk_models.sample_cov(adj_prices, frequency=252)

        # Estimate the expected returns
        mu = expected_returns.capm_return(adj_prices)

        if (constraint == "Minimise risk for a given return"): 
            # Perform portfolio optimization
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg)

            # Setup for creating Efficient Frontier
            fig, ax = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

            # Minimize risk for the give return => Expected return = m (from the first constraint)
            ef.efficient_return(user_value)

            # Create Efficient Frontier
            expected_annual_return, annual_volatility, sharpe_ratio = performance = ef.portfolio_performance(verbose=True)
            ax.scatter(
            annual_volatility,  # Volatility
            expected_annual_return, # Return
            color="red",
            marker="o",
            label="Optimal Point",
            )
            ax.legend()

            # Save the efficient frontier plot to a BytesIO object and encode it to base64
            frontier_io = BytesIO()
            plt.savefig(frontier_io, format='png', bbox_inches='tight')
            plt.close(fig)
            frontier_io.seek(0)
            frontier_b64 = base64.b64encode(frontier_io.read()).decode('utf-8')

            # Cleaned Weights
            cleaned_weights = ef.clean_weights()

            # Calculate Volume for each stock
            Volume_for_each_stock = dict({k: round(v * volume, 2) for k,v in cleaned_weights.items()})

            # Convert the OrderedDict to a regular dict for JSON response
            weights_dict = dict(cleaned_weights)

        elif (constraint == "Maximise return for a given risk with L2 regularisation"):
            # Perform portfolio optimization
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg)

            # Setup for creating Efficient Frontier
            fig, ax = plt.subplots()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

            # Maximise return for a given risk, with L2 regularisation
            ef.efficient_risk(user_value)

            # Create Efficient Frontier
            expected_annual_return, annual_volatility, sharpe_ratio = performance = ef.portfolio_performance(verbose=True)
            ax.scatter(
            annual_volatility,  # Volatility
            expected_annual_return, # Return
            color="red",
            marker="o",
            label="Optimal Point",
            )
            ax.legend()

            # Save the efficient frontier plot to a BytesIO object and encode it to base64
            frontier_io = BytesIO()
            plt.savefig(frontier_io, format='png', bbox_inches='tight')
            plt.close(fig)
            frontier_io.seek(0)
            frontier_b64 = base64.b64encode(frontier_io.read()).decode('utf-8')

            # Cleaned Weights
            cleaned_weights = ef.clean_weights()

            # Calculate Volume for each stock
            Volume_for_each_stock = dict({k: round(v * volume, 2) for k,v in cleaned_weights.items()})

            # Convert the OrderedDict to a regular dict for JSON response
            weights_dict = dict(cleaned_weights)
        
        else: return
       
        # Create the pie chart using BytesIO
        series = pd.Series(cleaned_weights)
        fig, ax = plt.subplots()
        series.plot.pie(ax=ax)
        pie_chart_io = BytesIO()
        plt.savefig(pie_chart_io, format='png')
        plt.close(fig)
        pie_chart_io.seek(0)
        pie_chart_b64 = base64.b64encode(pie_chart_io.read()).decode('utf-8')  # Encode as base64

        # Update metadata with the new values
        current_time = datetime.now()
        for ticker, volume in Volume_for_each_stock.items():
            symbol = ticker.replace(".BK", "")  # Remove the .BK if present
            
            # Initialize metadata for the symbol if it doesn't exist
            if symbol not in metadata:

                metadata[symbol] = {
                    "start_time": (0, 0),
                    "end_time": (0, 0),
                    "want": 0,
                    "left": 0,
                    "market_volume": 0,
                    "market_value": 0,
                    "market_vwap": 0,
                    "plan": [],
                    "my_volume": 0,
                    "my_value": 0,
                    "my_vwap": 0,
                }

            # Now update the metadata for the symbol
            metadata[symbol]["start_time"] = (current_time.hour, current_time.minute)
            # metadata[symbol]["start_time"] = (10, 35)
            end_time = current_time + timedelta(hours=duration_hours)
            metadata[symbol]["end_time"] = (end_time.hour, end_time.minute)
            # metadata[symbol]["end_time"] = (16, 30)
            metadata[symbol]["want"] = volume
            metadata[symbol]["left"] = volume

        # Training the Volume Prediction Model
        stock_symbols = list(metadata.keys())
        Vol_Pred(stock_symbols)
        print("===== Finish training Volume Prediction Models for all selected stocks =====")

        # Prepare and return the response
        response = {
            "weights": weights_dict,
            "performance": ef.portfolio_performance(verbose=True),
            "volume for each stock": Volume_for_each_stock,
            "pie_chart_b64": pie_chart_b64,
            "ef": frontier_b64,
            "constraint": constraint
        }
        return jsonify(response)
    
    except Exception as e:
        # Log the error
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/executions', methods=['GET', 'POST'])
def executions():
    return jsonify(metadata)

# def background(a,b):
#     while not b.is_set():
#         print("Hello world ",a)
#         time.sleep(1)

if __name__ == '__main__':
    # a = threading.Event()
    # t = threading.Thread( target=background , args=(1,a))
    # Consider using a more secure host and port in production
    # t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
    # print("Hello world")
    # a.set()
    # t.join()
