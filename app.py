from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__, static_folder='static')

# Enable CORS
CORS(app)

import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-GUI backend


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mia Khalifa"})

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio_optimization():
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
        duration = data['duration']

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
            ef.efficient_return(user_value)
            cleaned_weights = ef.clean_weights()

            # Calculate Volume for each stock
            Volume_for_each_stock = dict({k: round(v * volume, 2) for k,v in cleaned_weights.items()})

            # Convert the OrderedDict to a regular dict for JSON response
            weights_dict = dict(cleaned_weights)

        elif (constraint == "Maximise return for a given risk with L2 regularisation"):
            # Perform portfolio optimization
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg)
            ef.efficient_risk(user_value)
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

        # Prepare and return the response
        response = {
            "weights": weights_dict,
            "performance": ef.portfolio_performance(verbose=True),
            "volume for each stock": Volume_for_each_stock,
            "pie_chart_b64": pie_chart_b64,
            "constraint": constraint
        }
        return jsonify(response)
    
    except Exception as e:
        # Log the error
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/executions')
def executions():
    return jsonify({"message": "Executions Page"})

if __name__ == '__main__':
    # Consider using a more secure host and port in production
    app.run(host='0.0.0.0', port=5000, debug=True)
