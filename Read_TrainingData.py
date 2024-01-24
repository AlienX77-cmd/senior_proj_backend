import os
import json
import pandas as pd
import datetime

def Read_TrainingData(stock_symbols):
    # Initialize a dictionary to hold sample data for each stock
    Sample_Data = {symbol: {} for symbol in stock_symbols}

    # Loop over the stock symbols and dates to read and prepare data
    dates = os.listdir("./data")
    for symbol in stock_symbols:
        for date in dates:
            with open(f'data/{date}/TradeTickerLog.txt', 'r') as file:
                # Filter out the lines that correspond to the current symbol
                data = [json.loads(line.strip()) for line in file if json.loads(line.strip())["Symbol"] == symbol]

            # --------------------- Data cleaning and structuring ----------------------
            df = pd.DataFrame(data)
            df.Price = df.Price / (10 ** df.PriceDecimal) # Decimal Division Problem
            df = df[(df.Symbol == symbol) & (df.DealSource != "Trade Report")].copy()
            # --------------------------------------------------------------------------

            # --------------------------- Extract ATO volume ---------------------------
            ATO = df[df.DealSource == "Auction"]
            if len(ATO) > 0 and datetime.datetime.fromtimestamp(ATO.iloc[0].Timestamp/1e9).hour <= 11: # divide by nano
                ATO = ATO.iloc[0].Quantity # iloc[0] to select the first row which the stock which is traded at ATO time\
            else: 
                raise Exception("No ATO for today")
            # --------------------------------------------------------------------------

            # -------------- Filter and choose only DealSource != Auction --------------
            df = df[df.DealSource != "Auction"]
            # Create Cumulative Sum for Quantity
            df["CumSumQuantity"] = df["Quantity"].cumsum()
            # --------------------------------------------------------------------------

            # Extract total volume of the chosen stock that are traded for the whole day 
            if len(df) > 0:
                total_volume = df.iloc[-1].CumSumQuantity
            else:
                raise Exception("No volume traded today")
            # --------------------------------------------------------------------------

            # ------------ Transform the Unix time to normal readable time -------------
            df["Date"] = df["Timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x/1e9))
            df["Hour"] = df["Date"].apply(lambda x: ( x + datetime.timedelta(minutes=1) ).hour) # timedelta mean plus (+1 minute)
            df["Minute"] = df["Date"].apply(lambda x: ( x + datetime.timedelta(minutes=1) ).minute)
            # --------------------------------------------------------------------------

            # ----------------------- Group by Hour and Minute -------------------------
            group_df = df.groupby(["Hour","Minute"]).last()
            # --------------------------------------------------------------------------

            # -------------------------- Creating Sample Data --------------------------
            for i in range(len(group_df)):
                start_time = group_df.iloc[i].name
                if start_time not in Sample_Data[symbol]:
                    Sample_Data[symbol][start_time] = {}
                for j in range(i+1, len(group_df)):
                    end_time = group_df.iloc[j].name
                    if end_time not in Sample_Data[symbol][start_time]:
                        Sample_Data[symbol][start_time][end_time] = []
                    # Append the sample data for the symbol
                    # For each time frame (10:00 - 10:01 or 10:00 - 10:02 etc...), append ATO, Cumulative Volume at start time (from t0 to ti (current or start time)), and the change in cumulative volume between start time and end time
                    Sample_Data[symbol][start_time][end_time].append(
                        (ATO, group_df.iloc[i].CumSumQuantity, group_df.iloc[j].CumSumQuantity - group_df.iloc[i].CumSumQuantity)
                    )
            # ---------------------------------------------------------------------------
    return Sample_Data