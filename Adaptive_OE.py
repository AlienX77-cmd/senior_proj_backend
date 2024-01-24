import json
import pandas as pd
import numpy as np

import os
import datetime
import pickle 

import random
import math

from Utils import *

def Adapt_OE(metadata):
    # ===================== Read Real-time Data ====================
    dates = os.listdir("./data")
    for date in dates:
        data = []
        # Open the file in read mode ('r')
        with open(f'data/{date}/TradeTickerLog.txt', 'r') as file:
            # Read each line in the file one by one
            for line in file:
                # Print each line
                line = line.strip()  # strip() removes the newline character at the end
                data.append( json.loads(line) )

    # Open the file in read mode ('r')
    market_data = []
    with open('data/20240118/MarketByPriceLog.txt', 'r') as file:
    # Read each line in the file one by one
        for line in file:
        # Print each line
            line = line.strip()  # strip() removes the newline character at the end
            market_data.append( json.loads(line) )
    # ==============================================================

    # Extract Bid and Offer (Normally this file (MarketByPriceLog.txt) must come from Real-time API 
    # and it will come in every T second), so we will have to implement python thread, but I haven't never use thread before, so I don't know how
    prices_data = {}
    for p in market_data:
        bid = p["LimitOrderBook"]["Bid"]["Prices"] 
        offer = p["LimitOrderBook"]["Offer"]["Prices"]
        if bid == None or len(bid) == 0: continue
        if offer == None or len(offer) == 0: continue
        prices_data[p["Symbol"]] = [ bid[0] / ( 10 ** p["PriceDecimal"] ) , offer[0] / ( 10 ** p["PriceDecimal"] ) ] 

    # ==============================================================

    cumulative_volume = {}
    cumulative_volumes = {}
    ATO = {}
    prices = {}

    mo_lo_changerate = 0.25

    last_time = None
    for trade_data in data:

        # Convert Unix Time to normal readable time
        time = datetime.datetime.fromtimestamp(trade_data["Timestamp"]/1e9)
        product = trade_data["FinancialProduct"]
        symbol = trade_data["Symbol"]

        current_time = (time.hour, time.minute)

        # Find Common Stock Only
        if product != "CS":
            continue

        # Find ATO
        if trade_data["DealSource"] == "Auction" and time.hour < 11:
            ATO[symbol] = trade_data["Quantity"]

        # Find Current Volume
        if trade_data["DealSource"] == "Auto match":
            if symbol not in cumulative_volume:
                cumulative_volume[symbol] = 0
                cumulative_volumes[symbol] = []
            cumulative_volume[symbol] += trade_data["Quantity"] # Cumulative volume เก็บ volume ของแต่ละ trade ไปเรื่อยๆ
            if symbol not in prices:
                prices[symbol]=[[]]
            prices[symbol][-1].append(
                [trade_data["Price"]/ ( 10 ** trade_data["PriceDecimal"] ),trade_data["Quantity"]]
            ) # Same as prices[symbol][:]
            if symbol in metadata:
                start_time = metadata[symbol]["start_time"]
                end_time = metadata[symbol]["end_time"]
                # t1 < t2 = -1, t1 == t2 = 0, t1 > t2 = 1
                # start_time < current_time < end_time
                if time_cmp( start_time , current_time ) == -1 and  time_cmp( end_time , current_time ) == 1:
                    metadata[symbol]["market_volume"] += trade_data["Quantity"]
                    metadata[symbol]["market_value"] += trade_data["Price"] / ( 10 ** trade_data["PriceDecimal"] ) * trade_data["Quantity"]
                    metadata[symbol]["market_vwap"] = metadata[symbol]["market_value"] / metadata[symbol]["market_volume"]
        else:
            continue

    #     current_time = (time.hour, time.minute)
        if is_market_open(current_time) == False:
            continue
        if current_time != last_time:
            for s in prices: # for transforming the data structure of the last price in to a list of array
                to_append = []
                if len( prices[s][-1] ) > 0:
                    to_append.append( prices[s][-1][-1] ) # extract last price element into list of array
                    del prices[s][-1][-1] # remove last price element
                prices[s].append(to_append) # add the last price element in the form of list of array back
            # Trying to get V(t(i)) and V(t(i-1))
            for s in cumulative_volume:
                if s == symbol:
                    cumulative_volumes[s].append( cumulative_volume[s] - trade_data["Quantity"] ) # difference of current cumulative volume and trade quantity
                else:
                    cumulative_volumes[s].append( cumulative_volume[s] )
                if s not in metadata:
                    continue 
                start_time = metadata[s]["start_time"]
                end_time = metadata[s]["end_time"]
                if time_cmp( start_time , current_time ) == -1 and  time_cmp( end_time , current_time ) == 1:
                    # Calculating volume plan for each trade
                    upper = 0
                    lower = 0 
                    if len(cumulative_volumes[s]) >= 2:
                        # upper = V(t(i)) - V(t(i-1))
                        upper = cumulative_volumes[s][-1] - cumulative_volumes[s][-2] #-1 means t(i) => current and -2 mean t(i-1) => previous one

                        # To load a model later for a given symbol and time frame
                        model_filename = f'models/{s}_{start_time}_{end_time}.pkl'
                        with open(model_filename, 'rb') as model_file:
                            loaded_model = pickle.load(model_file)

                        # model_filename = f'models/{s}.pkl'
                        # if s not in models:
                        #     with open(model_filename, 'rb') as model_file:
                        #         model = pickle.load(model_file)
                        #         models[s] = model
                        
                        # loaded_model = models[s][start_time][end_time]

                        # models/symbol.pkl
                        # {
                        #     (10,30) : {
                        #         (22,30) : model
                        #     }
                        # }

                        # lower = V(hat) - V(t(i-1)) ; V(hat) = V(t(i)) + V(predict) => Volume ที่เกิดจากการ trade จนถึงเวลาปัจจุบันรวมกับ Volume ของทั้งวันที่อาจจะเกิดขึ้นที่ได้จากการทำนายจนจบวัน
                        lower = cumulative_volumes[s][-1] + loaded_model.predict(
                            [[ATO[s],cumulative_volumes[s][-1]]]
                        )[0] - cumulative_volumes[s][-2]

                        # V(t(i)) - V(t(i-1)) / V(hat) - V(t(i-1))
                        ratio = upper/lower

                        need_to_execute_lo = 0
                        mo_price = prices_data[s][1]
                        lo_price = prices_data[s][0]
                        last_trade = np.array(prices[s][-2])

                        if len(last_trade) == 0:
                            continue
                        offer_volume = last_trade[last_trade[:,0] >= mo_price][:,1].sum() # for calculating MO
                        bid_volume = last_trade[last_trade[:,0] < mo_price][:,1].sum() # for calculating LO

                        # ------------------------- NEED SVM for performing price prediction --------------------------
    #                     up_or_down = 0 # -> svm predict , up = 1 and don't go up or down = 0 and down = -1
                        up_or_down = random.choice([-1, 0, 1])
                        # ---------------------------------------------------------------------------------------------

                        lo_ratio = ( bid_volume ) / ( bid_volume + offer_volume ) # (total volume = bid_volume + offer_volume)
                        mo_ratio = ( offer_volume ) / (bid_volume + offer_volume)

                        if up_or_down == 1: # up
                            mo_ratio = mo_ratio * ( 1 + mo_lo_changerate ) # max at 1
                            lo_ratio = 1 - mo_ratio 
                        elif up_or_down == -1: # down
                            lo_ratio = lo_ratio * ( 1 + mo_lo_changerate ) # max at 1
                            mo_ratio = 1 - lo_ratio 

                        # Lowest is 0, no negative value of trade volume that will be placed
                        need_to_execute_mo = max(metadata[s]["left"] * ratio * mo_ratio, 0)
                        need_to_execute_lo = max(metadata[s]["left"] * ratio * lo_ratio, 0)

                        if (need_to_execute_mo > 0) or (need_to_execute_lo > 0):
                            # round to ...00 only
                            need_to_execute_mo = math.ceil(need_to_execute_mo / 100) * 100
                            need_to_execute_lo = math.ceil(need_to_execute_lo / 100) * 100
                            metadata[s]["left"] = metadata[s]["left"] - need_to_execute_lo - need_to_execute_mo
                            metadata[s]["my_volume"] += need_to_execute_mo + need_to_execute_lo
                            metadata[s]["my_value"] += need_to_execute_mo * mo_price + need_to_execute_lo * lo_price
                            metadata[s]["my_vwap"] =  metadata[s]["my_value"] /  metadata[s]["my_volume"]
                            metadata[s]["plan"].append( {
                                "LO_VOLUME": need_to_execute_lo,
                                "MO_VOLUME": need_to_execute_mo,
                                "LO_PRICE": lo_price,
                                "MO_PRICE": mo_price,
                                "TIME": current_time
                            })
                # At the end 
                if time_cmp( end_time, current_time ) == 0:
                        need_to_execute_mo = metadata[s]["left"]
                        need_to_execute_lo = 0
                        mo_price = prices_data[s][1]
                        lo_price = prices_data[s][0]
                        metadata[s]["left"] = metadata[s]["want"] - metadata[s]["my_volume"]
                        metadata[s]["plan"].append( {
                            "LO_VOLUME": 0,
                            "MO_VOLUME": need_to_execute_mo,
                            "LO_PRICE": lo_price,
                            "MO_PRICE": mo_price,
                            "TIME": current_time
                        })


        last_time = current_time
        
    return metadata