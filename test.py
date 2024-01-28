import queue
import json
import threading
import sys
import pickle
import pika
import datetime
import numpy as np
import random
import math

# ================================== Utils ==================================

def is_market_open(time): 
    # Market is closed before 10 AM # Morning
    if time[0] < 10:
        return False
    # Market is closed between 12:30 PM and 1 PM # Lunch Break
    if time[0] == 12 and time[1] >= 30:
        return False
    # Market is closed at 1 PM
    if time[0] == 13:
        return False
    # Market is closed between 2 PM and 2:30 PM
    if time[0] == 14 and (time[1] == 0 or time[1] <= 30):
        return False
    # Market is closed after 4 PM
    if time[0] > 16:
        return False
    # If none of the above conditions are met, the market is open
    return True

def time_cmp(time1,time2): # t1 < t2 = -1, t1 == t2 = 0, t1 > t2 = 1
    # For Hours
    if time1[0] < time2[0]:
        return -1
    elif time2[0] < time1[0]:
        return 1
    # For Minutes
    elif time1[1] > time2[1]:
        return 1
    elif time1[1] < time2[1]:
        return -1
    return 0 # t1 == t2 case

# ===========================================================================

# Global variables for your algorithm
last_time = None
prices_data = {}
cumulative_volume = {}
cumulative_volumes = {}
ATO = {}
prices = {}
models = {}
mo_lo_changerate = 0.25

metadata = {
    "AOT": {
        "start_time" : (10,30),
        "end_time": (12,20),
        "want": 10000,
        "left": 10000,
        "market_volume":0,
        "market_value":0,
        "market_vwap":0,
        "plan":[],
        "my_volume":0,
        "my_value":0,
        "my_vwap":0
    }, 
    "BANPU": {
        "start_time" : (10,30),
        "end_time": (12,20),
        "want": 10000,
        "left": 10000,
        "market_volume":0,
        "market_value":0,
        "market_vwap":0,
        "plan":[],
        "my_volume":0,
        "my_value":0,
        "my_vwap":0
    },
    "CPALL": {
        "start_time" : (10,30),
        "end_time": (12,20),
        "want": 10000,
        "left": 10000,
        "market_volume":0,
        "market_value":0,
        "market_vwap":0,
        "plan":[],
        "my_volume":0,
        "my_value":0,
        "my_vwap":0
    },
    "PTT": {
        "start_time" : (10,30),
        "end_time": (12,20),
        "want": 10000,
        "left": 10000,
        "market_volume":0,
        "market_value":0,
        "market_vwap":0,
        "plan":[],
        "my_volume":0,
        "my_value":0,
        "my_vwap":0
    },
    "SCC": {
        "start_time" : (10,30),
        "end_time": (12,20),
        "want": 10000,
        "left": 10000,
        "market_volume":0,
        "market_value":0,
        "market_vwap":0,
        "plan":[],
        "my_volume":0,
        "my_value":0,
        "my_vwap":0
    },
}

# Message processing queue
message_queue = queue.Queue()

# Establish a connection to RabbitMQ server
connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost', heartbeat=30)
)
channel = connection.channel()

# Ensure that the queue exists with the correct properties
channel.queue_declare(queue='AllData', durable=True)

# Consumer callback that puts messages into the processing queue
def consumer_callback(ch, method, properties, body):
    message_queue.put(body)

# Message processing function
def message_processor(stop_event):
    global message_queue, last_time, prices_data, cumulative_volume, cumulative_volumes, ATO, prices, models, mo_lo_changerate, metadata
    
    while not stop_event.is_set():
        try:
            body = message_queue.get(timeout=1)  # Adjust timeout as needed
        except queue.Empty:
            continue
        
        message = json.loads(body)

        # ====== Extract Bid and Offer (Normally this file (MarketByPriceLog.txt) must come from Real-time API ======
        if "LimitOrderBook" in message:
            p = message
            bid = p["LimitOrderBook"]["Bid"]["Prices"]
            offer = p["LimitOrderBook"]["Offer"]["Prices"]
            if bid == None or len(bid) == 0: continue 
            if offer == None or len(offer) == 0: continue       
            prices_data[p["Symbol"]] = [ bid[0] / ( 10 ** p["PriceDecimal"] ), offer[0] / ( 10 ** p["PriceDecimal"] ) ] 
            # print("MarketByPriceLog")
        # ===========================================================================================================        
        elif "Price" in message and "Quantity" in message:
            # print("TradeTicker")

            trade_data = message

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
                
            # current_time = (time.hour, time.minute)
            if is_market_open(current_time) == False:
                continue

            if current_time != last_time:
                print(f"Time frame passed {last_time} -> {current_time}")
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

                    print(f"calculating for {s}") 
                        
                    start_time = metadata[s]["start_time"]
                    end_time = metadata[s]["end_time"]
                    
                    if time_cmp( start_time , current_time ) == -1 and  time_cmp( end_time , current_time ) == 1:
                        print("with in time frame")
                        # Calculating volume plan for each trade
                        upper = 0
                        lower = 0 
                        if len(cumulative_volumes[s]) >= 2:
                            # upper = V(t(i)) - V(t(i-1))
                            upper = cumulative_volumes[s][-1] - cumulative_volumes[s][-2] #-1 means t(i) => current and -2 mean t(i-1) => previous one
                                                
                            model_filename = f'models/{s}.pkl'
                            if s not in models:
                                with open(model_filename, 'rb') as model_file:
                                    model = pickle.load(model_file)
                                    models[s] = model
                                    
                            loaded_model = models[s][current_time][end_time]
                            
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
                                print("1", metadata)

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
                            print("2", metadata)
                            
            last_time = current_time

# Set up the consumer in a separate thread
def consumer_thread(stop_event):
    channel.basic_qos(prefetch_count=1000)
    channel.basic_consume(queue='AllData', on_message_callback=consumer_callback, auto_ack=True)
    while not stop_event.is_set():
        channel.connection.process_data_events(time_limit=1)  # Non-blocking event processing

# Main execution
if __name__ == "__main__":
    stop = threading.Event()

    # Start the consumer and processor threads
    consumer_t = threading.Thread(target=consumer_thread, args=(stop,))
    processor_t = threading.Thread(target=message_processor, args=(stop,))
    
    consumer_t.start()
    processor_t.start()

    try:
        # Keep the main thread alive until a keyboard interrupt
        while consumer_t.is_alive() or processor_t.is_alive():
            consumer_t.join(timeout=1)
            processor_t.join(timeout=1)
    except KeyboardInterrupt:
        # Signal the threads to stop and clean up
        stop.set()
        consumer_t.join()
        processor_t.join()
        connection.close()
        sys.exit(0)
