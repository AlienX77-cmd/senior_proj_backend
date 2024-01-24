# ================================== Utils ==================================

def is_market_open(time): #time[0] is hour and time[1] is minute
    if ( (time[0] < 10) | (time[0] == 12 and time[1] >= 30) | (time[0] == 13) | (time[0] == 14 and time[1] <= 30) | (time[0] > 16) ): return False
    elif time[0] == 15: return True

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

def plus_1_minute(time):
    time = list(time)
    time[1]+=1
    if time[1] == 60:
        time[1] == 0 # Set minute to 0
        time[0] += 1 # increase hour by 1
    return tuple(time)

# ===========================================================================