import numpy as np


def homogenous_poisson_process(rate: float, t_stop: float, t_start: float = 0):
    """
    Simulate a poisson process. Returns event timings in seconds.     
    
    params:
        rate: intensity of time poisson process. How many events per second.
        t_stop: the time after which sampling stops
        t_start: the time from which sampling starts
    returns:
        numpy array containing event times in seconds
    """
    beta = 1 / rate
    events = []
    last_event = 0
    while True:
        new_event = np.random.exponential(scale=beta)
        new_event += last_event
        if new_event > (t_stop - t_start):
            break
        else:
            events.append(new_event)
            last_event = new_event
    return np.array(events) + t_start


def imhomogenous_poisson_process(time_rate: list, t_start: float = 0):
    """
    Simulate an imhomogenous poisson process. Returns event times in seconds.

    params:
        time_rate: List of tuples with elements (time_period, rate).
                For each sucessive tuple, a homogenous process of duration time_period with 
                rate (intensity) rate will be appended
        t_stop: the time after which no sampling will occur
        t_start: the time at which sampling is started
    returns:
        a numpy array of event times in seconds
    """
    out = np.array([])
    for timeperiod, rate in time_rate:
        new_data = homogenous_poisson_process(
            rate=rate, t_stop=(timeperiod + t_start), t_start=t_start
        )
        out = np.concatenate([out, new_data])
        t_start += timeperiod
    return out
