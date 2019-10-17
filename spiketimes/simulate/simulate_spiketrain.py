import numpy as np


def homogenous_poisson_process(rate, t_stop, t_start=0):
    """ Return timings of events arrising from a homogenous poisson process of a
    given rate (intensity) between a given start and stop times
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


def imhomogenous_poisson_process(time_rate: list, t_start=0):
    """Simulate an imhomogenous poisson process

    time_rate: list of tuples with elements (timeperiod, rate)
    t_stop: the maximum time allowed event time
    t_start: the start time
    """
    out = np.array([])
    for timeperiod, rate in time_rate:
        new_data = homogenous_poisson_process(
            rate=rate, t_stop=(timeperiod + t_start), t_start=t_start
        )
        out = np.concatenate([out, new_data])
        t_start += timeperiod
    return out
