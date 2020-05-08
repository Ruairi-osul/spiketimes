import numpy as np


def homogeneous_poisson_process(rate: float, t_stop: float, t_start: float = 0):
    """
    Simulate a poisson process.

    Args:
        rate: The intensity of the poisson process. The average number of events per second.
        t_stop: The time after which sampling stops
        t_start: The time from which sampling starts
    Returns:
        A numpy array containing event times in seconds
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


def imhomogeneous_poisson_process(time_rate: list, t_start: float = 0):
    """
    Simulate an imhomogeneous poisson process.

    Args:
        time_rate: List of tuples with elements (time_period, rate).
                   The first element in each tuple is the time period. The second element is the
                   average number of events per second during that time period.
        t_start: The time from which sampling starts
    Returns:
        A numpy array containing event timings in seconds
    """
    out = np.array([])
    for timeperiod, rate in time_rate:
        new_data = homogeneous_poisson_process(
            rate=rate, t_stop=(timeperiod + t_start), t_start=t_start
        )
        out = np.concatenate([out, new_data])
        t_start += timeperiod
    return out
