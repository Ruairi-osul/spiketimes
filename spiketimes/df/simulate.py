from .conversion import list_to_df
import spiketimes.simulate


def homogeneous_poisson_processes(
    rate: float, t_stop: float, n: int, t_start: float = 0,
):
    """
    Simulate n spiketrains as homogeneous poisson processes.

    Each spiketrain has the same characteristics.

    Args:
        rate: intensity of the poisson processes. How many events per second.
        t_stop: the time after which sampling stops
        n: the number of spiketrains to simulate
        t_start: the time from which sampling starts
    Returns:
        A pandas dataframe containing the simulated spiketrains
        with columns {"spiketrain", "spiketimes"}
    """
    return list_to_df(
        [
            spiketimes.simulate.homogeneous_poisson_process(
                rate=rate, t_stop=t_stop, t_start=t_start
            )
            for _ in range(n)
        ]
    )


def imhomogeneous_poisson_processes(time_rate: list, n: int, t_start: float = 0):
    """
    Simulate n spiketrains as imhomgeneous poisson processes.

    Each spiketrain has the same time-varying firing rates.

    Args:
        time_rate: list of tuples containing the timespan and rate of each
                   firing rate (time_span, firing_rate).
        n: number of spiketrains to generate
        t_start: if specified, starts the first time interval in time_rate from this time.
    Returns:
        A pandas dataframe containing the simulated spiketrains with columns {"spiketrain", "spiketimes"}
    """
    return list_to_df(
        [
            spiketimes.simulate.imhomogeneous_poisson_process(
                time_rate=time_rate, t_start=t_start
            )
            for _ in range(n)
        ]
    )
