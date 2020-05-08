==============================
Spiketimes Pandas Tutorial
==============================

Pandas dataframes (dfs) provide a convienient way to store and manipulate complex data. Dataframes are a natural way to store grouped data such as collections of spiketrains belonging to different trials, neurons and animals. 

Spiketimes provides a wide range of functions to analyse neuroscience data stored in pandas dataframes. Data is always assumed to follow tidy data principals: one row per observation and one column per variable.

Generating Dataframes of Spiketrains
=======================================

Conversion from Numpy Arrays 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`spiketimes.df.conversion` moudle provides various functions to convert between numpy arrays and dfs.

To convert between a list of arrays and a df:

::

    >>> from spiketimes.df.conversion import list_to_df
    >>> from spiketimes.simulate import homogeneous_poisson_process
    >>> 
    >>> # generate list of 20 spiketrains
    >>> st_list = [homogeneous_poisson_process(rate=3, t_stop=60) for _ in range(20)]
    >>> df = list_to_df(st_list)
    >>> df.sample(3)
        spiketrain	spiketimes
    91	11	        31.853600
    55	14	        19.636683
    159	7	        48.439054


Simulate Spiketrains
~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`spiketimes.df.simulate` moudule provides functions for simulating spiketrains.

To simulate 50 spiketrains as homogeneous poisson processes:

::

    >>> from spiketimes.df.simulate import homogeneous_poisson_processes 
    >>> 
    >>> df = homogeneous_poisson_processes(rate=3, t_start=0, t_stop=20, n=50)
    >>> df.sample(3)
        spiketrain	spiketimes
    13	5	        2.965396
    38	9	        14.640657
    34	28	        8.546893


To simulate 50 spiketrains as with fluctuating firing rates:

::

    >>> from spiketimes.df.simulate import imhomogeneous_poisson_processes
    >>> 
    >>> time_rate = [
    >>>     (60, 2),
    >>>     (10, 10),
    >>>     (50, 3)
    >>> ]
    >>> df = imhomogeneous_poisson_processes(time_rate=time_rate, n=50, t_start=0)
    >>> df.sample(3)
    	spiketrain	spiketimes
    320	42	        109.499891
    98	3	        52.532897
    349	13	        106.532938


Generate Surrogate Spiketrains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`spiketimes.df.surrogates` module contains functions for generating spiketrain surrogates. Surrogates are are spiketrains which are similar share some properties with a parent spiketrain but are otherwise different. They are often used in resampling statistics to generate bootstrap replicates.


To generate multiple surrogates for each spiketrain in a dataframe:

::

    >>> from spiketimes.df.surrogates import shuffled_isi_spiketrains_by
    >>>
    >>> # simulate many parent spiketrains
    >>> df_parents = homogeneous_poisson_processes(rate=20, t_stop=20, n=10)
    >>> df_parents.sample(3)
        spiketrain	spiketimes
    246	2	        13.239410
    323	9	        14.699663
    192	4	        10.124832
    >>> # generate 500 shuffled ISI surrogates for each spiketrain
    >>> df_surrogates = shuffled_isi_spiketrains_by(
    >>>                         df_parents,
    >>>                         spiketimes_col="spiketimes",
    >>>                         by_col="spiketrain",
    >>>                         n=500
    >>>                         )
    >>> df_surrogates.sample(3)
            spiketrain	surrogate	spiketimes
    1130311	5       	395	        3.975811
    738878	3	        434	        12.215798
    719863	3	        386	        17.599182


To generate 50 jitter spiketrains for each parent spiketrain:

::

    >>> from spiketimes.df.surrogates import jitter_spiketrains_by
    >>> 
    >>> df_surrogates = jitter_spiketrains_by(
    >>>         df_parents, 
    >>>         jitter_window_size=1, 
    >>>         spiketimes_col="spiketimes", 
    >>>         by_col="spiketrain", 
    >>>         n=50)
    >>> df_surrogates.sample(3)
    	            spiketrain	surrogate	spiketimes
        45771	    2	        25	        18.159243
        172165	    9	        9	        17.885836
        155980	    8	        18	        11.078205


Aligning Data
=================

The :mod:`spiketimes.df.alignment` module contains functions for aligning data to events.

To align all data to an array of evenets:

::

    >>> from spiketimes.df.alignment import align_around
    >>> from spiketimes.df.simulate import homogeneous_poisson_processes
    >>> import numpy as np 
    >>> 
    >>> # generate events
    >>> events = np.cumsum(np.random.random(40) * 5)
    >>> 
    >>> # generate spiketrains
    >>> df = homogeneous_poisson_processes(rate=10, t_stop=120, n=50)
    >>> 
    >>> # align the spiketrains to the events
    >>> df = align_around(df, data_colname="spiketimes", events=events, t_before=1)
    >>> 
    >>> df.sample(3)
        spiketrain	spiketimes	    aligned
    297	33	        28.858288	    0.837890
    453	49	        44.039706	    1.588492
    274	42	        27.450466	    -0.569932

To align different sets of data to different sets of events:

::

    >>> from spiketimes.df.alignment import align_around_by
    >>> import pandas as pd
    >>> 
    >>> # generate spiketrains recorded accross 5 sessions
    >>> df_data = pd.concat(
    >>>     [
    >>>         homogeneous_poisson_processes(rate=3, t_stop=100, n=20).assign(session=i)
    >>>         for i in range(1, 6)
    >>>     ]
    >>> )
    >>> 
    >>> # generate 100 events across 5 sessions
    >>> df_events = pd.concat(
    >>>     [
    >>>         pd.DataFrame({"session": i, "spiketimes": np.random.randint(0, 100, size=100)})
    >>>         for i in range(1, 6)
    >>>     ]
    >>> )
    >>> 
    >>> print(df_data.sample(3), "\n")
                spiketrain  spiketimes     session
    271          10         92.705551        5
    281           7         90.418194        5
    227          19         85.093771        4 
    >>> print(df_events.sample(3))
        session  spiketimes
    19        4           80
    0         3           80
    85        3           68
    >>> # align data from each session to events from the corresponding session
    >>> df_data = align_around_by(df_data=df_data, 
    >>>                             df_data_data_colname="spiketimes", 
    >>>                             df_data_group_colname="session",
    >>>                             df_events=df_events.sort_values("spiketimes"),
    >>>                             df_events_event_colname="spiketimes",
    >>>                             df_events_group_colname="session")
    >>> 
    >>> # data aligned to events from that session.
    >>> df_data.sample(3)
        spiketrain	spiketimes	session	aligned
    117	2	        41.899730	5	    1.899730
    271	12	        75.683014	4	    0.683014
    19	13	        6.800494	5	    0.800494


Binning Data
================

The :mod:`spiketimes.df.binning` module contains functions for binning data.

To bin events into counts along a regular interval:

::

    >>> from spiketimes.df.binning import binned_spiketrain
    >>> from spiketimes.df.simulate import homogeneous_poisson_processes
    >>> 
    >>> # Generate some random spiketrains
    >>> df_data = homogeneous_poisson_processes(rate=4, t_stop=120, n=20)
    >>> df_data.head(3)
        spiketrain	spiketimes
    0	0	0.509930
    1	0	0.643373
    2	0	0.751396
    >>> # Count the number of spikes occuring every 0.5 seconds (2Hz sampling rate) per neuron
    >>> df_binned = binned_spiketrain(
    >>>     df=df_data, spiketimes_col="spiketimes", by_col="spiketrain", fs=2, t_start=0
    >>> )
    >>> df_binned.tail(3)
            spiketrain	time	spike_count
    4777	19	        118.0	3
    4778	19	        118.5	4
    4779	19	        119.0	2


To get event counts at user-specified bins per spiketrain.

::

    >>> from spiketimes.df.binning import binned_spiketrain_bins_provided
    >>> import numpy as np 
    >>> 
    >>> # specify bins
    >>> bins = np.arange(0.5, 110, 5)
    >>> 
    >>> # get counts of events in each bin by spiketrain 
    >>> binned = binned_spiketrain_bins_provided(df_data, bins=bins)
    >>> binned.head()
            spiketrain	bin	        counts
    0	    0	        0.50	    17
    1	    0	        5.50	    19
    2	    0	        10.5    	14
    3	    0	        15.5    	19
    4	    0	        20.5    	15

To get the closest event to each spiketrain (useful for assigning each spike to a trial):

::

    >>> from spiketimes.df.binning import which_bin
    >>> 
    >>> # get bin value and idx for corresponding bin for each event
    >>> binned = which_bin(df=df_data, bin_edges=bins)
    >>> binned.head()
            spiketrain	bin_idx	bin_values	spiketimes
    0	    0	        NaN	    NaN	        0.204892
    1	    0	        NaN	    NaN	        0.243031
    2	    0	        NaN	    NaN	        0.343491
    3	    0	        0.0	    0.5	        1.166362
    4	    0	        0.0	    0.5	        1.172659

To get spike counts following events:

::

    >>> from spiketimes.df.binning import spike_count_around_event
    >>> 
    >>> # generate 5 spiketrains
    >>> df_data = homogeneous_poisson_processes(rate=2, t_stop=120, n=5)
    >>> # generate some events
    >>> events = np.arange(5, 120, 5)
    >>> 
    >>> # get spike counts 0.5 seconds after each event per spiketrain 
    >>> df_counts = spike_count_around_event(df=df_data, events=events, binsize=0.5, spiketimes_col="spiketimes")
    >>> df_counts.head(4)

        spiketrain	event	counts
    0	0	        05	    1
    1	0	        10	    0
    2	0	        15	    0
    3	0	        20	    4

To get spike counts following events where different spiketrains have different sets of events. For example different event times and spiketrains from different sessions.

::

    >>> from spiketimes.df.binning import spike_count_around_event_by
    >>> import pandas as pd
    >>> 
    >>> # generate spiketrains recorded accross 5 sessions
    >>> df_data = pd.concat(
    >>>     [
    >>>         homogeneous_poisson_processes(rate=3, t_stop=100, n=20).assign(session=i)
    >>>         for i in range(1, 6)
    >>>     ]
    >>> )
    >>> df_data.head(3)
        spiketrain	spiketimes	session
    0	0	        0.048724	1
    1	0	        0.821620	1
    2	0	        1.283268    1
    >>> # generate events at slightly differnt times across 5 sessions
    >>> df_events = pd.concat(
    >>>     [
    >>>         pd.DataFrame({"session": i, "spiketimes": np.arange(2, 100, 3) + np.random.random() * i})
    >>>         for i in range(1, 6)
    >>>     ]
    >>> )
    >>> df_events.tail(3)
        session	    spiketimes
    30	5	        96.450331
    31	5	        99.450331
    32	5	        102.450331
    >>> # get spikecount 0.2s following each event per spiketrain recorded in that session
    >>> df_counts = spike_count_around_event_by(df_data=df_data, 
    >>>                             binsize=0.2, 
    >>>                             df_data_data_colname="spiketimes",
    >>>                             df_data_group_colname="session",
    >>>                             df_data_spiketrain_colname="spiketrain",
    >>>                             df_events=df_events,
    >>>                             df_events_event_colname="spiketimes",
    >>>                             df_events_group_colname="session")
    >>> df_counts.head(4)
        spiketrain	event	    counts	session
    0	0	        2.972297	0	    1
    1	0	        5.972297	0	    1
    2	0	        8.972297	0	    1
    3	0	        11.972297	1	    1


Statistics
================

The :mod:`spiketimes.df.statistics` module contains functions for calculating statistics on groups of spiketrains.

To calculate the mean firing rate of each spiketrain in a DataFrame:

::

    >>> from spiketimes.df.statistics import mean_firing_rate_by
    >>> from spiketimes.df.simulate import homogeneous_poisson_processes
    >>> 
    >>> df_spikes = homogeneous_poisson_processes(rate=5, t_stop=120, n=5)
    >>> df_mfr = mean_firing_rate_by(df=df_spikes, t_start=0, t_stop=120)
    >>> df_mfr.head(3)
        spiketrain	mean_firing_rate
    0	0	        5.166667
    1	1	        5.316667
    2	2	        4.758333

To calculate mean firing rate excluding periods of silence:

::

    >>> from spiketimes.df.statistics import mean_firing_rate_ifr_by
    >>> from spiketimes.df.simulate import imhomogeneous_poisson_processes
    >>> 
    >>> # generate 20 spiketrains with a firing rate of 10 Hz 120 second silent period
    >>> time_rate = [
    >>>     (120, 10),
    >>>     (320, 0.2),
    >>>     (120, 10)
    >>> ]
    >>> df_spikes2 = imhomogeneous_poisson_processes(time_rate=time_rate, n=20)
    >>> 
    >>> # calculate mean firing rate by spiketrain excluding silent periods
    >>> df_mfr2 = mean_firing_rate_ifr_by(df=df_spikes2, 
    >>>                                   fs=1, 
    >>>                                   exclude_below=0.5, 
    >>>                                   t_start=0)
    >>> df_mfr2.head(3)
    	spiketrain	mean_firing_rate_ifr
    0	0	        9.631736
    1	1	        9.669442
    2	2	        9.486311

To estimate "instantaneous" firing rate at a regular interval:

::

    >>> from spiketimes.df.statistics import ifr_by
    >>> from spiketimes.df.simulate import homogeneous_poisson_processes
    >>> 
    >>> # simulate 40 spiketrains with a 10Hz firing rate
    >>> df_spikes = homogeneous_poisson_processes(rate=10, t_stop=120, n=40)
    >>> 
    >>> # estimate the firing rate of each neuron twice every second from 0 to 120 seconds
    >>> df_ifr = ifr_by(df=df_spikes, 
    >>>                 fs=2, 
    >>>                 t_start=0,
    >>>                 t_stop=120)
    >>> df_ifr.head(3)
        spiketrain	time	ifr
    0	0	        0.0	    10.579114
    1	0	        0.5	    10.578063
    2	0	        1.0	    10.575974


To calculate the coefficient of variation of inter-spike-intervals for each spiketrain in a dataframe:

::

    >>> from spiketimes.df.statistics import cv_isi_by
    >>> from spiketimes.df.simulate import homogeneous_poisson_processes
    >>> 
    >>> df_spikes = homogeneous_poisson_processes(rate=5, t_stop=120, n=5)
    >>> df_cv = cv_isi_by(df_spikes)
    >>> df_cv.head(3)
	    spiketrain	cv_isi
    0	0	        0.964865
    1	1	        0.971645
    2	2	        1.010274


Correlating Spiketrains
==================================

The :mod:`spiketimes.df.correlate` module contains functions for correlaing spiketrains.

To calculate the autocorrelation histogram for each spiketrain in a DataFrame:

::

    >>> from spiketimes.df.simulate import homogeneous_poisson_processes
    >>> from spiketimes.df.correlate import auto_corr
    >>> 
    >>> df_spikes = homogeneous_poisson_processes(rate=3, t_stop=1200, n=10)
    >>> df_auto = auto_corr(df=df_spikes, 
    >>>              num_lags=50,
    >>>              spiketrain_col="spiketrain", 
    >>>              spiketimes_col="spiketimes")
    >>> df_auto.head(3)
	    spiketrain	time_bin	autocorrelation
    0	0	        -0.50	    119
    1	0	        -0.49	    108
    2	0	        -0.48	    92

To the cross correlation histogram between each spiketrain in a DataFrame:

::

    >>> from spiketimes.df.correlate import cross_corr
    >>> 
    >>> df_cc = cross_corr(df_spikes, 
    >>>                    binsize=0.1, 
    >>>                    num_lags=50)
    >>> df_cc.head(3)
        spiketrain_1	spiketrain_2	time_bin	crosscorrelation
    0	0	            1	            -5.0	    995
    1	0	            1	            -4.9	    999
    2	0	            1	            -4.8	    1068

To calculate spike count correlations between all pairs of neurons in a DataFrame:

::

    >>> from spiketimes.df.correlate import spike_count_correlation
    >>> 
    >>> df1 = spike_count_correlation(df_spikes, binsize=0.1)
    >>> df2 = spike_count_correlation(df_spikes, binsize=0.1, use_multiprocessing=True)
    >>> df1.head()
        spiketrain_1	spiketrain_2	R_spike_count
    0	0	            1	            -0.009033
    1	0	            2	            -0.009751
    2	0	            3	            0.015005
    3	0	            4	            0.000109
    4	0	            5	            -0.013520

To test significance of correlations:

::

    >>> from spiketimes.df.correlate import spike_count_correlation_test
    >>> 
    >>> df1 = spike_count_correlation_test(df, binsize=0.01, use_multiprocessing=True, max_cores=10, adjust_p=True)
    >>> df1.head()
        spiketrain_1	spiketrain_2	R_spike_count	p
    0	0	            1	            -0.003057	    2.0
    1	0	            2	            -0.001368	    2.0
    2	0	            3	            0.001874	    2.0
