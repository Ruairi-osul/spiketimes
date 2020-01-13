# spiketimes

spiketimes is a small python package for analysing electrophysiology data.


## Installation

Spiketimes is currently under development. If you would like to install a pre reslease build type the following into the terminal.

```
$ git clone https://github.com/Ruairi-osul/spiketimes.git
$ cd spiketimes
$ pip install -e .
```

## Quickstart


### Simulating Spiketrains

Generate simple poisson spiketrains.

#### Examples:
Generate a numpy array of spiketimes from a poisson spiketrain

```
>>> from spiketimes.simulate import homogenous_poisson_process
>>> spiketrain = homogenous_poisson_process(rate=10, t_start=0, t_stop=18000)
```

Simulate 10 neurons firing for 120 seconds at 5Hz, 50s at 10Hz and 40s at 7Hz

```
>>> from spiketimes.simulate import imhomogenous_poisson_process
>>> time_rate = [(120, 5), (50, 10), (40, 7)]
>>> spiketrains = [imhomogenous_poisson_process(time_rate=time_rate) for _ in range(10)]
```

### Statistics

Various spiketrain discrictive statistics and correlation metrics.


#### Examples:

Calculate the coefficient of variation of interspike intervals of a spiketrain

```
>>> from spiketimes.statistics import cv_isi
>>> from spiketimes.simulate import homogenous_poisson_process
>>>
>>> spiketrain_1 = homogenous_poisson_process(rate=6, t_start=0, t_stop=18000)
>>> cv_isi(spiketrain_1)
```

Calculate the cross-correlation between two neurons

```
>>> from spiketimes.statistics import cross_corr
>>> from spiketimes.simulate import homogenous_poisson_process
>>>
>>> spiketrain_1 = homogenous_poisson_process(rate=10, t_start=0, t_stop=18000)
>>> spiketrain_1 = homogenous_poisson_process(rate=15, t_start=0, t_stop=18000)
>>> lags, cross_corr_values = cross_corr(spiketrain_1, spiketrain_1, fs=100, num_lags=100)
```


