# spiketimes

spiketimes is a small python package for simulating and analysing simple spiketrains.


## Simulation

Generate homogeneous or imhomogeneous poisson processes. Some people beleive that these processes model spiking behaviour of neurons. Homogeneous poisson processes model spike trains with constant intensity (or "firing rate"). Imhomogenous poisson processes model spike trains with time varying intensity.


## Plotting

Supports raster plots of single or multiple spiketrains as well as peri-stimulus time histograms (PSTHs). PSTHs plot the distrobution of latencies of spiking events relative to some event.

## Alignment

Often in spiketrain analysis, it is desirible to align spikes from one entity to another. For example, align spikes from one spiketrain to an event. Functions in this subpackage aid with this. 
