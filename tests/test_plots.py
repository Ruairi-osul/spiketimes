import pytest
import matplotlib.pyplot as plt
import numpy as np
from spiketimes.plots import (
    psth,
    add_event_vlines,
    raster_single_neuron,
    raster_multiple_neurons,
)


def get_points(ax):
    # helper function to return plotted data given a matplotlib scatter axis
    data = []
    for d in ax.collections:
        d.set_offset_position("data")
        data.append(d.get_offsets().data)
    return data


class TestPSTH:
    pass


class TestAddEventVLines:
    pass


class TestRasterSingleNeuron:
    def test_plotted_xdata(self):
        input_data = np.array([1, 2, 3, 3.5, 4])
        _, ax = plt.subplots()
        ax = raster_single_neuron(input_data, ax=ax)
        actual = get_points(ax)[0][:, 0]
        np.testing.assert_allclose(actual, input_data)

    def test_plotted_ydata(self):
        specified_y = 125
        input_data = np.array([1, 2, 3, 3.5, 4])
        expected_x_data_1 = np.array([1, 1, 1, 1, 1])
        expected_x_data_2 = np.array(
            [specified_y, specified_y, specified_y, specified_y, specified_y]
        )

        _, ax = plt.subplots()
        ax = raster_single_neuron(input_data, ax=ax)
        actual_default_y = get_points(ax)[0][:, 1]
        np.testing.assert_allclose(
            actual_default_y, expected_x_data_1
        )  # default y_data

        _, ax = plt.subplots()
        ax = raster_single_neuron(input_data, ax=ax, y_data_ind=specified_y)
        actual_specified_y = get_points(ax)[0][:, 1]
        np.testing.assert_allclose(
            actual_specified_y, expected_x_data_2
        )  # default y_data

    def test_multiple_spiketrain_input(self):
        pass


class TestRasterMultipleNeurons:
    def test_plotted_xdata(self):
        st1 = np.array([1, 2, 3, 3.5, 4])
        st2 = np.array([1.3, 2.2, 3.1, 3.9, 4.2])
        input_data = [st1, st2]
        _, ax = plt.subplots()
        ax = raster_multiple_neurons(input_data, ax=ax)
        actual_1 = get_points(ax)[0][:, 0]
        actual_2 = get_points(ax)[1][:, 0]
        np.testing.assert_allclose(actual_1, input_data[0])
        np.testing.assert_allclose(actual_2, input_data[1])

    def test_plotted_ydata(self):
        st1 = np.array([1, 2, 3, 3.5, 4])
        st2 = np.array([1.3, 2.2, 3.1, 3.9, 4.2])
        input_data = [st1, st2]
        expected_default_1 = np.array([0, 0, 0, 0, 0])
        expected_default_2 = expected_default_1 + 1

        # default
        _, ax = plt.subplots()
        ax = raster_multiple_neurons(input_data, ax=ax)
        actual_1 = get_points(ax)[0][:, 1]
        actual_2 = get_points(ax)[1][:, 1]
        np.testing.assert_allclose(actual_1, expected_default_1)
        np.testing.assert_allclose(actual_2, expected_default_2)

    def test_single_spiketrain_input(self):
        st1 = np.array([1, 2, 3, 3.5, 4])
        st1_in_list = [st1]
        with pytest.raises(TypeError):
            _ = raster_multiple_neurons(st1)
            _ = raster_multiple_neurons(st1_in_list)
