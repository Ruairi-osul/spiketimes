import pytest
from spiketimes.alignment import align_to


class TestAlignTo:
    def test_two_scaler_input(self):
        bad_input_1 = 1
        bad_input_2 = 2
        with pytest.raises(ValueError):
            align_to(to_be_aligned=bad_input_1, to_align_to=bad_input_2)

    def test_list_input(self):
        pass

    def test_events_before_first_aligning_event(self):
        pass

    def test_output_shape(self):
        pass
