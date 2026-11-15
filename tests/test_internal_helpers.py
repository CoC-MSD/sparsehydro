"""Shared internal helpers: column extraction, validation, units."""

import pickle

import numpy as np
import pandas as pd
import pytest

from sparsehydro import units
from sparsehydro._columns import ColumnExtractor
from sparsehydro._validation import positive_integer


def test_column_extractor_reads_frames_and_mappings():
    frame = pd.DataFrame({"q": [1, 2, 3]})
    np.testing.assert_array_equal(ColumnExtractor("q")(frame), [1, 2, 3])
    assert ColumnExtractor("q")(frame).dtype == frame.q.dtype
    assert ColumnExtractor("q", dtype=float)(frame).dtype == float
    assert ColumnExtractor("q", dtype=float)({"q": [1, 2]}).dtype == float


def test_column_extractor_pickles_including_legacy_state():
    original = ColumnExtractor("q", dtype=float)
    assert pickle.loads(pickle.dumps(original)) == original
    legacy = ColumnExtractor.__new__(ColumnExtractor)
    legacy.__setstate__("q")
    assert (legacy.key, legacy.dtype) == ("q", None)


@pytest.mark.parametrize("value,minimum,ok", [
    (1, 1, True), (np.int64(3), 1, True), (0, 0, True),
    (0, 1, False), (True, 0, False), (1.0, 1, False), ("2", 1, False),
])
def test_positive_integer(value, minimum, ok):
    if ok:
        positive_integer("n", value, minimum=minimum)
    else:
        with pytest.raises(ValueError, match="n must be an integer"):
            positive_integer("n", value, minimum=minimum)


def test_unit_constants():
    assert units.ACRE_INCH_TO_FT3 == 3630.0
    assert units.IN_AC_PER_HR_TO_CFS == pytest.approx(1.008333, rel=1e-6)
    assert units.MM_AC_PER_HR_TO_CFS == pytest.approx(units.IN_AC_PER_HR_TO_CFS / 25.4)
    assert units.MGD_TO_CFS == pytest.approx(1.547229, rel=1e-6)


def test_verification_helpers():
    from sparsehydro._validation import VerificationError, verify_close, verify_equal

    verify_close([1.0, 2.0], [1.0, 2.0 + 1e-12])
    verify_equal([1, 2], [1, 2])
    with pytest.raises(VerificationError, match="Total flow do not agree"):
        verify_close([1.0], [1.1], what="Total flow")
    with pytest.raises(VerificationError, match="Snow forcing changed"):
        verify_equal([1.0], [2.0], what="Snow forcing")
    assert issubclass(VerificationError, RuntimeError)
