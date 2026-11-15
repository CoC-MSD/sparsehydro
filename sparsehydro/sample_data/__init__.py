"""Small sample datasets shipped with sparsehydro for examples and quick tests.

``tc-nb-001``
    One month (March 2022) of 5-minute rainfall [in] and RDII stormflow [cfs]
    from sewer flow meter TC-NB-001. Columns: ``datetime``, ``rain``,
    ``stormflow``. Stormflow is the meter's separated wet-weather component and
    can dip slightly below zero during dry weather.

Example::

    from sparsehydro.sample_data import load_sample
    df = load_sample("tc-nb-001")
"""

from .samples import load_sample, sample_names, sample_path

__all__ = ["load_sample", "sample_names", "sample_path"]
