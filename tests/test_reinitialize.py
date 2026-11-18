"""Initializing a model twice must not duplicate its registrations."""

import pytest

import sparsehydro.models  # noqa: F401  (registers the built-in models)
from sparsehydro.models.ensemble import EnsembleModel
from sparsehydro.models.rdii import RDIIModel
from sparsehydro.registry import registry


def _snapshot(model):
    return (
        [c.name for c in model._constraint_registry],
        list(model._output_fields),
        list(model.scalar_parameter_names),
    )


def _default_constructible():
    names = []
    for name in registry.names():
        try:
            registry.create(name)
        except TypeError:
            continue  # needs constructor arguments
        names.append(name)
    return names


@pytest.mark.parametrize("name", _default_constructible())
def test_registered_model_initializes_idempotently(name):
    model = registry.create(name)
    model.initialize()
    first = _snapshot(model)
    model.initialize()
    assert _snapshot(model) == first


def test_ensemble_initializes_idempotently():
    model = EnsembleModel([(RDIIModel(), "rdii_cfs"), (RDIIModel(), "rdii_cfs")])
    model.initialize()
    first = _snapshot(model)
    model.initialize()
    assert _snapshot(model) == first
