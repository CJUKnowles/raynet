"""Observation and step-result helpers for RayNet Python runners."""

from __future__ import annotations


def observation_to_list(observation):
    """Convert one native observation object to a plain Python list."""
    if hasattr(observation, "to_list"):
        return observation.to_list()
    return list(observation)


def serialize_observations(observations, *, value_type=None, key_type=None):
    """Convert a mapping of native observations to serializable lists."""
    serialized = {}
    for agent_id, observation in (observations or {}).items():
        key = key_type(agent_id) if key_type is not None else agent_id
        values = observation_to_list(observation)
        if value_type is not None:
            values = [value_type(value) for value in values]
        serialized[key] = values
    return serialized


def typed_dict(values, value_type, *, key_type=str):
    """Convert mapping keys and values to JSON/process-safe Python types."""
    return {
        key_type(key) if key_type is not None else key: value_type(value)
        for key, value in (values or {}).items()
    }


def float_dict(values, *, key_type=str):
    """Convert mapping values to floats."""
    return typed_dict(values, float, key_type=key_type)


def bool_dict(values, *, key_type=str):
    """Convert mapping values to bools."""
    return typed_dict(values, bool, key_type=key_type)
