"""Observation and step-result helpers for RayNet Python runners."""

from __future__ import annotations


def observation_to_list(observation):
    """Convert one native observation object to a plain Python list."""
    if hasattr(observation, "to_list"):
        return observation.to_list()
    return list(observation)


def observation_to_dict(observation):
    """Convert one native observation object to a plain Python dict if labelled."""
    if isinstance(observation, dict):
        return dict(observation)
    if hasattr(observation, "to_dict"):
        raw = observation.to_dict()
        if raw:
            return dict(raw)
    return None


def _cast_value(value, value_type):
    if value_type is None:
        return value
    return value_type(value)


def _field_value(values, field):
    index = int(field["index"])
    return values[index]


def observation_to_raw_dict(observation, fields, *, value_type=None):
    """Convert one observation into the named raw-info dict requested by a caller."""
    raw_observation = observation_to_dict(observation)
    if raw_observation is not None and not fields:
        return {
            str(key): _cast_value(value, value_type)
            for key, value in raw_observation.items()
        }

    values = None
    raw = {}
    for field in fields or []:
        if isinstance(field, str):
            name = field
            if raw_observation is not None and name in raw_observation:
                value = raw_observation[name]
            else:
                values = observation_to_list(observation) if values is None else values
                value = values[len(raw)]
            raw[name] = _cast_value(value, value_type)
            continue

        name = str(field["name"])
        if raw_observation is not None and name in raw_observation:
            value = raw_observation[name]
        else:
            values = observation_to_list(observation) if values is None else values
            value = _field_value(values, field)
        raw[name] = _cast_value(value, value_type)
        for alias in field.get("aliases") or []:
            raw[str(alias)] = _cast_value(value, value_type)
    return raw


def serialize_observations(observations, *, value_type=None, key_type=None,
                           fields=None):
    """Convert a mapping of native observations to JSON-safe objects."""
    serialized = {}
    for agent_id, observation in (observations or {}).items():
        key = key_type(agent_id) if key_type is not None else agent_id
        raw_observation = observation_to_dict(observation)
        if raw_observation is not None and not fields:
            serialized[key] = {
                str(name): _cast_value(value, value_type)
                for name, value in raw_observation.items()
            }
        elif fields:
            serialized[key] = observation_to_raw_dict(
                observation, fields, value_type=value_type)
        else:
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


def info_value(value):
    """Convert one metadata value to a JSON-safe Python scalar."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def info_dict(values, *, key_type=str):
    """Convert step metadata while preserving numeric timing values."""
    return {
        key_type(key) if key_type is not None else key: info_value(value)
        for key, value in (values or {}).items()
    }
