from enum import Enum


class Orientation(Enum):
    LATERAL = "Lateral"
    MEDIAL = "Medial"
    DORSAL = "Dorsal"
    VENTRAL = "Ventral"
    ANTERIOR = "Anterior"
    POSTERIOR = "Posterior"


def get_orientation(value, listify: bool = True) -> list:
    if isinstance(value, str):
        value = Orientation[value.upper()]
    elif isinstance(value, (list, tuple)):
        value = [get_orientation(element, listify=False) for element in value]
    elif isinstance(value, Orientation):
        pass
    else:
        message = f'Invalid orientation configuration type: {type(value)}!'
        raise TypeError(message)
    requires_listing = listify and not isinstance(value, list)
    return [value] if requires_listing else value
