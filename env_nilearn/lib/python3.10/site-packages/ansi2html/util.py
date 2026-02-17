from typing import BinaryIO, List


def read_to_unicode(obj: BinaryIO) -> List[str]:
    return [line.decode("utf-8") for line in obj.readlines()]
