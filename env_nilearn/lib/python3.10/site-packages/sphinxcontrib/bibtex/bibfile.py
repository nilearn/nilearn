"""
Classes and methods to work with bib files.

.. autoclass:: BibFile
    :members:

.. autoclass:: BibData
    :members:

.. autofunction:: parse_bibdata

.. autofunction:: is_bibdata_outdated

.. autofunction:: process_bibdata
"""

import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Set

from docutils.nodes import make_id
from pybtex.database import BibliographyData, BibliographyDataError
from pybtex.database.input.bibtex import Parser
from sphinx.util.logging import getLogger

logger = getLogger(__name__)


class BibFile(NamedTuple):
    """Contains information about a parsed bib file."""

    mtime: float  #: Modification time of file when last parsed.
    keys: Dict[str, None]  #: Set of keys for this bib file as ordered dict.


class BibData(NamedTuple):
    """Contains information about a collection of bib files."""

    encoding: str  #: Encoding of all bib files.
    bibfiles: Dict[Path, BibFile]  #: Maps bib filename to information about it.
    data: BibliographyData  #: Data parsed from all bib files.


def get_mtime(bibfilename: Path) -> float:
    try:
        return bibfilename.lstat().st_mtime
    except OSError:
        return -math.inf


def parse_bibdata(bibfilenames: List[Path], encoding: str) -> BibData:
    """Parse *bibfilenames* with given *encoding*, and return parsed data."""
    parser = Parser(encoding)
    bibfiles: Dict[Path, BibFile] = {}
    keys: Dict[str, None] = {}
    for filename in bibfilenames:
        logger.info("parsing bibtex file {0}... ".format(filename), nonl=True)
        if not filename.is_file():
            logger.warning(
                "could not open bibtex file {0}.".format(filename),
                type="bibtex",
                subtype="bibfile_error",
            )
            new_keys: Dict[str, None] = {}
        else:
            try:
                parser.parse_file(filename)
            except BibliographyDataError as exc:
                logger.warning(
                    "bibliography data error in {0}: {1}".format(filename, exc),
                    type="bibtex",
                    subtype="bibfile_data_error",
                )
            keys, old_keys = dict.fromkeys(parser.data.entries.keys()), keys
            assert all(key in keys for key in old_keys)
            new_keys = dict.fromkeys(key for key in keys if key not in old_keys)
            logger.info("parsed {0} entries".format(len(new_keys)))
        bibfiles[filename] = BibFile(mtime=get_mtime(filename), keys=new_keys)
    return BibData(encoding=encoding, bibfiles=bibfiles, data=parser.data)


def is_bibdata_outdated(
    bibdata: BibData, bibfilenames: List[Path], encoding: str
) -> bool:
    return (
        bibdata.encoding != encoding
        or list(bibdata.bibfiles) != bibfilenames
        or any(
            bibfile.mtime != get_mtime(filename)
            for filename, bibfile in bibdata.bibfiles.items()
        )
    )


def process_bibdata(
    bibdata: BibData, bibfilenames: List[Path], encoding: str
) -> BibData:
    """Parse *bibfilenames* and store parsed data in *bibdata*."""
    logger.info("checking bibtex cache... ", nonl=True)
    if is_bibdata_outdated(bibdata, bibfilenames, encoding):
        logger.info("out of date")
        return parse_bibdata(bibfilenames, encoding)
    else:
        logger.info("up to date")
        return bibdata


# function does not really fit in any module, but used by both
# cite and footcite domains, so for now it's residing here
def _make_ids(docname: str, lineno: int, ids: Set[str], raw_id: str) -> List[str]:
    if raw_id:
        id_ = make_id(raw_id)
        if id_ in ids:
            logger.warning(
                f"duplicate citation id {id_}",
                location=(docname, lineno),
                type="bibtex",
                subtype="duplicate_id",
            )
            return []
        else:
            ids.add(id_)
            return [id_]
    else:
        return []
