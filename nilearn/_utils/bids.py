from nilearn.interfaces.bids._utils import _bids_entities


def create_bids_filename(fields, entities_to_include=None):
    """Create BIDS filename from dictionary of entity-label pairs.

    Parameters
    ----------
    fields : :obj:`dict` of :obj:`str`
        Dictionary of entity-label pairs, for example:

        {
         "suffix": "T1w",
         "extension": "nii.gz",
         "entities": {"acq":  "ap",
                      "desc": "preproc"}
        }.

    Returns
    -------
    BIDS filename : :obj:`str`

    """
    if entities_to_include is None:
        entities_to_include = _bids_entities()["raw"]

    filename = ""

    for key in entities_to_include:
        if key in fields["entities"]:
            value = fields["entities"][key]
            if value not in (None, ""):
                filename += f"{key}-{value}_"
    if "suffix" in fields:
        filename += f"{fields['suffix']}"
    if "extension" in fields:
        filename += f".{fields['extension']}"

    return filename
