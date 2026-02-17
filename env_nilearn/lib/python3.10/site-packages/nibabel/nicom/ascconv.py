# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parse the "ASCCONV" meta data format found in a variety of Siemens MR files.
"""

import ast
import re
from collections import OrderedDict

ASCCONV_RE = re.compile(
    r'### ASCCONV BEGIN((?:\s*[^=\s]+=[^=\s]+)*) ###\n(.*?)\n### ASCCONV END ###',
    flags=re.MULTILINE | re.DOTALL,
)


class AscconvParseError(Exception):
    """Error parsing ascconv file"""


class Atom:
    """Object to hold operation, object type and object identifier

    An atom represents an element in an expression.  For example::

        a.b[0].c

    has four elements.  We call these elements "atoms".

    We represent objects (like ``a``) as dicts for convenience.

    The last element (``.c``) is an ``op = ast.Attribute`` operation where the
    object type (`obj_type`) of ``c`` is not constrained (we can't tell from
    the operation what type it is).  The `obj_id` is the name of the object --
    "c".

    The second to last element ``[0]``, is ``op = ast.Subscript``, with object type
    dict (we know from the subsequent operation ``.c`` that this must be an
    object, we represent the object by a dict).  The `obj_id` is the index 0.

    Parameters
    ----------
    op : {'name', 'attr', 'list'}
        Assignment type.  Assignment to name (root namespace), attribute or
        list element.
    obj_type : {list, dict, other}
        Object type being assigned to.
    obj_id : str or int
        Key (``obj_type is dict``) or index (``obj_type is list``)
    """

    def __init__(self, op, obj_type, obj_id):
        self.op = op
        self.obj_type = obj_type
        self.obj_id = obj_id


class NoValue:
    """Signals no value present"""


def assign2atoms(assign_ast, default_class=int):
    """Parse single assignment ast from ascconv line into atoms

    Parameters
    ----------
    assign_ast : assignment statement ast
        ast derived from single line of ascconv file.
    default_class : class, optional
        Class that will create an object where we cannot yet know the object
        type in the assignment.

    Returns
    -------
    atoms : list
        List of :class:`atoms`.  See docstring for :class:`atoms`.  Defines
        left to right sequence of assignment in `line_ast`.
    """
    if not len(assign_ast.targets) == 1:
        raise AscconvParseError('Too many targets in assign')
    target = assign_ast.targets[0]
    atoms = []
    prev_target_type = default_class  # Placeholder for any scalar value
    while True:
        if isinstance(target, ast.Name):
            atoms.append(Atom(target, prev_target_type, target.id))
            break
        if isinstance(target, ast.Attribute):
            atoms.append(Atom(target, prev_target_type, target.attr))
            target = target.value
            prev_target_type = OrderedDict
        elif isinstance(target, ast.Subscript):
            index = target.slice.value
            atoms.append(Atom(target, prev_target_type, index))
            target = target.value
            prev_target_type = list
        else:
            raise AscconvParseError(f'Unexpected LHS element {target}')
    return reversed(atoms)


def _create_obj_in(atom, root):
    """Find / create object defined in `atom` in dict-like given by `root`

    Returns corresponding value if there is already a key matching
    `atom.obj_id` in `root`.

    Otherwise, create new object with ``atom.obj_type`, insert into dictionary,
    and return new object.

    Can therefore modify `root` in place.
    """
    name = atom.obj_id
    obj = root.get(name, NoValue)
    if obj is not NoValue:
        return obj
    obj = atom.obj_type()
    root[name] = obj
    return obj


def _create_subscript_in(atom, root):
    """Find / create and insert object defined by `atom` from list `root`

    The `atom` has an index, defined in ``atom.obj_id``.  If `root` is long
    enough to contain this index, return the object at that index.  Otherwise,
    extend `root` with None elements to contain index ``atom.obj_id``, then
    create a new object via ``atom.obj_type()``, insert at the end of the list,
    and return this object.

    Can therefore modify `root` in place.
    """
    curr_n = len(root)
    index = atom.obj_id
    if curr_n > index:
        return root[index]
    obj = atom.obj_type()
    root += [None] * (index - curr_n) + [obj]
    return obj


def obj_from_atoms(atoms, namespace):
    """Return object defined by list `atoms` in dict-like `namespace`

    Parameters
    ----------
    atoms : list
        List of :class:`atoms`
    namespace : dict-like
        Namespace in which object will be defined.

    Returns
    -------
    obj_root : object
        Namespace such that we can set a desired value to the object defined in
        `atoms` with ``obj_root[obj_key] = value``.
    obj_key : str or int
        Index into list or key into dictionary for `obj_root`.
    """
    root_obj = namespace
    for el in atoms:
        prev_root = root_obj
        if isinstance(el.op, (ast.Attribute, ast.Name)):
            root_obj = _create_obj_in(el, root_obj)
        else:
            root_obj = _create_subscript_in(el, root_obj)
        if not isinstance(root_obj, el.obj_type):
            raise AscconvParseError(f'Unexpected type for {el.obj_id} in {prev_root}')
    return prev_root, el.obj_id


def _get_value(assign):
    value = assign.value
    if isinstance(value, ast.Constant):
        return value.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        return -value.operand.value
    raise AscconvParseError(f'Unexpected RHS of assignment: {value}')


def parse_ascconv(ascconv_str, str_delim='"'):
    """Parse the 'ASCCONV' format from `input_str`.

    Parameters
    ----------
    ascconv_str : str
        The string we are parsing
    str_delim : str, optional
        String delimiter.  Typically '"' or '""'

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.
    attrs : OrderedDict
        Any attributes stored in the 'ASCCONV BEGIN' line

    Raises
    ------
    AsconvParseError
        A line of the ASCCONV section could not be parsed.
    """
    attrs, content = ASCCONV_RE.match(ascconv_str).groups()
    attrs = OrderedDict(tuple(x.split('=')) for x in attrs.split())
    # Normalize string start / end markers to something Python understands
    content = content.replace(str_delim, '"""').replace('\\', '\\\\')
    # Use Python's own parser to parse modified ASCCONV assignments
    tree = ast.parse(content)

    prot_dict = OrderedDict()
    for assign in tree.body:
        atoms = assign2atoms(assign)
        obj_to_index, key = obj_from_atoms(atoms, prot_dict)
        obj_to_index[key] = _get_value(assign)

    return prot_dict, attrs
