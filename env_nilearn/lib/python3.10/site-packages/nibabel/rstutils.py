"""ReStructured Text utilities

* Make ReST table given array of values
"""

import numpy as np


def rst_table(
    cell_values,
    row_names=None,
    col_names=None,
    title='',
    val_fmt='{0:5.2f}',
    format_chars=None,
):
    """Return string for ReST table with entries `cell_values`

    Parameters
    ----------
    cell_values : (R, C) array-like
        At least 2D.  Can be greater than 2D, in which case you should adapt
        the `val_fmt` to deal with the multiple entries that will go in each
        cell
    row_names : None or (R,) length sequence, optional
        Row names.  If None, use ``row[0]`` etc.
    col_names : None or (C,) length sequence, optional
        Column names.  If None, use ``col[0]`` etc.
    title : str, optional
        Title for table.  Add as heading above table
    val_fmt : str, optional
        Format string using string ``format`` method mini-language. Converts
        the result of ``cell_values[r, c]`` to a string to make the cell
        contents. Default assumes a floating point value in a 2D `cell_values`.
    format_chars : None or dict, optional
        With keys 'down', 'along', 'thick_long', 'cross' and 'title_heading'.
        Values are characters for: lines going down; lines going along; thick
        lines along; two lines crossing; and the title overline / underline.
        All missing values filled with rst defaults.

    Returns
    -------
    table_str : str
        Multiline string with ascii table, suitable for printing
    """
    # formatting
    if format_chars is None:
        format_chars = {}
    down = format_chars.pop('down', '|')
    along = format_chars.pop('along', '-')
    thick_long = format_chars.pop('thick_long', '=')
    cross = format_chars.pop('cross', '+')
    title_heading = format_chars.pop('title_heading', '*')
    if len(format_chars) != 0:
        raise ValueError(f"Unexpected ``format_char`` keys {', '.join(format_chars)}")
    down_joiner = ' ' + down + ' '
    down_starter = down + ' '
    down_ender = ' ' + down
    cross_joiner = along + cross + along
    cross_starter = cross + along
    cross_ender = along + cross
    cross_thick_joiner = thick_long + cross + thick_long
    cross_thick_starter = cross + thick_long
    cross_thick_ender = thick_long + cross
    # lengths of row names, column names and values
    cell_values = np.asarray(cell_values)
    R, C = cell_values.shape[:2]
    if row_names is None:
        row_names = [f'row[{r}]' for r in range(R)]
    elif len(row_names) != R:
        raise ValueError('len(row_names) != number of rows')
    if col_names is None:
        col_names = [f'col[{c}]' for c in range(C)]
    elif len(col_names) != C:
        raise ValueError('len(col_names) != number of columns')
    row_len = max(len(name) for name in row_names)
    col_len = max(len(name) for name in col_names)
    # Compile row value strings, find longest, extend col length to match
    row_str_list = []
    for row_no in range(R):
        row_strs = [val_fmt.format(val) for val in cell_values[row_no]]
        max_len = max(len(s) for s in row_strs)
        if max_len > col_len:
            col_len = max_len
        row_str_list.append(row_strs)
    row_name_fmt = '{0:<' + str(row_len) + '}'
    row_names = [row_name_fmt.format(name) for name in row_names]
    col_name_fmt = '{0:^' + str(col_len) + '}'
    col_names = [col_name_fmt.format(name) for name in col_names]
    col_headings = [' ' * row_len] + col_names
    col_header = down_joiner.join(col_headings)
    row_val_fmt = '{0:<' + str(col_len) + '}'
    table_strs = []
    if title != '':
        table_strs += [
            title_heading * len(title),
            title,
            title_heading * len(title),
            '',
        ]
    along_headings = [along * len(h) for h in col_headings]
    crossed_line = cross_starter + cross_joiner.join(along_headings) + cross_ender
    thick_long_headings = [thick_long * len(h) for h in col_headings]
    crossed_thick_line = (
        cross_thick_starter + cross_thick_joiner.join(thick_long_headings) + cross_thick_ender
    )
    table_strs += [
        crossed_line,
        down_starter + col_header + down_ender,
        crossed_thick_line,
    ]
    for row_no, row_name in enumerate(row_names):
        row_vals = [row_val_fmt.format(row_str) for row_str in row_str_list[row_no]]
        row_line = down_starter + down_joiner.join([row_name] + row_vals) + down_ender
        table_strs.append(row_line)
    table_strs.append(crossed_line)
    return '\n'.join(table_strs)
