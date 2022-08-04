# -*- coding: utf-8 -*-

import re


# adapted from
# https://stackoverflow.com/a/71252536


def trim_noqa(app, what, name, obj, options, lines):
    noqa_regex = re.compile(r'^(.*)\s#\snoqa:.*$')
    for i, line in enumerate(lines):
        if noqa_regex.match(line):
            lines[i] = noqa_regex.sub(r'\1', line)


def setup(app):
    app.connect('autodoc-process-docstring', trim_noqa)
