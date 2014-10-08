#!/bin/sh
for svg_fn in *.svg; do
    echo ----------------------------------------
    echo $svg_fn
    echo ----------------------------------------
    json_fn="$(basename $svg_fn .svg).json"
    python svg_to_json_converter.py $svg_fn $json_fn
done

