#!/bin/sh
for svg_fn in svg_plots/*.svg; do
    echo ----------------------------------------
    echo $svg_fn
    echo ----------------------------------------
    json_fn="generated_json/$(basename $svg_fn .svg).json"
    python svg_to_json_converter.py $svg_fn $json_fn
done

