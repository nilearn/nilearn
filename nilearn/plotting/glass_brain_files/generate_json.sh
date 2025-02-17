#!/bin/sh

# Quick and dirty script to regenerate the brain schematics JSONs from
# the SVGs. This script has to be launched from the folder it lives
# in, i.e. as ./generate_json.sh

for svg_fn in *.svg; do
    echo ----------------------------------------
    echo $svg_fn
    echo ----------------------------------------
    json_fn="$(basename $svg_fn .svg).json"
    python svg_to_json_converter.py $svg_fn $json_fn
done
