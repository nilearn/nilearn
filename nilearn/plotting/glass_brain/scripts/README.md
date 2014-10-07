* `svg_to_json_converter.py`: svg to json exporter which leverages off
  https://github.com/cjlano/svg and exports the coordinates of the
  paths in the svg together with a few path attributes (id,
  stroke-color, stroke-width, etc.)
* `align_svg.py`: script which plots the svg on top of the anatomy image
  for a slice in each direction.
* `generate_json.sh`: simple bash script to regenerate all the json
  files from the svg in svg_plots
