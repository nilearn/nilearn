"""Update the machine name in the JSON file located at ~/.asv-machine.json.
The script takes a new machine name as a command line argument and updates
the JSON file accordingly.
"""

import json
import sys
from pathlib import Path

# accept the new machine name as a command line argument
if len(sys.argv) != 2:
    print("Usage: python set_machine_name.py <new_machine_name>")
    sys.exit(1)

new_machine_name = sys.argv[1]

json_file_path = Path("~/.asv-machine.json").expanduser()
loaded_json = json.load(Path.open(json_file_path))

print(f"Current machine info: {loaded_json}")

# copy the first key into a new key new_machine_name
loaded_json[new_machine_name] = loaded_json[next(iter(loaded_json.keys()))]
# also update "machine" value in the new key
loaded_json[new_machine_name]["machine"] = new_machine_name

# remove the old key
del loaded_json[next(iter(loaded_json.keys()))]

print(f"Updated machine info: {loaded_json}")

# write the modified json back to the file
with Path.open(json_file_path, "w") as json_file:
    json.dump(loaded_json, json_file, indent=4)
