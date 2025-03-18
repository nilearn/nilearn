// This is a template for a JavaScript code snippet that will be used to
// implement a carousel of maps.
// Everything in  will be substituted by the corresponding values
// from the Python code using Tempita. The substituted code will be then be
// inserted into the HTML file.

function updateMaps() {
    var uid = "{{unique_id}}";
    var current_map_idx_{{unique_id}} = 0;
    var displayed_maps_{{unique_id}} = {{displayed_maps}};
    var number_maps_{{unique_id}} = {{len(displayed_maps)}};
    window['current_map_idx_' + uid] = current_map_idx_{{unique_id}};
    window['displayed_maps_' + uid] = displayed_maps_{{unique_id}};
    window['number_maps_' + uid] = number_maps_{{unique_id}};

    document.getElementById("comp-" + uid).innerHTML = displayed_maps_{{unique_id}}[current_map_idx_{{unique_id}}];

    function displayNextMap() {
      var current_map_idx = window['current_map_idx_' + uid];
      var displayed_maps = window['displayed_maps_' + uid];
      var number_maps = window['number_maps_' + uid];

      document.getElementById("map-" + uid + "-" + current_map_idx).style["display"] = "none";
      current_map_idx = current_map_idx + 1;
      if (current_map_idx >= number_maps) {
        current_map_idx = 0;
      }
      document.getElementById("map-" + uid + "-" + current_map_idx).style["display"] = "block";
      document.getElementById("comp-" + uid).innerHTML = displayed_maps[current_map_idx];
      window['current_map_idx_' + uid] = current_map_idx;
    }

    function displayPreviousMap() {
      var current_map_idx = window['current_map_idx_' + uid];
      var displayed_maps = window['displayed_maps_' + uid];
      var number_maps = window['number_maps_' + uid];

      document.getElementById("map-" + uid + "-" + current_map_idx).style["display"] = "none";
      current_map_idx = current_map_idx - 1;
      if (current_map_idx < 0) {
        current_map_idx = number_maps - 1;
      }
      document.getElementById("map-" + uid + "-" + current_map_idx).style["display"] = "block";
      document.getElementById("comp-" + uid).innerHTML = displayed_maps[current_map_idx];
      window['current_map_idx_' + uid] = current_map_idx;
    }

    // Attach functions to buttons
    document.querySelector("#prev-btn-" + uid).onclick = displayPreviousMap;
    document.querySelector("#next-btn-" + uid).onclick = displayNextMap;
}

document.addEventListener("DOMContentLoaded", function() {

updateMaps()

  });
