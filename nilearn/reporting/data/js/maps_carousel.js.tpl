// This is a template for a JavaScript code snippet that will be used to
// implement a carousel of maps.
// Everything in  will be substituted by the corresponding values
// from the Python code using Tempita. The substituted code will be then be
// inserted into the HTML file.


function updateMaps() {
    const uid = "{{unique_id}}";

    window['current_map_idx_' + uid] = 0;
    window['displayed_maps_' + uid] = {{displayed_maps}};
    window['number_maps_' + uid] = window['displayed_maps_' + uid].length;

    document.getElementById("comp-" + uid).innerHTML = window['displayed_maps_' + uid][0];

    function displayNextMap() {
      let current_map_idx = window['current_map_idx_' + uid];
      let displayed_maps = window['displayed_maps_' + uid];
      let number_maps = window['number_maps_' + uid];

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
      let current_map_idx = window['current_map_idx_' + uid];
      let displayed_maps = window['displayed_maps_' + uid];
      let number_maps = window['number_maps_' + uid];

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
    document.querySelector("#next-btn-" + uid).onclick = displayNextMap ;
}

document.addEventListener("DOMContentLoaded", function() {

updateMaps()

  });
