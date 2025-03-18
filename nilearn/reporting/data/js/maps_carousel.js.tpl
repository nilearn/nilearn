// This is a template for a JavaScript code snippet that will be used to
// implement a carousel of maps.
// Everything in  will be substituted by the corresponding values
// from the Python code using Tempita. The substituted code will be then be
// inserted into the HTML file.

function updateMaps(uid, displayed_maps) {
    let current_map_idx = 0;
    let number_maps = displayed_maps.length;

    window['current_map_idx_' + uid] = current_map_idx;
    window['displayed_maps_' + uid] = displayed_maps;
    window['number_maps_' + uid] = number_maps;

    function showMap(index) {
        displayed_maps.forEach((_, i) => {
            let mapElement = document.getElementById(`map-${uid}-${i}`);
            if (mapElement) {
                mapElement.style.display = i === index ? "block" : "none";
            }
        });

        let compElement = document.getElementById(`comp-${uid}`);
        if (compElement) {
            compElement.innerHTML = displayed_maps[index];
        }
    }

    function displayNextMap() {
        current_map_idx = (current_map_idx + 1) % number_maps;
        window['current_map_idx_' + uid] = current_map_idx;
        showMap(current_map_idx);
    }

    function displayPreviousMap() {
        current_map_idx = (current_map_idx - 1 + number_maps) % number_maps;
        window['current_map_idx_' + uid] = current_map_idx;
        showMap(current_map_idx);
    }

    // Initialize maps
    showMap(0);

    // Attach event listeners safely
    let prevButton = document.querySelector(`#prev-btn-${uid}`);
    let nextButton = document.querySelector(`#next-btn-${uid}`);

    if (prevButton) prevButton.addEventListener("click", displayPreviousMap);
    if (nextButton) nextButton.addEventListener("click", displayNextMap);
}

document.addEventListener("DOMContentLoaded", function () {
    updateMaps("{{unique_id}}", {{displayed_maps}});
});
