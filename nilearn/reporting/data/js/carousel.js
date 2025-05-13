/**
 * A class representing a map carousel that cycles through different maps.
 */
class MapCarousel {
    /**
     * Creates a new MapCarousel instance.
     * @param {string} uid - A unique identifier for the masker report.
     * @param {number[]} displayed_maps - An array of map IDs to cycle through.
     * @param {boolean} [is_sphere=false] - Determines whether the carousel operates on sphere masker.
     */
    constructor(uid, displayed_maps, is_sphere = false) {
        /** @private {string} */
        this.uid = uid;

        /** @private {number[]} */
        this.displayed_maps = displayed_maps;

        /** @private {number} */
        this.current_map_idx = 0;

        /** @private {number} */
        this.number_maps = displayed_maps.length;

        /** @private {boolean} */
        this.is_sphere = is_sphere;

        this.init();
    }

    /**
     * Initializes the carousel by setting up event listeners and displaying the first map.
     */
    init() {
        this.showMap(0);

        let prevButton = document.querySelector(`#prev-btn-${this.uid}`);
        let nextButton = document.querySelector(`#next-btn-${this.uid}`);

        if (prevButton) prevButton.addEventListener("click", () => this.displayPreviousMap());
        if (nextButton) nextButton.addEventListener("click", () => this.displayNextMap());
    }

    /**
     * Displays the map at the given index and hides all others.
     * @param {number} index - The index of the map to display.
     *
     * for sphere masker report this adapts the full title in the carousel
     */
    showMap(index) {
        this.displayed_maps.forEach((_, i) => {
            let mapElement = document.getElementById(`map-${this.uid}-${i}`);
            if (mapElement) {
                mapElement.style.display = i === index ? "block" : "none";
            }
        });

        let compElement = document.getElementById(`comp-${this.uid}`);
        if (compElement) {
            compElement.innerHTML = this.displayed_maps[index];
            if (this.is_sphere){
                if (index === 0){
                    compElement.innerHTML = "All Spheres"
                } else{
                    compElement.innerHTML = "Sphere " + this.displayed_maps[index];
                }
            }

        }
    }

    /**
     * Advances the carousel to the next map.
     *
     * using % modulo to ensure we 'wrap' back to start in the carousel
    */
    displayNextMap() {
        this.current_map_idx = (this.current_map_idx + 1) % this.number_maps;
        this.showMap(this.current_map_idx);
    }

    /**
     * Moves the carousel to the previous map.
     *
     * using % modulo to ensure we 'wrap' back to start in the carousel
    */
    displayPreviousMap() {
        this.current_map_idx = (this.current_map_idx - 1 + this.number_maps) % this.number_maps;
        this.showMap(this.current_map_idx);
    }
}
