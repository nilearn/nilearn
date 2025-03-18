class MapCarousel {
    constructor(uid, displayed_maps, is_sphere = false) {
        this.uid = uid;
        this.displayed_maps = displayed_maps;
        this.current_map_idx = 0;
        this.is_sphere = is_sphere
        this.number_maps = displayed_maps.length;

        this.init();
    }

    init() {
        this.showMap(0);

        let prevButton = document.querySelector(`#prev-btn-${this.uid}`);
        let nextButton = document.querySelector(`#next-btn-${this.uid}`);

        if (prevButton) prevButton.addEventListener("click", () => this.displayPreviousMap());
        if (nextButton) nextButton.addEventListener("click", () => this.displayNextMap());
    }

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

    // for the following methods
    // using % modulo to ensure we 'wrap' back to start in the carousel
    displayNextMap() {
        this.current_map_idx = (this.current_map_idx + 1) % this.number_maps;
        this.showMap(this.current_map_idx);
    }

    displayPreviousMap() {
        this.current_map_idx = (this.current_map_idx - 1 + this.number_maps) % this.number_maps;
        this.showMap(this.current_map_idx);
    }
}
