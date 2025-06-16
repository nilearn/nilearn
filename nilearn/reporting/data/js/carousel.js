/**
 * A class representing a carousel that cycles through different objects.
 */
class Carousel {
    /**
     * Creates a new Carousel instance.
     * @param {string} uid - A unique identifier for the masker report.
     * @param {number[]} displayed_objects - An array of IDs to cycle through.
     * @param {boolean} [is_sphere=false] - Determines whether the carousel operates on sphere masker.
     */
    constructor(uid, displayed_objects, is_sphere = false) {
        /** @private {string} */
        this.uid = uid;

        /** @private {number[]} */
        this.displayed_objects = displayed_objects;

        /** @private {number} */
        this.current_obj_idx = 0;

        /** @private {number} */
        this.number_objs = displayed_objects.length;

        /** @private {boolean} */
        this.is_sphere = is_sphere;

        this.init();
    }

    /**
     * Initializes the carousel by setting up event listeners and displaying the first object.
     */
    init() {
        this.showObj(0);

        let prevButton = document.querySelector(`#prev-btn-${this.uid}`);
        let nextButton = document.querySelector(`#next-btn-${this.uid}`);

        if (prevButton) prevButton.addEventListener("click", () => this.displayPrevious());
        if (nextButton) nextButton.addEventListener("click", () => this.displayNext());

        this.bindKeyboardEvents();
    }

    /**
     * Displays the object at the given index and hides all others.
     * @param {number} index - The index of the map to display.
     *
     * for sphere masker report this adapts the full title in the carousel
     */
    showObj(index) {
        this.displayed_objects.forEach((_, i) => {
            let mapElement = document.getElementById(`carousel-obj-${this.uid}-${i}`);
            if (mapElement) {
                mapElement.style.display = i === index ? "block" : "none";
            }
        });

        let compElement = document.getElementById(`comp-${this.uid}`);
        if (compElement) {
            compElement.innerHTML = this.displayed_objects[index];
            if (this.is_sphere){
                if (index === 0){
                    compElement.innerHTML = "All Spheres"
                } else{
                    compElement.innerHTML = "Sphere " + this.displayed_objects[index];
                }
            }

        }

        // For GLM only
        // Update the section the navbar links to.
        let matrixNavBar = document.querySelector(`#navbar-matrix-link-${this.uid}`);
        if (matrixNavBar) {
            matrixNavBar.href = `#design-matrix-${this.uid}-${index}`;
        }
        let contrastNavBar = document.querySelector(`#navbar-contrasts-link-${this.uid}`);
        if (contrastNavBar) {
            contrastNavBar.href = `#contrasts-${this.uid}-${index}`;
        }
    }

    /**
     * Advances the carousel to the next object.
     *
     * using % modulo to ensure we 'wrap' back to start in the carousel
    */
    displayNext() {
        this.current_obj_idx = (this.current_obj_idx + 1) % this.number_objs;
        this.showObj(this.current_obj_idx);
    }

    /**
     * Moves the carousel to the previous object.
     *
     * using % modulo to ensure we 'wrap' back to start in the carousel
    */
    displayPrevious() {
        this.current_obj_idx = (this.current_obj_idx - 1 + this.number_objs) % this.number_objs;
        this.showObj(this.current_obj_idx);
    }

    /**
     * Binds carousel to right and left arrow keys to cycle through carousel.
    */
    bindKeyboardEvents() {
        document.addEventListener("keydown", (event) => {
            if (event.key === "ArrowRight") {
                this.displayNext();
            } else if (event.key === "ArrowLeft") {
                this.displayPrevious();
            }
        });
    }
}
