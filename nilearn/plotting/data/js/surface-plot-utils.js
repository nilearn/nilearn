function decodeBase64(encoded, dtype) {

    let getter = {
        "float32": "getFloat32",
        "int32": "getInt32"
    }[dtype];

    let arrayType = {
        "float32": Float32Array,
        "int32": Int32Array
    }[dtype];

    let raw = atob(encoded);
    let buffer = new ArrayBuffer(raw.length);
    let asIntArray = new Uint8Array(buffer);
    for (let i = 0; i !== raw.length; i++) {
        asIntArray[i] = raw.charCodeAt(i);
    }

    let view = new DataView(buffer);
    let decoded = new arrayType(
        raw.length / arrayType.BYTES_PER_ELEMENT);
    for (let i = 0, off = 0; i !== decoded.length;
        i++, off += arrayType.BYTES_PER_ELEMENT) {
        decoded[i] = view[getter](off, true);
    }
    return decoded;
}

function getAxisConfig() {
    let axisConfig = {
        showgrid: false,
        showline: false,
        ticks: '',
        title: '',
        showticklabels: false,
            zeroline: false,
        showspikes: false,
        spikesides: false
    };

    return axisConfig;
}

function getLighting() {
    return {};
    // i.e. use plotly defaults:
    // {
    //     "ambient": 0.8,
    //     "diffuse": .8,
    //     "fresnel": .2,
    //     "specular": .05,
    //     "roughness": .5,
    //     "facenormalsepsilon": 1e-6,
    //     "vertexnormalsepsilon": 1e-12
    // };

}

function getConfig() {
    let config = {
        modeBarButtonsToRemove: ["hoverClosest3d"],
        displayLogo: false
    };

    return config;
}

function getCamera(plotDivId, viewSelectId) {
    let view = $("#" + viewSelectId).val();
    if (view === "custom") {
        try {
            return $("#" + plotDivId)[0].layout.scene.camera;
        } catch (e) {
            return {};
        }
    }
    let cameras = {
        "left": {eye: {x: -1.7, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
        "right": {eye: {x: 1.7, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
        "top": {eye: {x: 0, y: 0, z: 1.7},
                up: {x: 0, y: 1, z: 0},
                center: {x: 0, y: 0, z: 0}},
        "bottom": {eye: {x: 0, y: 0, z: -1.7},
                    up: {x: 0, y: 1, z: 0},
                    center: {x: 0, y: 0, z: 0}},
        "front": {eye: {x: 0, y: 1.7, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
        "back": {eye: {x: 0, y: -1.7, z: 0},
                    up: {x: 0, y: 0, z: 1},
                    center: {x: 0, y: 0, z: 0}},
    };

    return cameras[view];

}

function getLayout(plotDivId, viewSelectId, blackBg) {

    let camera = getCamera(plotDivId, viewSelectId);
    let axisConfig = getAxisConfig();

    let height = Math.min($(window).outerHeight() * .9,
                            $(window).width() * 2 / 3);
    let width = height * 3 / 2;

    let layout = {
        showlegend: false,
        height: height, width: width,
        margin: {l:0, r:0, b:0, t:0, pad:0},
        hovermode: false,
        paper_bgcolor: blackBg ? '#000': '#fff',
        axis_bgcolor: '#333',
        scene: {
            camera: camera,
            xaxis: axisConfig,
            yaxis: axisConfig,
            zaxis: axisConfig
        }
    };

    return layout;

}

function updateLayout(plotDivId, viewSelectId, blackBg) {
    let layout = getLayout(
        plotDivId, viewSelectId, blackBg);
    Plotly.relayout(plotDivId, layout);
}

function textColor(black_bg){
    if (black_bg){
        return "white";
    }
    return "black";
}

function addColorbar(colorscale, cmin, cmax, divId, layout, config,
                     fontsize=25, height=.5, color="black") {
    // hack to get a colorbar
    let dummy = {
        "opacity": 0,
        "colorbar": {"tickfont": {"size": fontsize, "color": color},
                     "len": height},
        "type": "mesh3d",
        "colorscale": colorscale,
        "x": [1, 0, 0],
        "y": [0, 1, 0],
        "z": [0, 0, 1],
        "i": [0],
        "j": [1],
        "k": [2],
        "intensity": [0.],
        "cmin": cmin,
        "cmax": cmax,
    };

    Plotly.plot(divId, [dummy], layout, config);

}


function decodeHemisphere(surfaceInfo, surface, hemisphere){

    let info = surfaceInfo[surface + "_" + hemisphere];

    for (let attribute of ["x", "y", "z"]) {
        if (!(attribute in info)) {
            info[attribute] = decodeBase64(
                info["_" + attribute], "float32");
        }
    }

    for (let attribute of ["i", "j", "k"]) {
        if (!(attribute in info)) {
            info[attribute] = decodeBase64(
                info["_" + attribute], "int32");
        }
    }

}
