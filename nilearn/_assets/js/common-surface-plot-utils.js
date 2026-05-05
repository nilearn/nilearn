function decodeBase64 (encoded, dtype) {
  const getter = {
    float32: 'getFloat32',
    int32: 'getInt32'
  }[dtype]

  const arrayType = {
    float32: Float32Array,
    int32: Int32Array
  }[dtype]

  const raw = atob(encoded)
  const buffer = new ArrayBuffer(raw.length)
  const asIntArray = new Uint8Array(buffer)
  for (let i = 0; i !== raw.length; i++) {
    asIntArray[i] = raw.charCodeAt(i)
  }

  const view = new DataView(buffer)
  const decoded = new arrayType(
    raw.length / arrayType.BYTES_PER_ELEMENT)
  for (let i = 0, off = 0; i !== decoded.length;
    i++, off += arrayType.BYTES_PER_ELEMENT) {
    decoded[i] = view[getter](off, true)
  }
  return decoded
}

function getAxisConfig () {
  const axisConfig = {
    showgrid: false,
    showline: false,
    ticks: '',
    title: '',
    showticklabels: false,
    zeroline: false,
    showspikes: false,
    spikesides: false
  }

  return axisConfig
}

function getLighting () {
  return {}
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

function getConfig () {
  const config = {
    modeBarButtonsToRemove: ['hoverClosest3d'],
    displayLogo: false
  }

  return config
}

function getCamera (plotDivId, viewSelectId) {
  const view = $('#' + viewSelectId).val()
  if (view === 'custom') {
    try {
      return $('#' + plotDivId)[0].layout.scene.camera
    } catch (e) {
      return {}
    }
  }
  const cameras = {
    left: {
      eye: { x: -1.7, y: 0, z: 0 },
      up: { x: 0, y: 0, z: 1 },
      center: { x: 0, y: 0, z: 0 }
    },
    right: {
      eye: { x: 1.7, y: 0, z: 0 },
      up: { x: 0, y: 0, z: 1 },
      center: { x: 0, y: 0, z: 0 }
    },
    top: {
      eye: { x: 0, y: 0, z: 1.7 },
      up: { x: 0, y: 1, z: 0 },
      center: { x: 0, y: 0, z: 0 }
    },
    bottom: {
      eye: { x: 0, y: 0, z: -1.7 },
      up: { x: 0, y: 1, z: 0 },
      center: { x: 0, y: 0, z: 0 }
    },
    front: {
      eye: { x: 0, y: 1.7, z: 0 },
      up: { x: 0, y: 0, z: 1 },
      center: { x: 0, y: 0, z: 0 }
    },
    back: {
      eye: { x: 0, y: -1.7, z: 0 },
      up: { x: 0, y: 0, z: 1 },
      center: { x: 0, y: 0, z: 0 }
    }
  }

  return cameras[view]
}

function getLayout (plotDivId, viewSelectId, blackBg) {
  const camera = getCamera(plotDivId, viewSelectId)
  const axisConfig = getAxisConfig()

  const height = Math.min($(window).outerHeight() * 0.9,
    $(window).width() * 2 / 3)
  const width = height * 3 / 2

  const layout = {
    showlegend: false,
    height,
    width,
    margin: { l: 0, r: 0, b: 0, t: 0, pad: 0 },
    hovermode: false,
    paper_bgcolor: blackBg ? '#000' : '#fff',
    axis_bgcolor: '#333',
    scene: {
      camera,
      xaxis: axisConfig,
      yaxis: axisConfig,
      zaxis: axisConfig
    }
  }

  return layout
}

function updateLayout (plotDivId, viewSelectId, blackBg) {
  const layout = getLayout(
    plotDivId, viewSelectId, blackBg)
  Plotly.relayout(plotDivId, layout)
}

function textColor (black_bg) {
  if (black_bg) {
    return 'white'
  }
  return 'black'
}

function addColorbar (colorscale, cmin, cmax, divId, layout, config,
  fontsize = 25, height = 0.5, color = 'black') {
  // hack to get a colorbar
  const dummy = {
    opacity: 0,
    colorbar: {
      tickfont: { size: fontsize, color },
      len: height
    },
    type: 'mesh3d',
    colorscale,
    x: [1, 0, 0],
    y: [0, 1, 0],
    z: [0, 0, 1],
    i: [0],
    j: [1],
    k: [2],
    intensity: [0.0],
    cmin,
    cmax
  }

  Plotly.plot(divId, [dummy], layout, config)
}

function decodeHemisphere (surfaceInfo, surface, hemisphere) {
  const info = surfaceInfo[surface + '_' + hemisphere]

  for (const attribute of ['x', 'y', 'z']) {
    if (!(attribute in info)) {
      info[attribute] = decodeBase64(
        info['_' + attribute], 'float32')
    }
  }

  for (const attribute of ['i', 'j', 'k']) {
    if (!(attribute in info)) {
      info[attribute] = decodeBase64(
        info['_' + attribute], 'int32')
    }
  }
}
