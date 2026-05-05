const data = []

function getOpacity () {
  const opacity = $('#opacity-range').val()
  return opacity === 100 ? 1 : opacity / 300
}

function makePlot (surface, hemisphere, divId) {
  decodeHemisphere(connectomeInfo, surface, hemisphere)
  const info = connectomeInfo[surface + '_' + hemisphere]
  info.type = 'mesh3d'
  info.color = '#aaaaaa'
  info.opacity = getOpacity()
  info.lighting = getLighting()
  data.push(info)

  const layout = getLayout('connectome-plot', 'select-view', false)

  layout.title = {
    text: connectomeInfo.connectome.title,
    font: {
      size: connectomeInfo.connectome.title_fontsize,
      color: textColor(connectomeInfo.black_bg)
    },
    yref: 'paper',
    y: 0.9
  }

  const config = getConfig()

  Plotly.plot(divId, data, layout, config)
}

function addPlot () {
  for (const hemisphere of ['left', 'right']) {
    makePlot('pial', hemisphere, 'connectome-plot')
  }
  if (connectomeInfo.connectome.markers_only) {
    return
  }
  if (connectomeInfo.connectome.colorbar) {
    addColorbar(
      connectomeInfo.connectome.line_colorscale,
      connectomeInfo.connectome.line_cmin,
      connectomeInfo.connectome.line_cmax,
      'connectome-plot',
      getLayout('connectome-plot', 'select-view', false),
      getConfig(),
      connectomeInfo.connectome.cbar_fontsize,
      connectomeInfo.connectome.cbar_height,
      textColor(connectomeInfo.black_bg)
    )
  }
}

function updateOpacity () {
  const opacity = getOpacity()
  data[0].opacity = opacity
  data[1].opacity = opacity
  Plotly.react(
    'connectome-plot',
    data,
    getLayout('connectome-plot', 'select-view', false),
    getConfig()
  )
}

function surfaceRelayout () {
  return updateLayout('connectome-plot', 'select-view', false)
}

function addConnectome () {
  const info = connectomeInfo.connectome

  for (const attribute of ['marker_x', 'marker_y', 'marker_z']) {
    if (!(attribute in info)) {
      info[attribute] = Array.from(
        decodeBase64(info['_' + attribute], 'float32')
      )
    }
  }

  const pyplot_elements = [
    {
      type: 'scatter3d',
      mode: 'markers+text',
      x: info.marker_x,
      y: info.marker_y,
      z: info.marker_z,
      opacity: 1,
      text: info.marker_labels,
      marker: {
        size: info.marker_size,
        color: info.marker_color,
        colorscale: info.marker_colorscale
      }
    }
  ]

  if (!info.markers_only) {
    for (const attribute of ['con_x', 'con_y', 'con_z', 'con_w']) {
      if (!(attribute in info)) {
        info[attribute] = Array.from(
          decodeBase64(info['_' + attribute], 'float32')
        )
        for (let i = 2; i < info[attribute].length; i += 3) {
          info[attribute][i] = null
        }
      }
    }
    pyplot_elements.push({
      type: 'scatter3d',
      mode: 'lines',
      x: info.con_x,
      y: info.con_y,
      z: info.con_z,
      opacity: 1,
      line: {
        width: info.line_width,
        color: info.con_w,
        colorscale: info.line_colorscale,
        cmin: info.line_cmin,
        cmax: info.line_cmax
      }
    })
  }

  Plotly.plot('connectome-plot', pyplot_elements)
}
