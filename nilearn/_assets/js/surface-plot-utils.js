function makePlot (surface, hemisphere, divId) {
  decodeHemisphere(surfaceMapInfo, surface, hemisphere)
  const info = surfaceMapInfo[surface + '_' + hemisphere]
  info.type = 'mesh3d'
  info.vertexcolor = surfaceMapInfo['vertexcolor_' + hemisphere]

  const data = [info]

  info.lighting = getLighting()
  const layout = getLayout(
    'surface-plot',
    'select-view',
    surfaceMapInfo.black_bg
  )
  layout.title = {
    text: surfaceMapInfo.title,
    font: {
      size: surfaceMapInfo.title_fontsize,
      color: textColor(surfaceMapInfo.black_bg)
    },
    yref: 'paper',
    y: 0.95
  }
  const config = getConfig()

  Plotly.react(divId, data, layout, config)

  if (surfaceMapInfo.colorbar) {
    addColorbar(
      surfaceMapInfo.colorscale,
      surfaceMapInfo.cmin,
      surfaceMapInfo.cmax,
      divId,
      layout,
      config,
      surfaceMapInfo.cbar_fontsize,
      surfaceMapInfo.cbar_height,
      (color = textColor(surfaceMapInfo.black_bg))
    )
  }
}

function addPlot () {
  const hemisphere = $('#select-hemisphere').val()
  const kind = $('#select-kind').val()

  if (surfaceMapInfo.view) {
    $('#select-view').val(surfaceMapInfo.view)
  }
  makePlot(kind, hemisphere, 'surface-plot')
}

function surfaceRelayout () {
  return updateLayout(
    'surface-plot',
    'select-view',
    surfaceMapInfo.black_bg
  )
}
