/* global surfaceMapInfo, textColor, addColorbar, updateLayout, getLayout, decodeHemisphere, getLighting, getConfig */

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
    const color = textColor(surfaceMapInfo.black_bg)

    addColorbar(
      surfaceMapInfo.colorscale,
      surfaceMapInfo.cmin,
      surfaceMapInfo.cmax,
      divId,
      layout,
      config,
      surfaceMapInfo.cbar_fontsize,
      surfaceMapInfo.cbar_height,
      (color)
    )
  }
}

function addPlot () { // eslint-disable-line no-unused-vars
  const hemisphere = $('#select-hemisphere').val()
  const kind = $('#select-kind').val()

  if (surfaceMapInfo.view) {
    $('#select-view').val(surfaceMapInfo.view)
  }
  makePlot(kind, hemisphere, 'surface-plot')
}

function surfaceRelayout () { // eslint-disable-line no-unused-vars
  return updateLayout(
    'surface-plot',
    'select-view',
    surfaceMapInfo.black_bg
  )
}
