const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 800 }

const functions = ['img', 'surf', 'surf_niivue', 'connectome', 'markers', 'img_on_surf']

const tolerance = 1100

const timeout = 15000

for (let i = 0; i < functions.length; i++) {
  template.fullTest('view_' + functions[i] + '.html', VIEWPORT, tolerance, timeout)
}
