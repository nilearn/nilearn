const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 800 }

const functions = ['img', 'surf', 'connectome', 'markers', 'img_on_surf']

for (let i = 0; i < functions.length; i++) {
  template.fullTest('view_' + functions[i] + '.html', VIEWPORT)
}
