const template = require('./template.js')

const functions = ['img', 'surf', 'connectome', 'markers', 'img_on_surf']

for (let i = 0; i < functions.length; i++) {
  template.fullTest('view_' + functions[i] + '.html',
    { x: 0, y: 0, width: 800, height: 600 }
  )
}
