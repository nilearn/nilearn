const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 750 }

const maskers = ['NiftiMasker', 'NiftiLabelsMasker', 'NiftiMapsMasker', 'SurfaceMasker', 'SurfaceLabelsMasker', 'SurfaceMapsMasker']

for (let i = 0; i < maskers.length; i++) {
  template.fullTest(maskers[i] + '_fitted.html', VIEWPORT)
}
