const template = require('./template.js')

const maskers = ['NiftiMasker', 'NiftiLabelsMasker', 'NiftiMapsMasker', 'SurfaceMasker', 'SurfaceLabelsMasker', 'SurfaceMapsMasker']

for (let i = 0; i < maskers.length; i++) {
  template.fullTest(maskers[i] + '_fitted.html',
    { x: 0, y: 0, width: 800, height: 650 }
  )
}
