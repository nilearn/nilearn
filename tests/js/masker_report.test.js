const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 750 }

const maskers = [
  { masker: 'NiftiMasker', tolerance: 1000 },
  { masker: 'NiftiLabelsMasker', tolerance: 1000 },
  { masker: 'NiftiMapsMasker', tolerance: 3000 }, // needs slightly higher tolerance due to font issue
  { masker: 'SurfaceMasker', tolerance: 1000 },
  { masker: 'SurfaceLabelsMasker', tolerance: 1000 },
  { masker: 'SurfaceMapsMasker', tolerance: 1000 }
]

for (let i = 0; i < maskers.length; i++) {
  template.fullTest(maskers[i].masker + '_fitted.html', VIEWPORT, maskers[i].tolerance)
}
