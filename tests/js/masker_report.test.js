const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 750 }

// tolerance might vary due to font issue
const maskers = [
  { masker: 'NiftiMasker_matplotlib', tolerance: 2500 },
  { masker: 'NiftiMasker_brainsprite', tolerance: 2500 },
  { masker: 'NiftiLabelsMasker_matplotlib', tolerance: 1500 },
  { masker: 'NiftiLabelsMasker_brainsprite', tolerance: 1500 },
  { masker: 'NiftiMapsMasker', tolerance: 3000 },
  { masker: 'SurfaceMasker', tolerance: 1600 },
  { masker: 'SurfaceLabelsMasker', tolerance: 1800 },
  { masker: 'SurfaceMapsMasker_matplotlib', tolerance: 3000 },
  { masker: 'SurfaceMapsMasker_plotly', tolerance: 3000 }
]

for (let i = 0; i < maskers.length; i++) {
  template.fullTest(maskers[i].masker + '_fitted.html', VIEWPORT, maskers[i].tolerance)
}
