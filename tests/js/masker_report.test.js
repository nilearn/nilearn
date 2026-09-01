const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 750 }

const timeout = 5000

// tolerance might vary due to font issue
const maskers = [
  { masker: 'NiftiMasker_matplotlib', tolerance: 2500, timeout },
  { masker: 'NiftiMasker_brainsprite', tolerance: 2500, timeout },
  { masker: 'NiftiLabelsMasker_matplotlib', tolerance: 1500, timeout },
  { masker: 'NiftiLabelsMasker_brainsprite', tolerance: 1500, timeout },
  { masker: 'NiftiMapsMasker', tolerance: 4100, timeout },
  { masker: 'SurfaceMasker', tolerance: 1600, timeout },
  { masker: 'SurfaceLabelsMasker', tolerance: 1800, timeout },
  { masker: 'SurfaceMapsMasker_matplotlib', tolerance: 3000, timeout },
  { masker: 'SurfaceMapsMasker_plotly', tolerance: 3000, timeout: 10000 }
]

for (let i = 0; i < maskers.length; i++) {
  template.fullTest(maskers[i].masker + '_fitted.html', VIEWPORT, maskers[i].tolerance, maskers[i].timeout)
}
