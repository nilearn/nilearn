const template = require('./template.js')

const file = 'nifti_masker_report.html'
template.fullTest(file,
  { x: 0, y: 0, width: 800, height: 340 }
)
