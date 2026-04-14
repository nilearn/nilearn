const template = require('./template.js')

const file = 'nifti_masker_report.html'
template.fullTest(file,
  // height was adapted to crop footer from png
  // to avoid the change in nilearn version from appearing
  { x: 0, y: 0, width: 800, height: 550 }
)
