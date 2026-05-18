const template = require('./template.js')

const glms = ['slm_oasis', 'flm_bids_features']

for (let i = 0; i < glms.length; i++) {
  template.fullTest(glms[i] + '.html',
  // height was adapted to crop footer from png
  // to avoid the change in nilearn version from appearing
    { x: 0, y: 0, width: 800, height: 550 }
  )
}
