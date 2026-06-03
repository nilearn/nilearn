const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 6000 }

const glms = ['slm_oasis', 'flm_bids_features', 'flm_surf']

for (let i = 0; i < glms.length; i++) {
  template.fullTest(glms[i] + '.html', VIEWPORT)
}
