const template = require('./template.js')

const VIEWPORT = { x: 0, y: 0, width: 1200, height: 6000 }

// tolerance might vary due to font issue
const glms = [
  { glm: 'slm_oasis', tolerance: 3500 },
  { glm: 'flm_bids_features', tolerance: 5000 }
]

for (let i = 0; i < glms.length; i++) {
  template.fullTest(glms[i].glm + '.html', VIEWPORT, glms[i].tolerance)
}
