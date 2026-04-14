/* global __BROWSER__ */

const path = require('path')
const fs = require('fs')
const PNG = require('pngjs').PNG
const pixelmatch = require('pixelmatch')

const buildFilePNG = function (file, prefix, suffix) {
  return `${path.resolve(__dirname)}/` + prefix + file.split('.')[0] + suffix + '.png'
}

module.exports.fullTest = (file, clip) => {
  describe('index page', () => {
    let page

    beforeAll(async () => {
      page = await __BROWSER__.newPage()
      await page.coverage.startJSCoverage()
      await page.goto('http://localhost:8080/' + file)
    }, 5000)

    afterAll(async () => {
      const jsCoverage = await page.coverage.stopJSCoverage()
      const pti = require('puppeteer-to-istanbul')
      pti.write([...jsCoverage], { includeHostname: true, storagePath: './.nyc_output' })

      await page.close()
    })

    it(
      'visual regression test',
      async () => {
        // take a screenshot of the page
        const fileCurrent = buildFilePNG(file, '', '')
        await page.screenshot({ clip, path: fileCurrent })

        // copy the screenshot as a thumbnail in the docs
        // also archive a copy of the screenshot as future reference, if specified
        const fileThumb = buildFilePNG(file, '../../docs/build/html/_images/sphx_glr_', '_thumb')
        fs.copyFileSync(fileCurrent, fileThumb)
        const fileReference = buildFilePNG(file, '', '_reference')
        if ('TEST_RUN' in process.env && process.env.TEST_RUN === 'init') {
          fs.copyFileSync(fileCurrent, fileReference)
        }

        // Compare the current and reference snapshots.
        // Trigger an error if there is any difference
        // and create a difference image
        const fileDiff = buildFilePNG(file, '', '_diff')
        const imgCurrent = PNG.sync.read(fs.readFileSync(fileCurrent))
        const imgReference = PNG.sync.read(fs.readFileSync(fileReference))
        const { width, height } = imgCurrent
        const imgDiff = new PNG({ width, height })
        const numDiffPixels = pixelmatch(imgCurrent.data, imgReference.data, imgDiff.data, width, height, { threshold: 0.2 })
        fs.writeFileSync(fileDiff, PNG.sync.write(imgDiff))
        expect(numDiffPixels).toBeLessThan(1000)
      },
      5000
    )
  })
}
