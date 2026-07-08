/* global __BROWSER__, describe, beforeAll, afterAll, it, expect */

// open the HTML documents, take a screenshot of it, and compare it to a reference screenshot

const path = require('path')
const fs = require('fs')
const PNG = require('pngjs').PNG
const pixelmatch = require('pixelmatch')

const buildFilePNG = function (file, prefix, suffix) {
  return `${path.resolve(__dirname)}/` + prefix + file.split('.')[0] + suffix + '.png'
}

module.exports.fullTest = (file, clip, tolerance = 1000, timeout = 5000) => {
  describe('index page', () => {
    let page

    beforeAll(async () => {
      page = await __BROWSER__.newPage()

      await page.goto('http://localhost:8080/' + file)
      await page.waitForTimeout(timeout)
    }, timeout)

    afterAll(async () => {
      await page.close()
    }, timeout)

    it(
      'visual regression test ' + file,
      async () => {
        // take a screenshot of the page
        const fileCurrent = buildFilePNG(file, '', '')
        await page.screenshot({ clip, path: fileCurrent })

        // archive a copy of the screenshot as future reference, if specified
        const fileReference = buildFilePNG(file, 'references/', '_reference')

        if ('TEST_RUN' in process.env && process.env.TEST_RUN === 'init') {
          fs.copyFileSync(fileCurrent, fileReference)
        } else {
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
          expect(numDiffPixels).toBeLessThan(tolerance)
        }
      },
      timeout
    )
  })
}
