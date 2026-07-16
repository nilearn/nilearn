/* eslint-disable no-console */

// Compare gallery example images ('sphx_glr_plot_*.png') between the
// 'stable' and 'dev' builds of the doc, published on nilearn.github.io.
// Gives an idea of how much example outputs visually change across
// nilearn versions. See https://github.com/nilearn/nilearn/issues/6342

'use strict'

const fs = require('fs')
const path = require('path')
const { PNG } = require('pngjs')
const pixelmatch = require('pixelmatch')

// expects a clone of nilearn/nilearn.github.io at this path, with at least
// dev/_images and stable/_images checked out (see CI workflow)
const ROOT = path.resolve(__dirname, '..')
const TMP_DIR = path.join(ROOT, 'tmp')
const CLONE_DIR = path.join(TMP_DIR, 'nilearn.github.io')
const DIFF_DIR = path.join(TMP_DIR, 'doc_image_diffs')
const IGNORE_FILE = path.join(__dirname, 'compare_doc_images_ignore.txt')

// per-pixel color threshold, same value used in tests/js/template.js
const PIXELMATCH_THRESHOLD = 0.2
// flag images with more than 1% of pixels differing
const DIFF_RATIO_TOLERANCE = 0.01

function loadIgnorePatterns (filePath) {
  if (!fs.existsSync(filePath)) {
    return []
  }
  return fs
    .readFileSync(filePath, 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
}

function globToRegExp (pattern) {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&')
  return new RegExp('^' + escaped.replace(/\*/g, '.*') + '$')
}

function isIgnored (name, patterns) {
  return patterns.some((pattern) => globToRegExp(pattern).test(name))
}

// strip the trailing '_<number>.png' index so consecutive outputs of the
// same example (e.g. sphx_glr_plot_foo_001.png, ..._002.png) are grouped
function baseName (name) {
  return name.replace(/_\d+\.png$/, '.png')
}

function checkClone (stableDir, devDir) {
  for (const dir of [stableDir, devDir]) {
    if (!fs.existsSync(dir)) {
      console.error(`Expected directory not found: ${dir}`)
      console.error(
        'Clone nilearn/nilearn.github.io into ' +
        `${CLONE_DIR} first (with dev/_images and stable/_images checked out).`
      )
      process.exit(1)
    }
  }
}

function listGalleryImages (dir) {
  return fs
    .readdirSync(dir)
    .filter((name) => name.startsWith('sphx_glr_plot_') && name.endsWith('.png'))
    .filter((name) => !name.endsWith('_thumb.png'))
}

function compareImage (name, stableDir, devDir) {
  const imgStable = PNG.sync.read(fs.readFileSync(path.join(stableDir, name)))
  const imgDev = PNG.sync.read(fs.readFileSync(path.join(devDir, name)))

  if (imgStable.width !== imgDev.width || imgStable.height !== imgDev.height) {
    return { name, status: 'size-changed', diffRatio: null }
  }

  const { width, height } = imgStable
  const imgDiff = new PNG({ width, height })
  const numDiffPixels = pixelmatch(
    imgStable.data,
    imgDev.data,
    imgDiff.data,
    width,
    height,
    { threshold: PIXELMATCH_THRESHOLD }
  )
  const diffRatio = numDiffPixels / (width * height)

  if (diffRatio > 0) {
    fs.mkdirSync(DIFF_DIR, { recursive: true })
    fs.writeFileSync(path.join(DIFF_DIR, name), PNG.sync.write(imgDiff))
  }

  return {
    name,
    status: diffRatio > DIFF_RATIO_TOLERANCE ? 'changed' : 'unchanged',
    diffRatio
  }
}

function main () {
  const stableDir = path.join(CLONE_DIR, 'stable', '_images')
  const devDir = path.join(CLONE_DIR, 'dev', '_images')

  checkClone(stableDir, devDir)

  const stableImages = new Set(listGalleryImages(stableDir))
  const devImages = new Set(listGalleryImages(devDir))

  const onlyInStable = [...stableImages].filter((n) => !devImages.has(n))
  const onlyInDev = [...devImages].filter((n) => !stableImages.has(n))
  const shared = [...stableImages].filter((n) => devImages.has(n))

  const ignorePatterns = loadIgnorePatterns(IGNORE_FILE)
  const ignored = shared.filter((n) => isIgnored(n, ignorePatterns))
  const toCompare = shared.filter((n) => !isIgnored(n, ignorePatterns))

  console.log(`\nCompared ${toCompare.length} gallery image(s) present in both stable and dev.`)
  if (onlyInStable.length) {
    console.log(`${onlyInStable.length} image(s) only in stable (removed in dev).`)
  }
  if (onlyInDev.length) {
    console.log(`${onlyInDev.length} image(s) only in dev (new since stable).`)
  }
  if (ignored.length) {
    console.log(
      `${ignored.length} image(s) ignored per ${path.basename(IGNORE_FILE)}.`
    )
  }

  const results = toCompare
    .map((name) => compareImage(name, stableDir, devDir))
    .sort((a, b) => {
      if (a.name !== b.name) {
        return a.name.localeCompare(b.name)
      }
      return (b.diffRatio || 0) - (a.diffRatio || 0)
    })

  const changed = results.filter((r) => r.status !== 'unchanged')
  const sizeChanged = results.filter((r) => r.status === 'size-changed')
  const pixelChanged = results.filter((r) => r.status === 'changed')

  console.log(
    `\n${changed.length} image(s) changed beyond tolerance (${DIFF_RATIO_TOLERANCE * 100}% of pixels):\n`
  )
  let previousBaseName = null
  for (const r of changed) {
    if (previousBaseName !== null && baseName(r.name) !== previousBaseName) {
      console.log('')
    }
    previousBaseName = baseName(r.name)

    const pct = r.diffRatio === null ? 'n/a' : (r.diffRatio * 100).toFixed(2) + '%'
    console.log(`  [${r.status}] ${r.name} - ${pct} pixels differ`)
  }

  if (changed.length) {
    console.log(`\nDiff images written to ${DIFF_DIR}`)
  }

  if (sizeChanged.length) {
    console.log(
      `\n${sizeChanged.length} image(s) changed dimensions (ignored for pass/fail).`
    )
  }

  if (pixelChanged.length) {
    console.error(
      `\nFAIL: ${pixelChanged.length} image(s) exceed the pixel diff tolerance.`
    )
    process.exit(1)
  }
}

main()
