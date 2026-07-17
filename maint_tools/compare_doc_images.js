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

const supportsColor =
  !process.env.NO_COLOR &&
  (process.stdout.isTTY || Boolean(process.env.FORCE_COLOR))

/**
 * Build a colorizer for a given ANSI SGR code.
 *
 * @param {string} code - ANSI SGR code, e.g. '1' for bold, '31' for red.
 * @returns {(text: string) => string} Function wrapping text in the ANSI
 *   escape sequence, or returning it unchanged if color is not supported.
 */
function ansi (code) {
  return (text) => (supportsColor ? `\x1b[${code}m${text}\x1b[0m` : text)
}

const style = {
  bold: ansi('1'),
  dim: ansi('2'),
  red: ansi('31'),
  green: ansi('32'),
  yellow: ansi('33'),
  magenta: ansi('35'),
  cyan: ansi('36')
}

/**
 * Load glob patterns from an ignore file.
 *
 * @param {string} filePath - Path to the ignore file. One pattern per
 *   line; blank lines and lines starting with '#' are skipped.
 * @returns {string[]} The patterns, or an empty array if the file
 *   doesn't exist.
 */
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

/**
 * Convert a glob pattern (only '*' is supported as a wildcard) to a
 * RegExp matching the whole string.
 *
 * @param {string} pattern - Glob pattern, e.g. 'sphx_glr_plot_foo_*.png'.
 * @returns {RegExp} Equivalent anchored regular expression.
 */
function globToRegExp (pattern) {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&')
  return new RegExp('^' + escaped.replace(/\*/g, '.*') + '$')
}

/**
 * Check whether a file name matches any of the given glob patterns.
 *
 * @param {string} name - File name to test.
 * @param {string[]} patterns - Glob patterns, as returned by
 *   {@link loadIgnorePatterns}.
 * @returns {boolean} Whether `name` matches at least one pattern.
 */
function isIgnored (name, patterns) {
  return patterns.some((pattern) => globToRegExp(pattern).test(name))
}

/**
 * Derive the example name from one of its gallery image file names, e.g.
 * 'sphx_glr_plot_foo_001.png' -> 'plot_foo', by stripping the 'sphx_glr_'
 * prefix and the trailing '_<number>.png' index. Used to group an
 * example's numbered outputs together and to locate its source script.
 *
 * @param {string} imageName - Gallery image file name.
 * @returns {string} The example's base name.
 */
function exampleName (imageName) {
  return imageName.replace(/^sphx_glr_/, '').replace(/_\d+\.png$/, '')
}

const examplePathCache = new Map()

/**
 * Find the path of the script that generates a given example, by
 * walking the 'examples/' directory tree. Results are memoized in
 * `examplePathCache` since the same example is looked up once per
 * group of changed images.
 *
 * @param {string} name - Example base name, as returned by
 *   {@link exampleName} (without the '.py' extension).
 * @returns {string|null} Path of the example script, relative to the
 *   'examples/' directory, or `null` if it could not be found.
 */
function findExampleRelPath (name) {
  if (examplePathCache.has(name)) {
    return examplePathCache.get(name)
  }

  const examplesDir = path.join(ROOT, 'examples')
  let found = null

  function walk (dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name)
      if (found) {
        return
      }
      if (entry.isDirectory()) {
        walk(fullPath)
      } else if (entry.name === `${name}.py`) {
        found = path.relative(examplesDir, fullPath)
      }
    }
  }

  walk(examplesDir)
  examplePathCache.set(name, found)
  return found
}

/**
 * Build the URL of the dev doc page for a given example, e.g.
 * '01_plotting/plot_haxby_masks.py' ->
 * https://nilearn.github.io/dev/auto_examples/01_plotting/plot_haxby_masks.html#sphx-glr-auto-examples-01-plotting-plot-haxby-masks-py
 *
 * @param {string} relPath - Example script path, relative to the
 *   'examples/' directory, as returned by {@link findExampleRelPath}.
 * @returns {string} URL of the example's dev doc page.
 */
function devDocURL (relPath) {
  const posixPath = relPath.split(path.sep).join('/')
  const withoutExt = posixPath.replace(/\.py$/, '')
  const anchor = 'sphx-glr-auto-examples-' + posixPath.replace(/[/_.]/g, '-')
  return `https://nilearn.github.io/dev/auto_examples/${withoutExt}.html#${anchor}`
}

/**
 * Exit the process with an error if the stable/dev image directories of
 * the nilearn.github.io clone are missing.
 *
 * @param {string} stableDir - Expected path of 'stable/_images'.
 * @param {string} devDir - Expected path of 'dev/_images'.
 * @returns {void}
 */
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

/**
 * List gallery example images ('sphx_glr_plot_*.png', excluding
 * thumbnails) in a doc build's '_images' directory.
 *
 * @param {string} dir - Path to a '_images' directory.
 * @returns {string[]} Matching image file names.
 */
function listGalleryImages (dir) {
  return fs
    .readdirSync(dir)
    .filter((name) => name.startsWith('sphx_glr_plot_') && name.endsWith('.png'))
    .filter((name) => !name.endsWith('_thumb.png'))
}

/**
 * Compare one gallery image between the stable and dev doc builds.
 * Writes a pixelmatch diff image under `DIFF_DIR` whenever any pixel
 * differs.
 *
 * @param {string} name - Image file name, present in both directories.
 * @param {string} stableDir - Path to the stable build's '_images'.
 * @param {string} devDir - Path to the dev build's '_images'.
 * @returns {{name: string, status: 'size-changed'|'changed'|'unchanged',
 *   diffRatio: number|null}} Comparison result; `diffRatio` is `null`
 *   when the two images' dimensions differ (`status: 'size-changed'`),
 *   since pixelmatch cannot diff images of different sizes.
 */
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

/**
 * Print a labeled, sorted list of image file names, e.g. images only
 * found in one of the two doc builds, which can be a sign that an
 * example failed to build, was renamed, or was added/removed.
 *
 * @param {string} label - Describes what the listed images have in
 *   common, e.g. 'only in dev (new since stable)'.
 * @param {string[]} names - Image file names to list.
 * @returns {void}
 */
function printFileList (label, names) {
  if (!names.length) {
    return
  }
  console.log(style.yellow(`\n${names.length} image(s) ${label}:`))
  for (const name of [...names].sort()) {
    console.log(style.yellow(`  ${name}`))
  }
}

/**
 * Compare the gallery images of the stable and dev doc builds, print a
 * report grouped by example, and exit with a non-zero status if any
 * image's pixel diff exceeds `DIFF_RATIO_TOLERANCE` (dimension-only
 * changes are reported but don't affect the exit status).
 *
 * @returns {void}
 */
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

  console.log(
    `\nCompared ${style.bold(toCompare.length)} gallery image(s) present in both stable and dev.`
  )
  printFileList('only in stable (removed in dev)', onlyInStable)
  printFileList('only in dev (new since stable)', onlyInDev)
  if (ignored.length) {
    console.log(
      style.dim(`${ignored.length} image(s) ignored per ${path.basename(IGNORE_FILE)}.`)
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

  const changedHeader = `${changed.length} image(s) changed beyond tolerance (${DIFF_RATIO_TOLERANCE * 100}% of pixels):`
  console.log(`\n${changed.length ? style.bold(changedHeader) : style.green(changedHeader)}\n`)

  let previousExampleName = null
  for (const r of changed) {
    const name = exampleName(r.name)
    if (name !== previousExampleName) {
      if (previousExampleName !== null) {
        console.log('')
      }
      const relPath = findExampleRelPath(name)
      console.log(
        relPath
          ? `${style.bold(name)}: ${style.cyan(devDocURL(relPath))}`
          : `${style.bold(name)}: ${style.dim('(example script not found)')}`
      )
      previousExampleName = name
    }

    const tag = r.status === 'changed' ? style.yellow(`[${r.status}]`) : style.magenta(`[${r.status}]`)
    const pct = r.diffRatio === null
      ? style.dim('n/a')
      : style[r.diffRatio > 0.05 ? 'red' : 'yellow']((r.diffRatio * 100).toFixed(2) + '%')
    console.log(`  ${tag} ${r.name} - ${pct} pixels differ`)
  }

  if (changed.length) {
    console.log(style.dim(`\nDiff images written to ${DIFF_DIR}`))
  }

  if (sizeChanged.length) {
    console.log(
      style.magenta(`\n${sizeChanged.length} image(s) changed dimensions (ignored for pass/fail).`)
    )
  }

  if (pixelChanged.length) {
    console.error(
      style.red(style.bold(`\nFAIL: ${pixelChanged.length} image(s) exceed the pixel diff tolerance.`))
    )
    process.exit(1)
  } else {
    console.log(style.green('\nPASS: no images exceed the pixel diff tolerance.'))
  }
}

main()
