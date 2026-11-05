// Load GoatCounter page view analytics (no cookies, no personal data).
// See https://www.goatcounter.com/
//
// Only report page views from the deployed documentation,
// not from local builds or pull request previews.
(function () {
  if (window.location.hostname !== 'nilearn.github.io') {
    return
  }
  const script = document.createElement('script')
  script.async = true
  script.src = 'https://gc.zgo.at/count.js'
  script.dataset.goatcounter = 'https://nilearn.goatcounter.com/count'
  document.head.appendChild(script)
})()
