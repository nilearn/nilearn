const NodeEnvironment = require('jest-environment-node').default
const puppeteer = require('puppeteer')

class TestEnvironment extends NodeEnvironment {
  async setup () {
    await super.setup()

    // Launch browser (instead of connecting to an external one)
    this.global.__BROWSER__ = await puppeteer.launch({
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    })
  }

  async teardown () {
    if (this.global.__BROWSER__) {
      await this.global.__BROWSER__.close()
    }
    await super.teardown()
  }

  runScript (script) {
    return super.runScript(script)
  }
}

module.exports = TestEnvironment
