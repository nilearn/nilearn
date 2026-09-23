module.exports = async function () {
  global.__SERVER__.close()
  await global.__BROWSER__.close()
}
