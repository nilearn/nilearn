window.jupyterliteShowIframe = (tryItButtonId, iframeSrc) => {
  const tryItButton = document.getElementById(tryItButtonId);
  const iframe = document.createElement("iframe");
  const buttonRect = tryItButton.getBoundingClientRect();

  const spinner = document.createElement("div");
  // hardcoded spinner height and width needs to match what is in css.
  const spinnerHeight = 50; // px
  const spinnerWidth = 50; // px
  spinner.classList.add("jupyterlite_sphinx_spinner");
  spinner.style.display = "none";
  // Add negative margins to center the spinner
  spinner.style.marginTop = `-${spinnerHeight / 2}px`;
  spinner.style.marginLeft = `-${spinnerWidth / 2}px`;

  iframe.src = iframeSrc;
  iframe.width = iframe.height = "100%";
  iframe.classList.add("jupyterlite_sphinx_iframe");

  tryItButton.style.display = "none";
  spinner.style.display = "block";

  tryItButton.parentNode.appendChild(spinner);
  tryItButton.parentNode.appendChild(iframe);
};

window.jupyterliteConcatSearchParams = (iframeSrc, params) => {
  const baseURL = window.location.origin;
  const iframeUrl = new URL(iframeSrc, baseURL);

  let pageParams = new URLSearchParams(window.location.search);

  if (params === true) {
    params = Array.from(pageParams.keys());
  } else if (params === false) {
    params = [];
  } else if (!Array.isArray(params)) {
    console.error("The search parameters are not an array");
  }

  params.forEach((param) => {
    value = pageParams.get(param);
    if (value !== null) {
      iframeUrl.searchParams.append(param, value);
    }
  });

  if (iframeUrl.searchParams.size) {
    return `${iframeSrc.split("?")[0]}?${iframeUrl.searchParams.toString()}`;
  } else {
    return iframeSrc;
  }
};

window.tryExamplesShowIframe = (
  examplesContainerId,
  iframeContainerId,
  iframeParentContainerId,
  iframeSrc,
  iframeHeight,
) => {
  const examplesContainer = document.getElementById(examplesContainerId);
  const iframeParentContainer = document.getElementById(
    iframeParentContainerId,
  );
  const iframeContainer = document.getElementById(iframeContainerId);
  var height;

  let iframe = iframeContainer.querySelector(
    "iframe.jupyterlite_sphinx_iframe",
  );

  if (!iframe) {
    // Add spinner
    const spinner = document.createElement("div");
    // hardcoded spinner width needs to match what is in css.
    const spinnerHeight = 50; // px
    const spinnerWidth = 50; // px
    spinner.classList.add("jupyterlite_sphinx_spinner");
    iframeContainer.appendChild(spinner);

    const examples = examplesContainer.querySelector(".try_examples_content");
    iframe = document.createElement("iframe");
    iframe.src = iframeSrc;
    iframe.style.width = "100%";
    if (iframeHeight !== "None") {
      height = parseInt(iframeHeight);
    } else {
      height = Math.max(tryExamplesGlobalMinHeight, examples.offsetHeight);
    }

    /* Get spinner position. It will be centered in the iframe, unless the
     * iframe extends beyond the viewport, in which case it will be centered
     * between the top of the iframe and the bottom of the viewport.
     */
    const examplesTop = examples.getBoundingClientRect().top;
    const viewportBottom = window.innerHeight;
    const spinnerTop = 0.5 * Math.min(viewportBottom - examplesTop, height);
    spinner.style.top = `${spinnerTop}px`;
    // Add negative margins to center the spinner
    spinner.style.marginTop = `-${spinnerHeight / 2}px`;
    spinner.style.marginLeft = `-${spinnerWidth / 2}px`;

    iframe.style.height = `${height}px`;
    iframe.classList.add("jupyterlite_sphinx_iframe");
    examplesContainer.classList.add("hidden");

    iframeContainer.appendChild(iframe);
  } else {
    examplesContainer.classList.add("hidden");
  }
  iframeParentContainer.classList.remove("hidden");
};

window.tryExamplesHideIframe = (
  examplesContainerId,
  iframeParentContainerId,
) => {
  const examplesContainer = document.getElementById(examplesContainerId);
  const iframeParentContainer = document.getElementById(
    iframeParentContainerId,
  );

  iframeParentContainer.classList.add("hidden");
  examplesContainer.classList.remove("hidden");
};

// this will be used by the "Open in tab" button that is present next
// # to the "go back" button after an iframe is made visible.
window.openInNewTab = (examplesContainerId, iframeParentContainerId) => {
  const examplesContainer = document.getElementById(examplesContainerId);
  const iframeParentContainer = document.getElementById(
    iframeParentContainerId,
  );

  window.open(
    // we make some assumption that there is a single iframe and the the src is what we want to open.
    // Maybe we should have tabs open JupyterLab by default.
    iframeParentContainer.getElementsByTagName("iframe")[0].getAttribute("src"),
  );
  tryExamplesHideIframe(examplesContainerId, iframeParentContainerId);
};

/* Global variable for try_examples iframe minHeight. Defaults to 0 but can be
 * modified based on configuration in try_examples.json */
var tryExamplesGlobalMinHeight = 0;
/* Global variable to check if config has been loaded. This keeps it from getting
 * loaded multiple times if there are multiple try_examples directives on one page
 */
var tryExamplesConfigLoaded = false;

// This function is used to check if the current device is a mobile device.
// We assume the authenticity of the user agent string is enough to
// determine that, and we also check the window size as a fallback.
window.isMobileDevice = (() => {
  let cachedUAResult = null;
  let hasLogged = false;

  const checkUserAgent = () => {
    if (cachedUAResult !== null) {
      return cachedUAResult;
    }

    const mobilePatterns = [
      /Android/i,
      /webOS/i,
      /iPhone/i,
      /iPad/i,
      /iPod/i,
      /BlackBerry/i,
      /IEMobile/i,
      /Windows Phone/i,
      /Opera Mini/i,
      /SamsungBrowser/i,
      /UC.*Browser|UCWEB/i,
      /MiuiBrowser/i,
      /Mobile/i,
      /Tablet/i,
    ];

    cachedUAResult = mobilePatterns.some((pattern) =>
      pattern.test(navigator.userAgent),
    );
    return cachedUAResult;
  };

  return () => {
    const isMobileBySize =
      window.innerWidth <= 480 || window.innerHeight <= 480;
    const isLikelyMobile = checkUserAgent() || isMobileBySize;

    if (isLikelyMobile && !hasLogged) {
      console.log(
        "Either a mobile device detected or the screen was resized. Disabling interactive example buttons to conserve bandwidth.",
      );
      hasLogged = true;
    }

    return isLikelyMobile;
  };
})();

// A config loader with request deduplication + permanent caching
const ConfigLoader = (() => {
  let configLoadPromise = null;

  const loadConfig = async (configFilePath) => {
    if (window.isMobileDevice()) {
      const buttons = document.getElementsByClassName("try_examples_button");
      for (let i = 0; i < buttons.length; i++) {
        buttons[i].classList.add("hidden");
      }
      tryExamplesConfigLoaded = true; // mock it
      return;
    }

    if (tryExamplesConfigLoaded) {
      return;
    }

    // Return the existing promise if the request is in progress, as we
    // don't want to make multiple requests for the same file. This
    // can happen if there are several try_examples directives on the
    // same page.
    if (configLoadPromise) {
      return configLoadPromise;
    }

    // Create and cache the promise for the config request
    configLoadPromise = (async () => {
      try {
        // Add a timestamp as query parameter to ensure a cached version of the
        // file is not used.
        const timestamp = new Date().getTime();
        const configFileUrl = `${configFilePath}?cb=${timestamp}`;
        const currentPageUrl = window.location.pathname;

        const response = await fetch(configFileUrl);
        if (!response.ok) {
          if (response.status === 404) {
            console.log("Optional try_examples config file not found.");
            return;
          }
          throw new Error(`Error fetching ${configFilePath}`);
        }

        const data = await response.json();
        if (!data) {
          return;
        }

        // Set minimum iframe height based on value in config file
        if (data.global_min_height) {
          tryExamplesGlobalMinHeight = parseInt(data.global_min_height);
        }

        // Disable interactive examples if file matches one of the ignore patterns
        // by hiding try_examples_buttons.
        Patterns = data.ignore_patterns;
        for (let pattern of Patterns) {
          let regex = new RegExp(pattern);
          if (regex.test(currentPageUrl)) {
            var buttons = document.getElementsByClassName(
              "try_examples_button",
            );
            for (var i = 0; i < buttons.length; i++) {
              buttons[i].classList.add("hidden");
            }
            break;
          }
        }
      } catch (error) {
        console.error(error);
      } finally {
        tryExamplesConfigLoaded = true;
      }
    })();

    return configLoadPromise;
  };

  return {
    loadConfig,
    // for testing/debugging only, could be removed
    resetState: () => {
      tryExamplesConfigLoaded = false;
      configLoadPromise = null;
    },
  };
})();

// Add a resize handler that will update the buttons' visibility on
// orientation changes
let resizeTimeout;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (!tryExamplesConfigLoaded) return; // since we won't interfere if the config isn't loaded

    const buttons = document.getElementsByClassName("try_examples_button");
    const shouldHide = window.isMobileDevice();

    for (let i = 0; i < buttons.length; i++) {
      if (shouldHide) {
        buttons[i].classList.add("hidden");
      } else {
        buttons[i].classList.remove("hidden");
      }
    }
  }, 250);
});

window.loadTryExamplesConfig = ConfigLoader.loadConfig;

window.toggleTryExamplesButtons = () => {
  /* Toggle visibility of TryExamples buttons. For use in console for debug
   * purposes. */
  var buttons = document.getElementsByClassName("try_examples_button");

  for (var i = 0; i < buttons.length; i++) {
    buttons[i].classList.toggle("hidden");
  }
};
