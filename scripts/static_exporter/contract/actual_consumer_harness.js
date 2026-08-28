"use strict";

// ACTUAL BROWSER CONSUMER HARNESS
//
// This does NOT reimplement anything. It loads the production website file
//   edvinli.github.io/assets/js/election-simulator.js
// verbatim, evaluates it in this process, and observes the user-visible
// outcome it produces. Every validation rule exercised here is the deployed
// rule, because the deployed source is what runs.
//
// The website file is an IIFE written for a browser, so the harness supplies
// the small, closed set of globals it touches (see the API inventory below)
// and nothing else. It deliberately does not use jsdom: a real DOM would add
// a heavyweight dependency without making the *validation* path any more
// authentic, since the file's accept/reject decisions never read back from
// the DOM. What the harness asserts on is the terminal `#election-app-status`
// text, which is exactly what a visitor sees.
//
// Usage:  node actual_consumer_harness.js <website-js-path> <publication-dir>
// Output: one JSON object on stdout.
//         exit 0 => the production consumer accepted the publication
//         exit 1 => the production consumer rejected it

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const APP_ID = "election-simulator-app";
const STATUS_ID = "election-app-status";
const LOADING_TEXT = "Loading the latest forecast…";

// ---------------------------------------------------------------------------
// Minimal DOM stand-ins covering exactly the API surface the production file
// touches: getElementById, createElement, getAttribute, setAttribute,
// querySelector, appendChild, addEventListener, and the hidden / className /
// textContent / innerHTML / style / value properties.
// ---------------------------------------------------------------------------

class StubElement {
  constructor(tagName, id) {
    this.tagName = tagName;
    this.id = id || "";
    this.className = "";
    this.hidden = true;
    this.disabled = false;
    this.type = "";
    this.style = {};
    this.children = [];
    this.attributes = {};
    this._textContent = "";
    this._innerHTML = "";
    this._listeners = {};
  }

  get textContent() {
    return this._textContent;
  }

  set textContent(value) {
    this._textContent = String(value);
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    // Assigning innerHTML replaces children in a real DOM; the production file
    // only ever assigns and then re-queries, never reads the markup back.
    this._innerHTML = String(value);
    this.children = [];
  }

  getAttribute(name) {
    return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  addEventListener(type, handler) {
    (this._listeners[type] = this._listeners[type] || []).push(handler);
  }

  dispatchEvent(event) {
    const handlers = this._listeners[event.type] || [];
    handlers.forEach((handler) => handler(event));
    return true;
  }

  querySelector(selector) {
    // The production file queries only for nodes it just wrote via innerHTML
    // and then writes style/textContent onto them. Nothing is ever read back,
    // so a fresh detached stub is behaviourally equivalent here.
    const found = new StubElement("div");
    found.matchedSelector = selector;
    return found;
  }

  // <select> semantics: `value` reflects the first option unless set. The
  // production file relies on this default when it calls update() after
  // populating the group dropdown.
  get value() {
    if (this._value !== undefined) return this._value;
    const firstOption = this.children.find((child) => child.tagName === "option");
    return firstOption ? firstOption.value : "";
  }

  set value(next) {
    this._value = String(next);
  }
}

function buildDocument(publicationBase, statusSettled) {
  const elements = new Map();
  for (const id of [
    "election-simulator-app",
    "election-app-status",
    "election-headline",
    "election-party-cards",
    "election-seats",
    "election-seat-bars",
    "election-parliament",
    "election-government-builder",
    "election-government-parties",
    "election-support-parties",
    "election-government-empty",
    "election-government-results",
    "election-government-alone-result",
    "election-government-support-result",
    "election-changes",
    "election-changes-status",
    "election-changes-content",
    "election-groups",
    "election-group-select",
    "election-group-result",
    "election-validation",
    "election-validation-content",
    "election-meta",
    "election-meta-list",
  ]) {
    elements.set(id, new StubElement("div", id));
  }

  // Mirrors _pages/election_simulator.md, which sets data-publication-base
  // from {{ site.baseurl }}/files/election-simulator.
  elements.get(APP_ID).setAttribute("data-publication-base", publicationBase);

  const status = elements.get(STATUS_ID);
  status._textContent = LOADING_TEXT;
  // The production file writes the status exactly once, in its terminal
  // .then() or .catch(). That write is the signal that loading has settled.
  Object.defineProperty(status, "textContent", {
    get() {
      return this._textContent;
    },
    set(value) {
      this._textContent = String(value);
      statusSettled.resolve();
    },
  });

  return {
    getElementById(id) {
      return elements.has(id) ? elements.get(id) : null;
    },
    createElement(tagName) {
      return new StubElement(tagName);
    },
    _elements: elements,
  };
}

function findButton(host, party) {
  if (!host) return null;
  return host.children.find((child) => child.getAttribute("data-party") === party) || null;
}

function clickButton(host, party) {
  const button = findButton(host, party);
  if (!button) throw new Error("Could not find coalition button for " + party);
  if (button.disabled) throw new Error("Coalition button is unexpectedly disabled for " + party);
  if (!(button._listeners.click || []).length) throw new Error("Coalition button has no click handler for " + party);
  button.dispatchEvent({ type: "click", target: button });
}

function coalitionSnapshot(document) {
  const section = document._elements.get("election-government-builder");
  const government = document._elements.get("election-government-parties");
  const support = document._elements.get("election-support-parties");
  const empty = document._elements.get("election-government-empty");
  const results = document._elements.get("election-government-results");
  const alone = document._elements.get("election-government-alone-result");
  const withSupport = document._elements.get("election-government-support-result");
  const describeButtons = (host) => (host ? host.children : []).map((button) => ({
    party: button.getAttribute("data-party"),
    mask: button.getAttribute("data-mask"),
    pressed: button.getAttribute("aria-pressed"),
    disabled: Boolean(button.disabled),
  }));
  return {
    available: Boolean(section && !section.hidden),
    empty_hidden: Boolean(!empty || empty.hidden),
    results_hidden: Boolean(!results || results.hidden),
    alone_hidden: Boolean(!alone || alone.hidden),
    support_hidden: Boolean(!withSupport || withSupport.hidden),
    government_buttons: describeButtons(government),
    support_buttons: describeButtons(support),
    alone_mask: alone ? alone.getAttribute("data-coalition-mask") : null,
    support_mask: withSupport ? withSupport.getAttribute("data-coalition-mask") : null,
    alone_html: alone ? alone.innerHTML : "",
    support_html: withSupport ? withSupport.innerHTML : "",
  };
}

// ---------------------------------------------------------------------------
// fetch over a local publication directory. A missing file is a 404, which is
// exactly how a static host reports an absent current.json — the one status
// the production file is allowed to fall back on.
// ---------------------------------------------------------------------------

function buildFetch(root, requested) {
  const base = path.resolve(root);
  return function fetchLike(requestPath) {
    const relative = String(requestPath).replace(/^\/+/, "");
    requested.push(relative);
    const resolved = path.resolve(base, relative);
    if (resolved !== base && !resolved.startsWith(base + path.sep)) {
      return Promise.resolve(makeResponse(403, null));
    }
    let stats;
    try {
      stats = fs.lstatSync(resolved);
    } catch (error) {
      return Promise.resolve(makeResponse(404, null));
    }
    if (stats.isSymbolicLink() || !stats.isFile()) {
      return Promise.resolve(makeResponse(404, null));
    }
    return Promise.resolve(makeResponse(200, fs.readFileSync(resolved, "utf8")));
  };
}

function makeResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text() {
      return Promise.resolve(body);
    },
    json() {
      return Promise.resolve(JSON.parse(body));
    },
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

async function main() {
  const [, , websiteJsPath, publicationDir] = process.argv;
  if (!websiteJsPath || !publicationDir) {
    process.stdout.write(
      JSON.stringify({ accepted: false, error: "usage: actual_consumer_harness.js <website-js> <publication-dir>" }) + "\n"
    );
    return 1;
  }

  const source = fs.readFileSync(websiteJsPath, "utf8");
  const statusSettled = deferred();
  const requested = [];
  const document = buildDocument("", statusSettled);

  const sandbox = {
    document,
    fetch: buildFetch(publicationDir, requested),
    crypto: globalThis.crypto,
    TextEncoder,
    Promise,
    Number,
    Object,
    Array,
    JSON,
    Error,
    String,
    Math,
    console,
  };
  sandbox.globalThis = sandbox;
  sandbox.window = sandbox;

  // Run the deployed source verbatim. No transformation, no substitution.
  vm.runInNewContext(source, vm.createContext(sandbox), { filename: websiteJsPath });

  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error("Production consumer never settled its status")), 15000)
  );
  await Promise.race([statusSettled.promise, timeout]);

  const status = document._elements.get(STATUS_ID);
  const statusText = status.textContent;
  const errored = status.className.indexOf("election-status--error") !== -1;
  const parliament = document._elements.get("election-parliament");
  const builderInitial = coalitionSnapshot(document);
  let builderGovernment = null;
  let builderWithSupport = null;
  if (builderInitial.available) {
    for (const party of ["M", "KD", "SD"]) {
      clickButton(document._elements.get("election-government-parties"), party);
    }
    builderGovernment = coalitionSnapshot(document);
    clickButton(document._elements.get("election-support-parties"), "L");
    builderWithSupport = coalitionSnapshot(document);
  }

  const verdict = {
    accepted: !errored,
    status_text: statusText,
    certified: statusText === "Certified forecast loaded.",
    error: errored ? statusText.replace(/^Forecast unavailable: /, "") : null,
    // aria-label records which seat-allocation path the production file took.
    parliament_aria_label: parliament.getAttribute("aria-label"),
    seat_nodes: parliament.children.length,
    requested_paths: requested,
    builder_initial: builderInitial,
    builder_government: builderGovernment,
    builder_with_support: builderWithSupport,
    source_file: path.resolve(websiteJsPath),
    source_sha256: require("crypto").createHash("sha256").update(source).digest("hex"),
  };
  process.stdout.write(JSON.stringify(verdict) + "\n");
  return verdict.accepted ? 0 : 1;
}

main().then(
  (code) => process.exit(code),
  (error) => {
    process.stdout.write(
      JSON.stringify({ accepted: false, error: String((error && error.message) || error) }) + "\n"
    );
    process.exit(1);
  }
);
