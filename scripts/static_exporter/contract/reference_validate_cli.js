"use strict";

// Node CLI for the REFERENCE validator (not the production consumer; see
// reference_publication_validator.js). Runs it against a local publication
// directory.  Missing files reject with `.status = 404`, which is
// exactly how a static host reports an absent current.json, so the pointer and
// legacy-fallback branches are both exercised for real.
//
// Usage:  node validate_publication_cli.js <publication-dir>
// Output: one JSON object on stdout.  Exit 0 when the publication is accepted,
//         1 when the browser consumer would reject it.

const fs = require("fs");
const path = require("path");

const { acceptPublication } = require("./reference_publication_validator.js");

function makeDirectoryFetch(root) {
  const base = path.resolve(root);
  return async function fetchText(requestPath) {
    // Requests are site-root relative; strip the leading separator so they
    // resolve inside the published directory rather than the filesystem root.
    const relative = String(requestPath).replace(/^\/+/, "");
    const resolved = path.resolve(base, relative);
    // A static host cannot serve anything outside the published root.
    if (resolved !== base && !resolved.startsWith(base + path.sep)) {
      const error = new Error("Could not load " + requestPath + " (403)");
      error.status = 403;
      throw error;
    }
    let stats;
    try {
      stats = fs.lstatSync(resolved);
    } catch (cause) {
      const error = new Error("Could not load " + requestPath + " (404)");
      error.status = 404;
      throw error;
    }
    if (stats.isSymbolicLink() || !stats.isFile()) {
      const error = new Error("Could not load " + requestPath + " (404)");
      error.status = 404;
      throw error;
    }
    return fs.readFileSync(resolved, "utf8");
  };
}

async function main() {
  const target = process.argv[2];
  if (!target) {
    process.stdout.write(JSON.stringify({ accepted: false, error: "No publication directory given" }) + "\n");
    return 1;
  }
  try {
    const verdict = await acceptPublication(makeDirectoryFetch(target), "");
    process.stdout.write(JSON.stringify(verdict) + "\n");
    return 0;
  } catch (error) {
    process.stdout.write(
      JSON.stringify({ accepted: false, error: String((error && error.message) || error) }) + "\n"
    );
    return 1;
  }
}

main().then(
  (code) => process.exit(code),
  (error) => {
    process.stdout.write(JSON.stringify({ accepted: false, error: String(error) }) + "\n");
    process.exit(1);
  }
);
