"use strict";

// REFERENCE publication contract validator — NOT the production consumer.
//
// This is an independent Node reimplementation of the acceptance rules that
// edvinli.github.io/assets/js/election-simulator.js applies. It exists so the
// exporter has a fast, dependency-free, CI-portable contract check that runs
// even when the website repository is not checked out.
//
// It is NOT extracted from, shared with, or byte-identical to the production
// file. Two independently maintained validators can drift, so this file alone
// does NOT close the exporter/browser contract gap. What closes that gap is
// the ACTUAL browser consumer test, which evaluates the deployed website
// source verbatim:
//
//   scripts/static_exporter/contract/actual_consumer_harness.js
//   tests/test_actual_browser_consumer.py
//
// Drift between this reference and production is caught by
// ReferenceValidatorDriftTests in tests/test_reference_publication_contract.py.
//
// The transport is injected as a `fetchText(path)` function that either
// resolves with the file's text or rejects with an error carrying `.status`,
// mirroring how the browser distinguishes a missing pointer (404, a valid
// legacy publication) from any other failure (a hard error).

const PUBLICATION_FILES = [
  "forecast.json",
  "parties.json",
  "seats.json",
  "groups.json",
  "calibration.json",
  "metadata.json",
  "manifest.json",
];
const PUBLICATION_CONTRACTS = PUBLICATION_FILES.slice(0, 6);
const POINTER_PATH_PATTERN = /^versions\/[A-Za-z0-9_-]+$/;

async function sha256Hex(text) {
  const subtle = globalThis.crypto && globalThis.crypto.subtle;
  if (!subtle || typeof TextEncoder === "undefined") {
    throw new Error("Browser cannot verify the publication manifest hash");
  }
  const buffer = await subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.prototype.map
    .call(new Uint8Array(buffer), (value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function joinPath(root, name) {
  return root.replace(/\/$/, "") + "/" + name;
}

async function getText(fetchText, root, name) {
  return fetchText(joinPath(root, name));
}

async function getJson(fetchText, root, name) {
  const text = await getText(fetchText, root, name);
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error("Could not parse " + name + " as JSON");
  }
}

async function loadContracts(fetchText, root, pointer) {
  const files = [];
  for (const name of PUBLICATION_CONTRACTS) {
    files.push(await getJson(fetchText, root, name));
  }
  if (!pointer) {
    const manifest = await getJson(fetchText, root, "manifest.json");
    return { files: files.concat([manifest]), manifest_sha256: null };
  }
  // Hash the exact manifest bytes addressed by current.json.  Parsing and
  // re-serializing JSON would hide whitespace/content tampering.
  const manifestText = await getText(fetchText, root, "manifest.json");
  let manifest;
  try {
    manifest = JSON.parse(manifestText);
  } catch (error) {
    throw new Error("Publication manifest is not valid JSON");
  }
  const manifestHash = await sha256Hex(manifestText);
  return { files: files.concat([manifest]), manifest_sha256: manifestHash };
}

function isValidPointer(pointer) {
  return Boolean(
    pointer &&
      pointer.publication_state === "COMPLETE" &&
      typeof pointer.publication_generation === "string" &&
      typeof pointer.manifest_sha256 === "string" &&
      typeof pointer.path === "string" &&
      pointer.path === "versions/" + pointer.publication_generation &&
      POINTER_PATH_PATTERN.test(pointer.path)
  );
}

async function loadPublication(fetchText, base) {
  // The pointer is the canonical web contract.  The 404 fallback keeps older
  // static publications readable while they are migrated, but a malformed
  // existing pointer is a hard error and is never bypassed.
  let pointer;
  try {
    pointer = await getJson(fetchText, base, "current.json");
  } catch (error) {
    if (error.status !== 404) throw error;
    const loaded = await loadContracts(fetchText, base, null);
    loaded.pointer = null;
    return loaded;
  }
  if (!isValidPointer(pointer)) {
    throw new Error("Current publication pointer is invalid");
  }
  const loaded = await loadContracts(fetchText, joinPath(base, pointer.path), pointer);
  loaded.pointer = pointer;
  return loaded;
}

function validatePublicationBundle(data, pointer, manifestHash) {
  const manifest = data[6] || {};
  if (manifest.publication_state && manifest.publication_state !== "COMPLETE") {
    throw new Error("Publication is not marked complete");
  }
  if (pointer && manifestHash !== pointer.manifest_sha256) {
    throw new Error("Current publication pointer hash does not match the manifest");
  }
  if (pointer && (manifest.source_worktree_clean !== true || !data[5] || data[5].source_worktree_clean !== true)) {
    throw new Error("Certified publication has dirty or incomplete source provenance");
  }
  const expected = manifest.deterministic_payload_sha256;
  const identities = data
    .slice(0, 6)
    .map((value) => value && value.deterministic_payload_sha256)
    .filter((value) => value);
  if (pointer && (!expected || identities.length !== 6 || identities.some((value) => value !== expected))) {
    throw new Error("Publication files do not all link the deterministic simulation payload");
  }
  if (!pointer && expected && identities.length > 1 && identities.some((value) => value !== expected)) {
    throw new Error("Publication files belong to different simulation payloads");
  }
  if (pointer && (manifest.publication_generation !== pointer.publication_generation || manifest.publication_state !== "COMPLETE")) {
    throw new Error("Publication pointer and manifest do not agree");
  }
}

function requireRepresentativeAllocation(seats, order) {
  // Under a pointer the seat contract must carry a legal joint allocation.
  // The legacy fallback path is allowed to synthesise one instead.
  const representative = seats.representative_allocation;
  if (representative && representative.seats) {
    const allocation = {};
    let total = 0;
    const valid = order.every((name) => {
      const value = representative.seats[name];
      if (!Number.isInteger(value) || value < 0) return false;
      allocation[name] = value;
      total += value;
      return true;
    });
    if (valid && total === 349 && representative.total_seats === 349) {
      return { allocation, source: "representative_joint_simulation_draw", total };
    }
  }
  throw new Error("Published seat contract has no valid representative joint allocation");
}

/**
 * Load and fully accept a publication, exactly as the website consumer would.
 * Resolves with a verdict object; rejects with the browser-visible error.
 */
async function acceptPublication(fetchText, base) {
  const publication = await loadPublication(fetchText, base);
  const data = publication.files;
  validatePublicationBundle(data, publication.pointer, publication.manifest_sha256);
  const seats = data[2];
  const order = seats.party_order || Object.keys(seats.seat_summary);
  let allocation = null;
  if (publication.pointer) {
    allocation = requireRepresentativeAllocation(seats, order);
  }
  const metadata = data[5];
  const manifest = data[6];
  const certified = metadata.source_worktree_clean === true && manifest && manifest.source_worktree_clean === true;
  return {
    accepted: true,
    mode: publication.pointer ? "pointer" : "legacy_flat_fallback",
    generation: publication.pointer ? publication.pointer.publication_generation : null,
    manifest_sha256: publication.manifest_sha256,
    certified,
    status_text: certified ? "Certified forecast loaded." : "Forecast loaded, but it is not certified.",
    seat_allocation_source: allocation ? allocation.source : "legacy_normalized_marginal_medians",
    seat_total: allocation ? allocation.total : null,
    schema_version: manifest ? manifest.schema_version : null,
    source_repository: manifest ? manifest.source_repository || null : null,
    deterministic_payload_sha256: manifest ? manifest.deterministic_payload_sha256 : null,
  };
}

module.exports = {
  PUBLICATION_FILES,
  PUBLICATION_CONTRACTS,
  acceptPublication,
  loadPublication,
  validatePublicationBundle,
  requireRepresentativeAllocation,
  isValidPointer,
  sha256Hex,
};
