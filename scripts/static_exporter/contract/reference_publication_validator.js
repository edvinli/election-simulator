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
// 1.4 is 1.3 plus metadata only (election_noise_law / election_noise_candidate);
// groups.json keeps the 1.3 coalition contract, histograms included.
const SUPPORTED_SCHEMA_VERSIONS = new Set(["1.0", "1.1", "1.2", "1.3", "1.4"]);
const HISTOGRAM_SCHEMA_VERSIONS = new Set(["1.3", "1.4"]);
const COALITION_BUILDER_SCHEMA_VERSIONS = new Set(["1.2", "1.3", "1.4"]);
const COALITION_PARTY_ORDER = ["M", "L", "C", "KD", "S", "V", "MP", "SD"];
const COALITION_SUMMARY_FIELDS = [
  "mask",
  "parties",
  "mean_seats",
  "median_seats",
  "p05_seats",
  "p10_seats",
  "p25_seats",
  "p75_seats",
  "p90_seats",
  "p95_seats",
  "prob_majority",
];
const COALITION_HISTOGRAM_FIELDS = ["min_seats", "counts"];
const COALITION_ENTRY_FIELDS_WITH_HISTOGRAM = COALITION_SUMMARY_FIELDS.concat(["seat_histogram"]);
const MAJORITY_THRESHOLD = 175;
const CHAMBER_SEATS = 349;

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isInteger(value) {
  return Number.isInteger(value);
}

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

function histogramValueAtOrderIndex(minimum, counts, index) {
  let remaining = index;
  for (let offset = 0; offset < counts.length; offset += 1) {
    if (remaining < counts[offset]) return minimum + offset;
    remaining -= counts[offset];
  }
  throw new Error("Histogram order-statistic index is outside its support");
}

function histogramQuantile(minimum, counts, quantile) {
  const total = counts.reduce((sum, count) => sum + count, 0);
  if (!Number.isSafeInteger(total) || total <= 0) {
    throw new Error("Cannot calculate a quantile from an empty histogram");
  }
  const position = (total - 1) * quantile;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const gamma = position - lowerIndex;
  const lower = histogramValueAtOrderIndex(minimum, counts, lowerIndex);
  const upper = histogramValueAtOrderIndex(minimum, counts, upperIndex);
  // Match NumPy's _lerp implementation, including its upper-end expression
  // for gamma >= 0.5.  The alternate expression avoids a one-ULP downward
  // drift before the existing integer truncation convention is applied.
  const interpolated = gamma >= 0.5
    ? upper - (upper - lower) * (1 - gamma)
    : lower + (upper - lower) * gamma;
  return Math.trunc(interpolated);
}

function histogramCount(histogram, seats) {
  const offset = seats - histogram.min_seats;
  if (offset < 0 || offset >= histogram.counts.length) return 0;
  return histogram.counts[offset];
}

function validateCoalitionHistogram(value, expectedTotal, entry, key) {
  if (!isObject(value)) throw new Error("coalition " + key + " has an invalid seat_histogram");
  const histogramKeys = Object.keys(value);
  if (histogramKeys.length !== COALITION_HISTOGRAM_FIELDS.length ||
      !COALITION_HISTOGRAM_FIELDS.every((field, index) => histogramKeys[index] === field)) {
    throw new Error("coalition " + key + " has an invalid seat_histogram");
  }
  const minimum = value.min_seats;
  const counts = value.counts;
  if (!isInteger(minimum) || minimum < 0 || minimum > CHAMBER_SEATS ||
      !Array.isArray(counts) || counts.length < 1) {
    throw new Error("coalition " + key + " seat_histogram support is invalid");
  }
  if (!counts.every((count) => isInteger(count) && count >= 0)) {
    throw new Error("coalition " + key + " seat_histogram counts must be non-negative integers");
  }
  const maximum = minimum + counts.length - 1;
  if (maximum > CHAMBER_SEATS || (counts.length > 1 && (counts[0] === 0 || counts[counts.length - 1] === 0))) {
    throw new Error("coalition " + key + " seat_histogram support is invalid");
  }
  const total = counts.reduce((sum, count) => sum + count, 0);
  if (!Number.isSafeInteger(total) || total !== expectedTotal) {
    throw new Error("coalition " + key + " seat_histogram counts do not sum to samples");
  }
  const mean = counts.reduce((sum, count, offset) => sum + (minimum + offset) * count, 0) / total;
  if (!isFiniteNumber(entry.mean_seats) || Math.abs(mean - entry.mean_seats) > 1e-12) {
    throw new Error("coalition " + key + " mean_seats disagrees with seat_histogram");
  }
  const quantileFields = [
    ["p05_seats", 0.05],
    ["p10_seats", 0.10],
    ["p25_seats", 0.25],
    ["median_seats", 0.50],
    ["p75_seats", 0.75],
    ["p90_seats", 0.90],
    ["p95_seats", 0.95],
  ];
  quantileFields.forEach(([field, quantile]) => {
    if (histogramQuantile(minimum, counts, quantile) !== entry[field]) {
      throw new Error("coalition " + key + " " + field + " disagrees with seat_histogram");
    }
  });
  const majorityCount = counts.reduce(
    (sum, count, offset) => sum + (minimum + offset >= MAJORITY_THRESHOLD ? count : 0),
    0,
  );
  if (Math.abs(majorityCount / total - entry.prob_majority) > 1e-12) {
    throw new Error("coalition " + key + " prob_majority disagrees with seat_histogram");
  }
  if (key === "0" && (minimum !== 0 || counts.length !== 1 || counts[0] !== total)) {
    throw new Error("Empty coalition must contain only zero-seat draws");
  }
  if (key === "255" && (minimum !== CHAMBER_SEATS || counts.length !== 1 || counts[0] !== total)) {
    throw new Error("Full coalition must contain only 349-seat draws");
  }
  return { min_seats: minimum, counts };
}

function validateCoalitionBuilder(builder, schemaVersion, expectedTotal) {
  if (!isObject(builder)) {
    throw new Error("Schema " + schemaVersion + " groups.json must include coalition_builder");
  }
  const builderKeys = Object.keys(builder);
  const expectedBuilderKeys = ["party_order", "encoding", "majority_threshold", "coalitions"];
  if (builderKeys.length !== expectedBuilderKeys.length ||
      !expectedBuilderKeys.every((field, index) => builderKeys[index] === field) ||
      builder.encoding !== "bitmask" || builder.majority_threshold !== MAJORITY_THRESHOLD ||
      !Array.isArray(builder.party_order) || builder.party_order.length !== COALITION_PARTY_ORDER.length ||
      !builder.party_order.every((party, index) => party === COALITION_PARTY_ORDER[index])) {
    throw new Error("coalition_builder metadata is invalid");
  }

  const coalitions = builder.coalitions;
  const coalitionKeys = isObject(coalitions) ? Object.keys(coalitions) : [];
  if (coalitionKeys.length !== 256 || !coalitionKeys.every((key, index) => key === String(index))) {
    throw new Error("coalition_builder must contain keys \"0\" through \"255\" in order");
  }
  const histogramRequired = HISTOGRAM_SCHEMA_VERSIONS.has(schemaVersion);
  if (histogramRequired && (!isInteger(expectedTotal) || expectedTotal <= 0)) {
    throw new Error("Schema 1.3 coalition histograms require a positive sample count");
  }
  const expectedEntryFields = histogramRequired
    ? COALITION_ENTRY_FIELDS_WITH_HISTOGRAM
    : COALITION_SUMMARY_FIELDS;
  const histograms = {};
  for (let mask = 0; mask < 256; mask += 1) {
    const key = String(mask);
    const entry = coalitions[key];
    if (!isObject(entry) || Object.keys(entry).length !== expectedEntryFields.length ||
        !expectedEntryFields.every((field, index) => Object.keys(entry)[index] === field) ||
        entry.mask !== mask || !Array.isArray(entry.parties)) {
      throw new Error("coalition " + key + " has unexpected or unordered fields");
    }
    const expectedParties = builder.party_order.filter((party, index) => (mask & (1 << index)) !== 0);
    if (entry.parties.length !== expectedParties.length ||
        !entry.parties.every((party, index) => party === expectedParties[index])) {
      throw new Error("coalition " + key + " has incorrect party membership");
    }
    const metricFields = COALITION_SUMMARY_FIELDS.slice(2);
    metricFields.forEach((field) => {
      const value = entry[field];
      const valid = field === "mean_seats" || field === "prob_majority"
        ? isFiniteNumber(value)
        : isInteger(value);
      if (!valid || (field === "prob_majority"
        ? value < 0 || value > 1
        : value < 0 || value > CHAMBER_SEATS)) {
        throw new Error("coalition " + key + " has an invalid " + field);
      }
    });
    const quantiles = [entry.p05_seats, entry.p10_seats, entry.p25_seats,
      entry.median_seats, entry.p75_seats, entry.p90_seats, entry.p95_seats];
    if (!quantiles.every((value, index) => index === 0 || value >= quantiles[index - 1])) {
      throw new Error("coalition " + key + " quantiles are not monotone");
    }
    if (mask === 0 && (entry.mean_seats !== 0 || quantiles.some((value) => value !== 0) || entry.prob_majority !== 0)) {
      throw new Error("Empty coalition must have zero seats and zero majority probability");
    }
    if (mask === 255 && (entry.mean_seats !== CHAMBER_SEATS || quantiles.some((value) => value !== CHAMBER_SEATS) || entry.prob_majority !== 1)) {
      throw new Error("Full coalition must have 349 seats and certainty of majority");
    }
    if (histogramRequired) histograms[mask] = validateCoalitionHistogram(entry.seat_histogram, expectedTotal, entry, key);
  }

  if (histogramRequired) {
    for (let mask = 0; mask < 256; mask += 1) {
      const complement = 255 ^ mask;
      if (mask > complement) continue;
      for (let seats = 0; seats <= CHAMBER_SEATS; seats += 1) {
        if (histogramCount(histograms[mask], seats) !== histogramCount(histograms[complement], CHAMBER_SEATS - seats)) {
          throw new Error("coalition " + mask + " and complement " + complement + " violate the 349-seat identity");
        }
      }
    }
  }
}

function validateCoalitionPublication(forecast, groups) {
  if (!isObject(groups) || !SUPPORTED_SCHEMA_VERSIONS.has(groups.schema_version)) {
    throw new Error("groups.json has unsupported schema version");
  }
  if (groups.majority_threshold !== MAJORITY_THRESHOLD) {
    throw new Error("Group majority threshold must be 175");
  }
  if (COALITION_BUILDER_SCHEMA_VERSIONS.has(groups.schema_version)) {
    validateCoalitionBuilder(groups.coalition_builder, groups.schema_version,
      HISTOGRAM_SCHEMA_VERSIONS.has(groups.schema_version)
        ? forecast && forecast.total_samples : null);
  } else if (Object.prototype.hasOwnProperty.call(groups, "coalition_builder")) {
    throw new Error("coalition_builder is only valid in schema 1.2 or 1.3 groups.json");
  }
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
  validateCoalitionPublication(data[0], data[3]);
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
