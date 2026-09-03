.PHONY: fetch-pollofpolls process-pollofpolls test-pollofpolls opinion-state backtest pop-baseline pop-baseline-benchmark pop-publication fetch-election-results process-election-results hindcast election-residuals election-layer election-layer-v2 vote-share-calibration fetch-mandate-data process-mandate-data test-mandate-allocation fetch-scb-support-voting process-scb-support-voting test-scb-support-voting process-threshold-events test-threshold-events run-scb-behavioral-diagnostic test-scb-behavioral-diagnostic run-pop-state-diagnostics test-pop-state-diagnostics run-opinion-precision-challenger test-opinion-precision-challenger check check-changed test-affected test-shard-plan test-changed test-full test-nightly-audit

PYTHON := $(shell which uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

# --- Developer loop -------------------------------------------------------
#
# Three layers, described in docs/ci-topology.md.
#
#   make check          the loop you run before pushing: compiles, the CI
#                       selector's own tests, and the tests your uncommitted
#                       and committed-vs-main changes actually affect.
#   make test-affected  just the affected tests, against a chosen BASE.
#   make test-full      everything the pull-request and full layers run, with
#                       the allocator audit at its reduced size.
#   make test-nightly-audit  the exhaustive 20,000-case allocator audit
#                       (about nine minutes), as the nightly schedule runs it.
#
# `make test-pollofpolls` still runs the entire suite unfiltered, including the
# exhaustive audit, and so still takes about twelve minutes.

# Reduced allocator parity for the developer and pull-request layers: every
# legal branch is still exercised and the dispatcher must still match the exact
# legal reference on every case. Override to raise it locally.
ADVERSARIAL_CASES ?= 700

# What to diff against when choosing affected tests.
BASE ?= main

check:
	$(PYTHON) -m compileall -q scripts tests
	$(PYTHON) -m unittest tests.test_ci_topology
	@$(MAKE) --no-print-directory test-affected

# Include uncommitted work: a test that a staged-but-uncommitted edit affects
# should run before the push, not after it.
test-affected:
	@modules="$$( { git diff --name-only $(BASE)...HEAD; git diff --name-only HEAD; git diff --name-only --cached; git ls-files --others --exclude-standard; } \
		| sort -u \
		| xargs $(PYTHON) -m scripts.ci.test_topology select --format unittest --changed )"; \
	if [ -z "$$modules" ]; then \
		echo "No Python test is affected by the current changes."; \
	else \
		echo "Running: $$modules"; \
		ELECTIONSIM_ADVERSARIAL_CASES=$(ADVERSARIAL_CASES) $(PYTHON) -m unittest $$modules; \
	fi

test-changed: test-affected

test-shard-plan:
	$(PYTHON) -m scripts.ci.test_topology plan --shards $(if $(SHARDS),$(SHARDS),4)

test-full:
	ELECTIONSIM_ADVERSARIAL_CASES=$(ADVERSARIAL_CASES) $(PYTHON) -m unittest discover -s tests -t . -v

test-nightly-audit:
	$(PYTHON) -m unittest -v tests.test_adversarial_mandates

fetch-pollofpolls:
	$(PYTHON) -m scripts.pollofpolls

process-pollofpolls:
	$(PYTHON) -m scripts.pollofpolls --offline

test-pollofpolls:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

opinion-state:
	$(PYTHON) -m scripts.pollofpolls.state $(if $(AS_OF),--as-of $(AS_OF),)

backtest:
	$(PYTHON) -m scripts.pollofpolls.backtest $(if $(MODEL),--model $(MODEL),) $(if $(START),--start $(START),) $(if $(END),--end $(END),) $(if $(HORIZONS),--horizons $(HORIZONS),) $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),)

pop-baseline:
	$(PYTHON) -m scripts.pop_baseline --origin $(ORIGIN) --horizon $(if $(HORIZON),$(HORIZON),28) $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),) $(if $(NO_SUPPORT),--disable-support-voting,)

pop-baseline-benchmark:
	$(PYTHON) -m scripts.pop_baseline.benchmark $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),) $(if $(START),--start $(START),) $(if $(END),--end $(END),) $(if $(HORIZONS),--horizons $(HORIZONS),)

pop-publication:
	$(PYTHON) -m scripts.publication_pipeline $(if $(AS_OF),--as-of $(AS_OF),) $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),)

fetch-election-results:
	$(PYTHON) -m scripts.elections.pipeline

process-election-results:
	$(PYTHON) -m scripts.elections.pipeline --offline

hindcast:
	$(PYTHON) -m scripts.hindcasts $(if $(MODELS),--models $(MODELS),) $(if $(HORIZONS),--horizons $(HORIZONS),) $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),)

election-residuals:
	$(PYTHON) -m scripts.election_residuals

election-layer:
	$(PYTHON) -m scripts.election_layer $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),)

election-layer-v2:
	$(PYTHON) -m scripts.election_layer_v2 $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),)

vote-share-calibration:
	$(PYTHON) -m scripts.vote_share_calibration $(if $(INITIAL_SAMPLES),--initial-samples $(INITIAL_SAMPLES),) $(if $(HIGH_SAMPLES),--high-samples $(HIGH_SAMPLES),) $(if $(SEED),--initial-seed $(SEED),)

fetch-mandate-data:
	$(PYTHON) -m scripts.mandates.pipeline --fetch

process-mandate-data:
	$(PYTHON) -m scripts.mandates.pipeline

test-mandate-allocation:
	$(PYTHON) -m unittest tests/test_mandate_allocation.py

fetch-geography-data:
	$(PYTHON) -m scripts.geography.pipeline --fetch

process-geography-data:
	$(PYTHON) -m scripts.geography.pipeline

test-geography:
	$(PYTHON) -m unittest tests/test_geographic_projection.py

simulate:
	$(PYTHON) -m scripts.simulator.pipeline $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),) $(if $(AS_OF),--as-of $(AS_OF),)

simulate-benchmark:
	$(PYTHON) -m scripts.simulator.pipeline --benchmark

simulate-audit:
	$(PYTHON) -m scripts.simulator.pipeline --audit-sensitivity

test-simulator:
	$(PYTHON) -m unittest tests/test_election_simulator.py

simulate-freeze-audit:
	$(PYTHON) -m scripts.simulator.freeze_audit

seat-hindcast:
	$(PYTHON) -m scripts.seat_hindcasts.pipeline $(if $(SAMPLES),--samples $(SAMPLES),) $(if $(SEED),--seed $(SEED),)

test-seat-hindcast:
	$(PYTHON) -m unittest tests/test_seat_hindcast.py

test-adversarial-mandates:
	$(PYTHON) -m unittest tests/test_adversarial_mandates.py

fetch-scb-support-voting:
	$(PYTHON) -m scripts.scb_support_voting.pipeline --fetch

process-scb-support-voting:
	$(PYTHON) -m scripts.scb_support_voting.pipeline --offline

test-scb-support-voting:
	$(PYTHON) -m unittest tests/test_scb_support_voting.py

process-threshold-events:
	$(PYTHON) -m scripts.threshold_events.pipeline

test-threshold-events:
	$(PYTHON) -m unittest tests/test_threshold_events.py

run-scb-behavioral-diagnostic:
	$(PYTHON) -m scripts.scb_behavioral_diagnostic.pipeline

test-scb-behavioral-diagnostic:
	$(PYTHON) -m unittest tests/test_scb_behavioral_diagnostic.py

run-pop-state-diagnostics:
	$(PYTHON) -m scripts.pop_state_diagnostics.pipeline

test-pop-state-diagnostics:
	$(PYTHON) -m unittest tests/test_pop_state_diagnostics.py

run-opinion-precision-challenger:
	$(PYTHON) -m scripts.opinion_precision_challenger.pipeline

test-opinion-precision-challenger:
	$(PYTHON) -m unittest tests/test_opinion_precision_challenger.py
