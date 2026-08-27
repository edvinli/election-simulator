.PHONY: fetch-pollofpolls process-pollofpolls test-pollofpolls opinion-state backtest pop-baseline pop-baseline-benchmark pop-publication fetch-election-results process-election-results hindcast election-residuals election-layer election-layer-v2 vote-share-calibration fetch-mandate-data process-mandate-data test-mandate-allocation fetch-scb-support-voting process-scb-support-voting test-scb-support-voting process-threshold-events test-threshold-events run-scb-behavioral-diagnostic test-scb-behavioral-diagnostic run-pop-state-diagnostics test-pop-state-diagnostics run-opinion-precision-challenger test-opinion-precision-challenger

PYTHON := $(shell which uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

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
