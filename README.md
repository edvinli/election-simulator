# Swedish Election Simulator v1.0

A leakage-safe, empirically validated probabilistic forecasting model for Swedish parliamentary elections.

---

## 🎯 Modeling Philosophy & Architecture

`ElectionSimulator v1.0` forecasts the joint probability distribution over Swedish Riksdag election outcomes by propagating uncertainty through three disentangled layers:

1. **Current Opinion State ($X_0 \sim F_{\text{state}}$)**: Multivariate ALR covariance estimated from recent poll residuals and house effects.
2. **Campaign Dynamics ($\Delta_h \sim F_{\text{movement}, h}$)**: Exact-horizon joint CLR transitions sampled from all available historical polling with sign symmetry ($\pm \Delta$).
3. **Structural Polling Error ($R \sim F_{\text{election\_error}}$)**: Empirical poll-to-election residuals (1998–2022) capturing late-decider swing and systematic pollster misses.
4. **Geography & Mandates**: 29-constituency historical projection and exact legal 310 fixed + 39 adjustment Sainte-Laguë mandate allocation.

For a detailed breakdown of our modeling philosophy, comparisons with Poll of Polls, and our empirical findings on tactical voting and state dynamics, see **[`docs/MODELING_PHILOSOPHY.md`](docs/MODELING_PHILOSOPHY.md)**.

---

## 📊 Comparison with Poll of Polls

| Component | Poll of Polls (PoP) | Swedish Election Simulator v1.0 |
| :--- | :--- | :--- |
| **Starting Point** | Daily PoP estimate | Daily PoP estimate (`M, L, C, KD, S, V, MP, SD, REST`) |
| **Current-State Uncertainty** | Not separately modeled | Multivariate ALR covariance from poll residuals |
| **Future Movement** | Random walk from current mandate period | Exact-horizon joint empirical movements from all history |
| **Directional Symmetry** | Symmetric gain/loss assumption | Empirical joint vectors with random sign symmetry ($\pm \Delta$) |
| **Election-Day Error** | No separate general layer | Empirical poll-to-election residual layer (1998–2022) |
| **Tactical Voting** | Explicit formula near 4% threshold | **None** (empirically tested and rejected) |
| **Geography** | Simpler seat conversion approach | 29-constituency historical projection |
| **Mandates** | Simulation / mandate calculation | Exact legal 310+39 modified Sainte-Laguë allocator |
| **Validation Philosophy** | Scenario exploration model | Strict out-of-sample proper scoring (CRPS, Energy Score) |

---

## 🔬 Key Empirical Findings

- **Tactical Voting Rejected ([`docs/scb_behavioral_diagnostic.md`](docs/scb_behavioral_diagnostic.md))**:
  - Across 9 elections (1991–2022), there were **0** cases where a party below 4% in final polls crossed on election day.
  - In 29 SCB PSU survey waves (2010–2026), second-choice affinity converted into cross-voting at an ordinary baseline rate ($\theta = +0.095$), with no threshold activation ($\alpha = -0.0559$, identical to 7% placebo).
- **All-History Dynamics Validated ([`docs/pop_state_dependence_evidence.md`](docs/pop_state_dependence_evidence.md))**:
  - In 3,610 rolling out-of-sample forecast cases, all-history Dynamics v2 ($\text{ES} = 1.265$) decisively outperformed state-conditioned nearest neighbors ($\text{ES} = 1.396$) and recent-period conditioning ($\text{ES} = 1.440$).
  - Raw directional momentum failed severely ($\text{ES} = 1.867$), demonstrating that sign symmetry ($\pm \Delta$) is essential.

---

## 📁 Technical Documentation

- **[Modeling Philosophy & Design Choices](docs/MODELING_PHILOSOPHY.md)**: Full design rationale, benchmark evaluation, and rejected alternatives log.
- **[Election Simulator v1.0 Specification](docs/election_simulator.md)**: End-to-end mathematical specification.
- **[PoP State-Dependence Diagnostic](docs/pop_state_dependence_evidence.md)**: Out-of-sample evaluation of state-conditioned dynamics.
- **[SCB Behavioral Threshold Diagnostic](docs/scb_behavioral_diagnostic.md)**: Regression analysis of 29 PSU survey waves.
- **[Historical Threshold Events](docs/threshold_event_evidence.md)**: Final 14-day polling vs election returns (1991–2022).
- **[Riksdag Mandate Allocation](docs/riksdag_mandate_allocation.md)**: Legal 349-seat Sainte-Laguë implementation.

---

## 🛠️ Reproduction & Testing

```bash
# Run full unit test suite (92 tests)
make test-mandate-allocation
make test-scb-support-voting
make test-threshold-events
make test-scb-behavioral-diagnostic
make test-pop-state-diagnostics

# Run data processing & diagnostic pipelines
make process-scb-support-voting
make process-threshold-events
make run-scb-behavioral-diagnostic
make run-pop-state-diagnostics
```
