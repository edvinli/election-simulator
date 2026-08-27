"""Construct canonical party-election threshold episode dataset and consensus details.
"""
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from scripts.threshold_events.config import (
    KNOWN_NAMED_MINOR_PARTIES,
    PARLIAMENTARY_PARTIES,
    POLLS_FILE,
    PROCESSED_DATA_DIR,
    TARGET_ELECTIONS,
    assign_threshold_band,
    grade_episode_quality,
)
from scripts.threshold_events.consensus import (
    ElectionConsensusResult,
    build_final_polling_consensus,
)
from scripts.threshold_events.election_results import (
    OfficialPartyResult,
    load_all_official_election_results,
)


def _passes_legal_4pct_threshold(votes: int, valid_votes_total: int) -> bool:
    """Evaluate the legal 4% threshold from exact vote counts.

    A non-positive valid-vote total is not a meaningful election result and
    cannot be classified from a rounded percentage.  Fail closed rather than
    allowing an approximate display value to decide legal eligibility.
    """
    if valid_votes_total <= 0:
        raise ValueError(
            "Cannot classify the legal 4% threshold when valid_votes_total "
            f"is non-positive: {valid_votes_total!r}"
        )
    return bool(25 * votes >= valid_votes_total)


def build_party_election_episodes(
    polls_df: Optional[pd.DataFrame] = None,
    official_results: Optional[Dict[int, Dict[str, OfficialPartyResult]]] = None,
    window_days: int = 14,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build canonical party-election threshold episodes table and consensus details audit table."""
    if polls_df is None:
        polls_df = pd.read_csv(POLLS_FILE)
    if official_results is None:
        official_results = load_all_official_election_results()
        
    episodes: List[Dict[str, Any]] = []
    consensus_details: List[Dict[str, Any]] = []
    
    for year, elec_info in sorted(TARGET_ELECTIONS.items()):
        elec_date = elec_info.election_date
        year_results = official_results.get(year, {})
        
        # Build consensus
        consensus_res = build_final_polling_consensus(
            election_date=elec_date,
            election_year=year,
            polls_df=polls_df,
            window_days=window_days,
        )
        
        if consensus_res is None:
            # Document excluded election episodes (e.g. 1991, 1998 in canonical 14d)
            for party, opt_res in year_results.items():
                if party in ["OTHER", "REST", "OGILTIGA", "VALSKOLKARE"]:
                    continue
                votes = opt_res.votes
                valid_tot = opt_res.valid_votes_total
                actual_share = opt_res.vote_share_pct
                passed = _passes_legal_4pct_threshold(votes, valid_tot)
                
                episodes.append({
                    "election_year": year,
                    "election_date": elec_date.isoformat(),
                    "party": party,
                    "party_raw": opt_res.party_raw,
                    "final_poll_consensus_pct": np.nan,
                    "actual_result_pct": round(actual_share, 4),
                    "votes": votes,
                    "valid_votes_total": valid_tot,
                    "residual_pp": np.nan,
                    "distance_from_4_pp": np.nan,
                    "passed_4pct": passed,
                    "forecast_side": None,
                    "actual_side": passed,
                    "quadrant": "unpolled_excluded",
                    "threshold_crossing_distance_pp": np.nan,
                    "threshold_band": "EXCLUDED",
                    "party_eligible_poll_count": 0,
                    "party_contributing_poll_count": 0,
                    "party_pollster_count": 0,
                    "party_sample_size_coverage": 0.0,
                    "election_poll_count": 0,
                    "election_pollster_count": 0,
                    "consensus_window_days": window_days,
                    "metadata_quality": "INDETERMINATE" if elec_info.canonical_inclusion_status == "EXCLUDE_MISSING_DATES" else "NO_POLLS",
                    "episode_quality": "EXCLUDE",
                    "source_notes": elec_info.notes,
                })
            continue
            
        # Record consensus details audit records
        for cp in consensus_res.contributing_records:
            for p_code, supp in cp.party_support.items():
                consensus_details.append({
                    "election_year": cp.election_year,
                    "election_date": cp.election_date.isoformat(),
                    "consensus_window_days": cp.window_days,
                    "poll_id": cp.poll_id,
                    "pollster": cp.pollster,
                    "pollster_original": cp.pollster_original,
                    "interview_start": cp.interview_start.isoformat() if cp.interview_start else None,
                    "interview_end": cp.interview_end.isoformat(),
                    "publication_date": cp.publication_date.isoformat(),
                    "sample_size": cp.sample_size,
                    "sample_size_missing": cp.sample_size_missing,
                    "weight": cp.weight,
                    "party": p_code,
                    "support_pct": supp,
                })
                
        # Build party-level episodes
        eligible_parties = set(consensus_res.party_consensus.keys()) | set(year_results.keys())
        
        for party in sorted(eligible_parties):
            if party in ["OTHER", "REST", "OGILTIGA", "VALSKOLKARE", "UNCERTAIN"]:
                continue
                
            is_parl = party in PARLIAMENTARY_PARTIES
            has_polling = party in consensus_res.party_consensus
            has_result = party in year_results
            
            if not is_parl and not (has_polling and has_result):
                continue
                
            p_cons_summary = consensus_res.party_consensus.get(party)
            p_actual_res = year_results.get(party)
            
            if p_actual_res is None:
                continue
                
            votes = p_actual_res.votes
            valid_tot = p_actual_res.valid_votes_total
            actual_share = p_actual_res.vote_share_pct
            
            # Exact threshold legal condition: 25 * V_p >= V_valid
            passed_4pct = _passes_legal_4pct_threshold(votes, valid_tot)
            # Keep the side used for threshold quadrants aligned with the
            # legal 4% rule.  A displayed percentage is rounded, whereas the
            # law is evaluated from the exact vote-count cross-product.
            actual_side = passed_4pct
            
            if p_cons_summary is not None:
                cons_val = p_cons_summary.consensus_pct
                res_pp = round(actual_share - cons_val, 4)
                dist_4_pp = round(cons_val - 4.0, 4)
                forecast_side = bool(cons_val >= 4.0)
                band = assign_threshold_band(cons_val)
                p_elig_poll_cnt = p_cons_summary.party_eligible_poll_count
                p_contrib_poll_cnt = p_cons_summary.party_contributing_poll_count
                p_house_cnt = p_cons_summary.party_pollster_count
                p_cov_n = p_cons_summary.party_sample_size_coverage
                
                # Determine 4-quadrant diagnostic
                if not forecast_side and not actual_side:
                    quadrant = "below_to_below"
                elif not forecast_side and actual_side:
                    quadrant = "below_to_above"
                elif forecast_side and not actual_side:
                    quadrant = "above_to_below"
                else:
                    quadrant = "above_to_above"
                    
                # Threshold crossing distance: |actual - 4| - |consensus - 4|
                # Positive means election day moved the party further away from 4%; negative means moved toward 4%
                crossing_dist_pp = round(abs(actual_share - 4.0) - abs(cons_val - 4.0), 4)
                
                meta_quality = "COMPLETE"
                quality_grade = grade_episode_quality(
                    party_pollster_count=p_house_cnt,
                    party_eligible_poll_count=p_elig_poll_cnt,
                    sample_size_coverage=p_cov_n,
                    metadata_complete=True,
                )
            else:
                cons_val = np.nan
                res_pp = np.nan
                dist_4_pp = np.nan
                forecast_side = None
                band = "UNPOLLED"
                p_elig_poll_cnt = 0
                p_contrib_poll_cnt = 0
                p_house_cnt = 0
                p_cov_n = 0.0
                quadrant = "unpolled"
                crossing_dist_pp = np.nan
                meta_quality = "NO_PARTY_POLLS"
                quality_grade = "EXCLUDE"
                
            episodes.append({
                "election_year": year,
                "election_date": elec_date.isoformat(),
                "party": party,
                "party_raw": p_actual_res.party_raw,
                "final_poll_consensus_pct": cons_val,
                "actual_result_pct": round(actual_share, 4),
                "votes": votes,
                "valid_votes_total": valid_tot,
                "residual_pp": res_pp,
                "distance_from_4_pp": dist_4_pp,
                "passed_4pct": passed_4pct,
                "forecast_side": forecast_side,
                "actual_side": actual_side,
                "quadrant": quadrant,
                "threshold_crossing_distance_pp": crossing_dist_pp,
                "threshold_band": band,
                "party_eligible_poll_count": p_elig_poll_cnt,
                "party_contributing_poll_count": p_contrib_poll_cnt,
                "party_pollster_count": p_house_cnt,
                "party_sample_size_coverage": p_cov_n,
                "election_poll_count": consensus_res.total_eligible_polls_in_window,
                "election_pollster_count": consensus_res.total_retained_pollsters,
                "consensus_window_days": window_days,
                "metadata_quality": meta_quality,
                "episode_quality": quality_grade,
                "source_notes": elec_info.notes,
            })
            
    df_episodes = pd.DataFrame(episodes)
    df_details = pd.DataFrame(consensus_details)
    
    df_episodes.sort_values(by=["election_year", "party"], inplace=True)
    if not df_details.empty:
        df_details.sort_values(by=["election_year", "pollster", "party"], inplace=True)
        
    return df_episodes, df_details


def generate_and_save_canonical_datasets(
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full episode and consensus pipeline and save CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Building canonical party-election threshold episodes (14-day window)...")
    df_episodes, df_details = build_party_election_episodes(window_days=14)
    
    episodes_file = output_dir / "party_election_threshold_events.csv"
    details_file = output_dir / "election_consensus_details.csv"
    
    df_episodes.to_csv(episodes_file, index=False, encoding="utf-8")
    df_details.to_csv(details_file, index=False, encoding="utf-8")
    
    print(f"  -> Saved {len(df_episodes)} episodes to {episodes_file}")
    print(f"  -> Saved {len(df_details)} consensus details to {details_file}")
    
    return df_episodes, df_details


if __name__ == "__main__":
    generate_and_save_canonical_datasets()
