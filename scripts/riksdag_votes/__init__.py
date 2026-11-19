"""Riksdag roll-call voting tracker: data pipeline (phase 0).

Independent of the election forecast: nothing here imports from, or writes
to, the simulator, certification, publication or rendering code.

Modules: ``fetch`` (cached raw downloads + manifest), ``ingest`` (parsing,
validation and bulk/API reconciliation), ``committees``, ``party_lines``,
``agreement``, ``ideal_points``, ``summarize`` (JSON payloads) and ``build``
(CLI). Full write-up: ``docs/riksdag_votes.md``.

Rules (user-approved):

1. Vote selection. Every vote keeps avser, votering, punkt, beteckning,
   committee and date. The default analysis set (agreement, party lines
   statistics, map, member statistics) is ``votering == "huvud"`` and
   ``avser == "sakfrågan"``. All votes stay in ``votes.json`` and in the
   per-vote party lines for the chamber overlay.
2. Avstår is a stance. A party's line is the plurality among its members
   voting Ja/Nej/Avstår (Frånvarande excluded); a tie gives line null with a
   tie flag. Agreement between two parties = identical lines among
   {Ja, Nej, Avstår} ("three-way", primary); secondary "Ja/Nej only" over
   votes where both lines are Ja or Nej. Both with n per cell.
3. Map without centre-pull. Ja=+1, Avstår=0, Nej=-1, Frånvarande=NaN, never
   imputed. Near-unanimous votes (minority < 2.5 % of Ja+Nej) are dropped.
   PCA (centred per vote) on the party x vote line matrix gives the geometry;
   each member is placed by least squares on the votes they cast,
   s = (W_O^T W_O)^-1 W_O^T (v_O - mu_O), with at least ``MIN_MP_MAP_VOTES``
   cast map votes. Axes are not called left/right.
4. Independents (parti "-") have no party line and are outside the party
   matrix but keep per-member data. A vote against the party line is a cast
   Ja/Nej/Avstår differing from the member's party's non-null line; isolated
   cases can be feltryckningar, which the record keeps as cast.
5. No truncation: exactly 349 rows per vote (benches 1..349), no duplicate
   (votering_id, intressent_id), totals equal to the API register -- or the
   build fails. People are tracked by intressent_id; each vote keeps the bench
   occupant (member or substitute).
6. Attendance: participation = share of seated analysis-set votes with
   rost != Frånvarande; no rankings, because Frånvarande includes kvittning.
"""
