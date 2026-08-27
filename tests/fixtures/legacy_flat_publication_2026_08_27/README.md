# Legacy flat publication (2026-08-27)

Byte-for-byte copy of the pre-extraction flat publication served from
`edvinli.github.io/files/election-simulator/`, captured at website commit
`d83ae26c5c8cc625b163cba005aabb1110724018`.

It exists solely as a **read-only regression fixture** so the legacy
compatibility path stays covered after the canonical contract moves to
`current.json` + `versions/<generation>/`. Its properties are intentional:

- publication schema `1.0`, so no `source_repository` (it means
  `edvinli/edvinli.github.io`);
- `manifest.json` carries no `publication_state` / `publication_generation`;
- only `forecast.json` and `metadata.json` link `deterministic_payload_sha256`;
- `seats.json` has no `representative_allocation`;
- `source_worktree_clean` is `false`, so it is permanently **uncertified**.

Never edit these files, never regenerate them, and never copy them into
`files/election-simulator/versions/`.
