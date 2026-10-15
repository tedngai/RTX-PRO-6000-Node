# Candidate configurations

Prepared or historical launch variants live here, named by profile
(e.g. `qwen27b-no-spec.sh`). This directory is intentionally empty until
experiments from `docs/NEXT_EXPERIMENT.md` produce results.

Rules: one candidate runs at a time, by hand, on an idle GPU 0; results go to
`evidence/` before the candidate is considered validated. No systemd, no
restart policies, no retry loops.
