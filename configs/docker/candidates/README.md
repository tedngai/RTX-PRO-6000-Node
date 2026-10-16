# Candidate configurations

Prepared or historical launch variants live here, named by profile.
Each file is a full launcher (same guards as the promoted script) so it
can be started by hand on an idle GPU 0. A candidate is not the live
config until evidence says so and it is copied into
`run_qwen27b_sglang.sh`.

Rules: one candidate runs at a time, by hand, on an idle GPU 0; results go to
`evidence/` before the candidate is considered validated. No systemd, no
restart policies, no retry loops.
