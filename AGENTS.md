# AGENTS.md

## Project Rules

Before editing code, documentation, comments, generated prose, or template
structure in this repository, read and follow:

- `.cursor/rules/00-lightweight-research-coding.mdc`
- `.cursor/rules/01-matlab-code-style.mdc`
- `.cursor/rules/02-local-optimizer-first.mdc`
- `.cursor/rules/03-public-note-and-paper-skill-bridge.mdc`
- `.cursor/rules/04-case-figure-and-metric-plan.mdc`
- `.cursor/rules/05-project-execution.mdc`
- `.cursor/rules/lexicon.md`

Treat these files as repository policy for Codex, Gemini CLI, Antigravity, and
other coding-agent work in this template. Antigravity compatibility is routed
through `GEMINI.md`.

For PowerLit writing/review handoff, also read
`.cursor/contracts/project-template-handoff.schema.yaml` and the active
`02_PAPER/config/paper-profile.yaml` (or its example). Keep reasoning policy in
the Skill and persistent state, MATLAB execution, manifests, and audit artifacts
in this template.

## Sync Policy

Paper sources follow rule 03: Markdown drafts live in `02_PAPER/manuscript/`;
formal LaTeX sources and publication figures live in `02_PAPER/latex/`.
Read `02_PAPER/formatting_rules.md` for the default IEEE PES journal template
and manuscript formatting conventions.

Commit and push `.cursor/`, `AGENTS.md`, `GEMINI.md`, and `CLAUDE.md` changes
together with template changes. Do not treat agent rule files as editor-only
local state.

## Validation

This template intentionally ships without a default executable test runner.
Do not launch MATLAB automatically when copying or deploying the template.
Run only the project-specific checks explicitly selected by the author.

## Current project override

Read `.cursor/rules/05-project-execution.mdc` before execution. Since
2026-10-26 the repository holds two papers: `paper_standardTMM/` (standard TMM
versus AQGMM) and `paper_groupedT/` (grouped t mixture versus AQGMM and
standard TMM). Shared code, data and raw results live in `shared/`; pre-split
material lives in `archive/`. The current numerical workflow is the Python
validation code in `shared/tests/python/`; the user authorized local and cloud
execution on 2026-10-21. Fitting voltages come from the MATLAB fast mapping
called by `shared/tests/python/fast_voltage_map.py`; the MC method
(per-scenario AC power flow) is only the exact reference. Results are grouped
into `heavytail_injection/` and `actual_output/` case groups. Do not modify or
push the original project `wuhanichina/AQPPF_TMM`; push only to this
repository. Use power-system probabilistic power-flow wording as required by
`.cursor/rules/lexicon.md`.
