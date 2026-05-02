# ADR-0001 — Record architecture decisions

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-05-02 |
| Deciders | @Mathews-Tom |
| Tags | process, governance |

## Context

transduce is starting from a green-field repository with a published development plan that spans five phases. The plan calls out eight open questions to be resolved as ADRs (Appendix C: manifest format, NLI model selection, signing approach, streaming-rollback UX, OTel SemConv pinning, MCP transport, multi-tenant auth, Rust extraction strategy). Without a written record of how these are decided, future contributors cannot tell whether a given pattern is intentional, a workaround, or an oversight.

The repository already enforces source-document traceability (`docs/overview.md`, `docs/pitch.md`, `docs/system-design.md`, `.docs/development-plan.md`). What is missing is a record of the *choices made within* those documents and the choices made *between* them when they conflict.

## Decision

We adopt Michael Nygard's lightweight Architecture Decision Record format. Every architecturally significant decision is captured as a numbered Markdown file under `.docs/adr/NNNN-kebab-case-title.md`, using the template at `.docs/adr/template.md`. ADRs are immutable once accepted; superseded decisions are recorded by a new ADR that links back to the prior one and updates its status field.

A decision is "architecturally significant" when it: (a) is hard to reverse, (b) affects more than one module, (c) commits the project to a third-party dependency or external standard, (d) resolves an open question listed in the development plan, or (e) departs from a documented design principle.

## Alternatives considered

- **No formal decision log** — rejected because the dev plan already enumerates eight deferred questions that need recorded outcomes; without a log, those resolutions vanish into commit messages.
- **Inline `DECISIONS.md`** — rejected because a single growing file destroys traceability and creates merge conflicts as parallel branches accumulate entries.
- **MADR (Markdown Any Decision Records) full template** — rejected as over-structured for this stage; we can adopt MADR Sections later if the lightweight Nygard format proves insufficient. MADR is a strict superset, so migration is mechanical.
- **GitHub Issues with a `decision` label** — rejected because issues are mutable, hard to grep from a checkout, and require external tooling to read.

## Consequences

### Positive

- Future contributors can grep `.docs/adr/` to understand why a given pattern exists without archaeology through commit history.
- Open questions in the development plan get bound to concrete records rather than slipping between phases.
- ADRs serve as the canonical place to document deviations from `docs/system-design.md`, satisfying the dev plan's "deviations require an ADR" requirement (§Verification Subsystem ordering).
- Reviewers have a fixed location to ask "is there an ADR for this?" during PR review.

### Negative

- Authors must remember to write an ADR when a significant decision is made; this is a process discipline, not an enforced gate.
- Numbering is sequential and global, which introduces a soft-coordination point on parallel branches. Conflicts are resolved by renumbering during rebase.

### Neutral

- ADR files are tracked under `.docs/adr/` despite the broader `.docs/` directory being gitignored. The gitignore exception is committed in the foundation bootstrap branch.

## Compliance and verification

- The dev plan calls out eight open questions tagged Q-01 through Q-08 in Appendix C. Each must land as an ADR before its phase exits.
- PR reviewers check the diff for new abstractions, third-party dependencies, or deviations from `docs/system-design.md`; absence of an ADR in such PRs is a review block.
- ADR titles use kebab-case to match repository conventions and remain greppable.

## References

- Michael Nygard, "Documenting Architecture Decisions" (<https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>)
- MADR — Markdown Any Decision Records (<https://adr.github.io/madr/>)
- `.docs/development-plan.md` Appendix C — Open questions deferred to ADRs
- `docs/system-design.md` — Design principles
