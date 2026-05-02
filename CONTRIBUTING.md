# Contributing to transduce

Thanks for your interest. transduce is a Unix-philosophy primitive for verifiable text transformation. Contributions are welcome across modes, scorers, backend adapters, fixtures, and documentation.

## Code of conduct

Participation in this project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report violations to the maintainers via the contact channel listed in [`SECURITY.md`](SECURITY.md).

## Reporting issues

- **Bugs:** open a GitHub issue with a minimal reproduction (config, request payload, observed vs expected behavior).
- **Security vulnerabilities:** do **not** open a public issue. Follow the disclosure process in [`SECURITY.md`](SECURITY.md).
- **Feature requests:** describe the use case before the proposed implementation.

## Development setup

Requirements:

- Python 3.12 or 3.13
- [`uv`](https://docs.astral.sh/uv/) — the only supported package manager
- Docker (for integration and end-to-end tests)
- [`just`](https://github.com/casey/just) — the project task runner

```bash
git clone https://github.com/Mathews-Tom/transduce.git
cd transduce
uv sync
uv run pre-commit install
just test
```

## Branching and commits

- Branch off `main`. Branch names use kebab-case and a deliverable-tied prefix: `feat/dejargon-mode`, `fix/cosine-empty-input`, `docs/adr-0002-nli-model`.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <subject>`.
- Subject ≤72 characters, imperative mood, lowercase first word.
- Body explains *why*, not *what*. Reference deliverable IDs from the development plan when applicable.
- One logical change per commit. Each commit on `main` builds, lints, type-checks, and passes its committed tests.

## Pull requests

1. Run the full local quality gate before opening:

   ```bash
   just lint typecheck test cov
   ```

2. Open a PR against `main`. Use a Conventional Commits subject for the PR title.
3. The PR body must include:
   - Summary of changes
   - Deliverable IDs covered (e.g., `P1-VER-02`)
   - Exit-criteria items advanced
   - Test plan checklist (unit / integration / e2e / coverage / mypy / ruff)
4. CI runs lint, typecheck, unit tests, coverage gate, secret scan, and security scan. PRs cannot merge until all pass.
5. At least one maintainer approval is required.

## Testing standards

| Scope | Threshold |
|---|---|
| Overall | 80% |
| New/modified code | 90% |
| Critical paths (`verification/`, `injection/`, `registry/`, `budget/`) | 95% |

- Test naming: `test_<unit>_<scenario>_<expected_outcome>`.
- AAA structure: Arrange, Act, Assert. One logical assertion per test.
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.security`. Unit tests perform no I/O.

## Mode contributions

Modes are loaded only from an explicit allowlist with sha256 pinning. The contribution path:

1. Author the manifest (`mode.toml`) and Jinja prompt template in a standalone repository.
2. Ship eval cases under `tests/` using the `transduce-faithfulness` corpus categories.
3. Submit a PR against the operator's allowlist; operators review the manifest, signature, and eval results before pinning.

Manifest-only modes are the primary contribution path. Python scorers are the reserved escape hatch and run in a sandboxed subprocess with secrets stripped.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
