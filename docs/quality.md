# Quality checks

The repository uses three validation lanes:

```bash
./scripts/test/check_pr.sh
./scripts/test/check_all.sh
./scripts/test/check_nightly.sh
```

`check_pr.sh` is the fast lane for source changes. It runs scoped linting, type checking, and CPU-safe tests for the touched modules.

`check_all.sh` is the broad local lane. It runs repo-wide linting and type checking, then the full pytest suite with coverage. The package-wide coverage percentage is informational; the enforced gate is source-based coverage in `scripts/test/check_source_coverage.sh`.

`check_nightly.sh` is the broad pytest lane for longer local or scheduled runs. Renderer tests are marked separately because they depend on headless rendering and GPU/display details.

GitHub Actions runs `check_pr.sh` for pull requests and pushes to `main`. The workflow installs the locked `full` environment but does not compile the optional CUDA extensions.

The tracked hooks use the same commands:

```bash
git config core.hooksPath scripts/hooks
```

`commit-msg` checks the Conventional Commit subject. `pre-push` runs `check_pr.sh`. The development bootstrap command configures the hooks for a clone.

For paper-command sanity, run:

```bash
pytest experiments/tests/test_reproduction.py experiments/tests/test_reproduction_hydra.py
```

Those tests check the public command surface for every recipe and Hydra-compose every recipe that goes through `scripts/run.sh`. They do not train models or verify numerical paper results.
