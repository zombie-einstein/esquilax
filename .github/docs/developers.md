# Developer Notes

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```bash
poetry install
```

### Pre-Commit Hooks

Pre commit hooks can be installed by running

```bash
pre-commit install
```

Pre-commit checks can then be run using

```bash
task lint
```

### Tests

Tests can be run with

```bash
task test
```

### Documentation

Docs can be built using Sphinx by running

```bash
task docs
```

Built docs will be generated in the `docs/build` folder.

Likewise, documentation tests can be run with

```bash
task doc-tests
```
