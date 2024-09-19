# Developer Notes

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```bash
poetry install
```

This will initialise a virtual environment,``.venv`` in
the project root (this behaviour can be modified in
the [`poetry.toml`](/poetry.toml) file). This environment
can then be activated with

```bash
source .venv/bin/activate
```

## Development Tasks

Common development tasks can be run using
[taskipy](https://github.com/taskipy/taskipy). The full
list of available tasks can be seen using.

```bash
task --list
```

### Linting

[Pre-commit](https://pre-commit.com/) hooks can be
installed by running

```bash
pre-commit install
```

Linting checks can then be run using

```bash
task lint
```

### Tests

Tests can be run with

```bash
task test
```

### Build & Test Documentation

Docs can be built using
[Sphinx](https://www.sphinx-doc.org/en/master/)
by running

```bash
task docs
```

Built docs will be generated in the `docs/build` folder.

Likewise, documentation tests can be run with

```bash
task doc-tests
```

## Documentation

Building documentation also automatically builds API
documentation from docstrings. Docstrings should be written
using the Numpy format. Types should be documented using typehints
not in the docstring.
