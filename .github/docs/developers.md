# Developer Notes

Esquilax uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Development Tasks

Common development tasks can be run using
[taskipy](https://github.com/taskipy/taskipy). The full
list of available tasks can be seen using.

```bash
uv run task --list
```

### Linting

[Pre-commit](https://pre-commit.com/) hooks can be
installed by running

```bash
uv run pre-commit install
```

Linting checks can then be run using

```bash
uv run task lint
```

### Tests

Tests can be run with

```bash
uv run task test
```

### Build & Test Documentation

Docs can be built using
[Sphinx](https://www.sphinx-doc.org/en/master/)
by running

```bash
uv run task docs
```

Built docs will be generated in the `docs/build` folder.

Likewise, documentation tests can be run with

```bash
uv run task doc-tests
```

## Documentation

Building documentation also automatically builds API
documentation from docstrings. Docstrings should be written
using the Numpy format. Types should be documented using typehints
not in the docstring.
