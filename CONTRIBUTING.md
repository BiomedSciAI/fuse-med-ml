# Contributing to FuseMedML

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

## How do I report a bug or suggest an enhancement?

- As a first step, please search in the [existing issues](https://github.com/IBM/fuse-med-ml/issues) to check if your point has already been addressed.
- If that is not the case, go ahead and [create an issue](https://github.com/IBM/fuse-med-ml/issues/new/choose) of the respective type, providing the details as instructed in the template.

## How do I submit a change?

We welcome contributions via pull requests:

- Fork (or clone) the repo and create your branch from the default branch
- If you have added code that should be tested, add tests
- If any documentation updates are needed, make them
- Ensure the test suite passes and the code lints
- Submit the pull request

Once you have submitted your PR:

- Note that a PR is considered for review only if Travis CI builds successfully
- Upon approval, PR is to be merged using the "squash and merge" option, so that the commit history remains linear and readable

## Styleguides

### Python Styleguide

We use **black**, **flake8** and **mypy** external tools to analyse and enforce uniform code style.

Formatter `black`

When writing code, you should not have to worry about how to format it best. When committing code to a pull request, it should be formatted in one specific way that reduces meaningless diff changes. You can [set up your IDE](https://black.readthedocs.io/en/stable/integrations/editors.html) to format your code on save.

to check for changes:

```sh
black <path> --check --diff --color
```

to apply changes:

```sh
black <path>
```

Linter `flake8`

Checks code style and detects various issues not covered by black.

usage:

```sh
flake8 <path>
```

Static typing with `mypy`

Enforces the usage of type annotations, but not the correctness of them.

usage:

```sh
mypy <path>
```
