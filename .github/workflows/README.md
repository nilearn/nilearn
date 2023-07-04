# GitHub Actions Specification

## Automatically assign issue

### assign.yml

Allows anyone to self-assign an issue automatically by commenting the word `take` on any issue.

## Auto comment

### auto-comment.yml

Automatically comments on a newly open pull request to provide some guidelines, useful links and a checklist. The checklist is only editable by maintainers at the moment.

## Black formatting

### black.yml

Runs black code formatter on the codebase both in pull requests and on main. Configurations can be found in [pyproject.toml](/pyproject.toml).

## Building the development documentation

### build-docs.yml

#### Full and partial doc builds

This workflow configuration is based on what is done in [scikit-learn](https://github.com/scikit-learn/scikit-learn).

On Pull Requests, Actions run "partial builds" by default which render all the rst files, but only build examples modified in the Pull Request. This saves a lot of time and resources when working on Pull Requests.

Occasionally, some changes necessitate rebuilding the documentation from scratch, for example to see the full effect of the changes. These are called "full builds".

Note that **Actions will always run full builds on main.**

You can request a full build from a Pull Request at any time by including the tag "[full doc]" in your commit message. Note that this will trigger a full build of the documentation which usually takes around 90 minutes.

```bash
$ git commit -m "[full doc] request full build"
```

Though partial build will build modified examples, sometimes code changes on the module side could affect the plots in unmodified examples.
For this, you can request for the CI to build a specific example by using the tag "[example]" and the name of the example. This is useful when wanting to get quick feedback from reviewers.

```bash
$ git commit -m "[example] plot_nilearn_101.py"
```

However for quick checks to do yourself you should always opt for local builds following the instructions here: [building-documentation](https://nilearn.github.io/stable/development.html#building-documentation).

#### Dataset caching

We also implemented a dataset caching strategy within this Actions workflow such that datasets are only downloaded once every week. Once these datasets are cached, they will be used by all jobs running on Actions without requiring any download. This saves a lot of time and avoids potential network errors that can happen when downloading datasets from remote servers.

Note that you can request to download all datasets and ignore the cache at any time by including the tag "[force download]" in your commit message.

To run a full build and download all datasets, you would then combine both tags:

```bash
$ git commit -m "[full doc][force download] request full build"
```

#### Skip CI

You can decide to skip documentation building and tests execution at any time by including the tag "[skip ci]" in your commit message.
For more information, see: https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs

```bash
$ git commit -m "[skip ci] commit message"
```

### trigger-hosting.yml

Runs only if the workflow in `build-docs.yml` completes successfully. Triggers the CircleCI job described below.

## Hosting and deploying development documentation

### [.circleci/config.yml](/.circleci/config.yml)

Artifacts hosting and deployment of development docs use CircleCI. See [.circleci/README.md](/.circleci/README.md) for details.
On a pull request, only the "host" job is run. Then the artifacts can be accessed from the `host_and_deploy_doc` workflow seen under the checks list. Click on "Details" and then on the "host_docs" link on the page that opens. From there you can click on the artifacts tab to see all the html files. If you click on any of them you can then normally navigate the pages from there.
With a merge on main, both "host" and "deploy" jobs are run.

## Check spelling errors

### codespell.yml

Checks for spelling errors. Configured in [pyproject.toml](/pyproject.toml). More information here: https://github.com/codespell-project/actions-codespell

## PEP8 check

### flake8.yml

Uses flake8 tool to verify code is PEP8 compliant. Configured in [.flake8](/.flake8)

## f strings

### f_strings.yml

Checks for f strings in the codebase with [flynt](https://pypi.org/project/flynt/).
Configured in [pyproject.toml](/pyproject.toml)
Flynt will check if it automatically convert "format" or "%" strings to "f strings".
This workflow will fail if it finds any potential target to be converted.

## Sort imports automatically

### isort.yml

Sorts Python imports alphabetically and by section. Configured in [pyproject.toml](/pyproject.toml)

## Running unit tests

### testing.yml

Runs pytest in several environments including several Python and dependencies versions as well as on different systems.

## Test installation

### testing_install.yml

Tries to install Nilearn from wheel & check installation on all operating systems.

### detect_test_pollution.yml

Runs once a month.
Use pytest with the pytest-random-order plugin to run all tests in a random order.
This aims to detect tests that are not properly isolated from each other (test pollution).

## Update precommit hooks

### update_precommit_hooks.yml

Runs weekly to check for updates in versions of precommit hooks and creates a pull request automatically to apply updates.

## Update authors

### update_authors.yml

If the CITATION.CFF file is modified,
this workflow will run to update the AUTHORS file
and the and doc/changes/names.rst file.
