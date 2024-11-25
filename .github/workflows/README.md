# GitHub Actions Specification

## Automatically assign issue

### assign.yml

Allows anyone to self-assign an issue automatically by commenting the word `take` on any issue.

## Auto comment

### auto-comment.yml

Automatically comments on a newly open pull request to provide some guidelines, useful links and a checklist. The checklist is only editable by maintainers at the moment.

## Building the development documentation

### build-docs.yml

#### Full and partial doc builds

This original workflow derived from on what is done in [scikit-learn](https://github.com/scikit-learn/scikit-learn).

On Pull Requests, Actions run "partial builds" by default which render all the rst files,
but only build examples modified in the Pull Request.
This saves a lot of time and resources when working on Pull Requests.

Occasionally, some changes necessitate rebuilding the documentation from scratch,
for example to see the full effect of the changes.
These are called "full builds".

Note that **Actions will always run full builds on main.**

You can request a full build from a Pull Request at any time by including the tag "[full doc]" in your commit message.
Note that this will trigger a full build of the documentation which usually takes around 90 minutes.

```bash
$ git commit -m "[full doc] request full build"
```

Though partial build will build modified examples, sometimes code changes on the module side could affect the plots in unmodified examples.
For this, you can request for the CI to build a specific example by using the tag "[example]" and the name of the example. This is useful when wanting to get quick feedback from reviewers.

```bash
$ git commit -m "[example] plot_nilearn_101.py"
```

However for quick checks to do yourself you should always opt for local builds following the instructions here: [building-documentation](https://nilearn.github.io/stable/development.html#building-documentation).

Note: setuptools needs to be installed to run the doc build with python >=3.12.

Upon a successful build of the doc, it is zipped and uploaded as an artifact.
A circle-ci workflow is then triggered. See below.

#### Dataset caching

We also implemented a dataset caching strategy within this Actions workflow such that datasets are only downloaded once every month.
Once these datasets are cached, they will be used by all jobs running on Actions without requiring any download.
This saves a lot of time and avoids potential network errors that can happen when downloading datasets from remote servers.

Note that you can request to download all datasets and ignore the cache at any time
by including the tag "[force download]" in your commit message.

To run a full build and download all datasets, you would then combine both tags:

```bash
$ git commit -m "[full doc][force download] request full build"
```

## Building the stable release documentation

### release-docs.yml

Should be triggered automatically after merging and tagging a release PR to
build the stable docs with a GitHub runner and push to nilearn.github.io.
Can also be triggered manually.

## Hosting and deploying development documentation

### [.circleci/config.yml](/.circleci/config.yml)

Artifacts hosting and deployment of development docs use CircleCI.
See [.circleci/README.md](../../.circleci/README.md) for details.
On a pull request, only the "host" job is run.
Then the artifacts can be accessed from the `host_and_deploy_doc` workflow seen under the checks list.
Click on "Details" and then on the "host_docs" link on the page that opens.
From there you can click on the artifacts tab to see all the html files.
If you click on any of them you can then normally navigate the pages from there.
With a merge on main, both "host" and "deploy" jobs are run.

## Check run time of CI

### check_gha_workflow.yml

Pings github API to collect information about:
- how long each run of the test suite lasted,
- how long the build of the doc lasted.

Plots the results and saves it as an artefact to download and manually inspect
to see if there is a trend in tests taking longer.

## Style checks

### check_style_guide.yml

Relies on pre-commit to run a series of check on the content of the repositories.
See the config [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml).

## Running unit tests

### test_with_tox.yml

Runs pytest in several environments including several Python and dependencies versions as well as on different systems.
All environments are defined in [tox.ini](../../tox.ini).

### nightly_dependencies.yml

Run test suite using the nightly release of Nilearn dependencies.
Runs on `main` (by a push or on manual trigger from the `Action` tab)
or from a PR if commit message includes `[test nightly]`.

When running on `main`, if the workflow fails the action will open an issue
using this issue [template](../nightly_failure.md).

## Test installation

### testing_install.yml

Tries to install Nilearn from wheel & check installation on all operating systems.

### testing_minimum.yml

This workflow is triggered when a new commit is pushed to the main branch (or when a pull request is merged) and is also automatically run once a month.

Checks that installing the minimum version of a given dependency of Nilearn
along with the latest version of all the other dependencies leads to a successful run of all the tests.

## Update precommit hooks

### update_precommit_hooks.yml

Runs weekly to check for updates in versions of precommit hooks and creates a pull request automatically to apply updates.

## Update authors

### update_authors.yml

If the CITATION.CFF file is modified,
this workflow will run to update the AUTHORS file
and the and doc/changes/names.rst file.
