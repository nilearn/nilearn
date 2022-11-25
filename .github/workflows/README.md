# Full and Partial doc builds now use GitHub Actions

The purpose of this file is to provide some basic explanations of the workflow to build the documentation and how users can configure documentation generation through commit messages.
This workflow configuration is based on what is done in [scikit-learn](https://github.com/scikit-learn/scikit-learn).

On Pull Requests, Actions run "partial builds" by default which render all the rst files, but only build examples modified in the Pull Request. This saves a lot of time and resources when working on Pull Requests.

Occasionally, some changes necessitate rebuilding the documentation from scratch, for example to see the full effect of the changes. These are called "full builds".

Note that **Actions will always run full builds on main.**

You can request a full build from a Pull Request at any time by including the tag "[full doc]" in your commit message. Note that this will trigger a full build of the documentation which usually takes around 90 minutes.

```bash
$ git commit -m "[full doc] request full build"
```

## Documentation artifacts hosting and deployment on CircleCI

See [.circleci/README.md](/.circleci/README.md) for details.

## Dataset caching

We also implemented a dataset caching strategy within the Actions workflow such that datasets are only downloaded once every week. Once these datasets are cached, they will be used by all jobs running on Actions without requiring any download. This saves a lot of time and avoids potential network errors that can happen when downloading datasets from remote servers.

Note that you can request to download all datasets and ignore the cache at any time by including the tag "[force download]" in your commit message.

To run a full build and download all datasets, you would then combine both tags:

```bash
$ git commit -m "[full doc][force download] request full build"
```

## Skip CI

You can decide to skip documentation building and tests execution at any time by including the tag "[skip ci]" in your commit message.

```bash
$ git commit -m "[skip ci] commit message"
```
