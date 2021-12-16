# Documentation building on CircleCI

CircleCI is used to build our documentations and tutorial examples. The purpose of this file is to provide some basic explanations on how users can configure documentation generation on CircleCI through commit messages.

## Full builds and Partial builds

On Pull Requests, CircleCI run "partial builds" by default which render all the rst files, but only build examples modified in the Pull Request. This saves a lot of time and ressources when working on Pull Requests.

Occasionally, some changes necessitate rebuilding the documentation from scratch, for example to see the full effect of the changes. These are called "full builds".

Note that **CircleCI will always run full builds on main.**

You can request a CircleCI full build from a Pull Request at any time by including the tag "[circle full]" in your commit message. Note that this will trigger a full build of the documentation which usually takes around 90 minutes.

```bash
$ git commit -m "[circle full] request full build"
```

## Dataset caching

We also implemented a dataset caching strategy within the CircleCI workflow such that datasets are only downloaded once every week. Once these datasets are cached, they will be used by all jobs running on CircleCI without requiring any download. This saves a lot of time and avoids potential network errors that can happen when downloading datasets from remote servers.

Note that you can request to download all datasets and ignore the cache at any time by including the tag "[force download]" in your commit message.

To run a full build and download all datasets, you would then combine both tags:

```bash
$ git commit -m "[circle full][force download] request full build"
```

## Skip CI

You can decide to skip documentation building and tests execution at any time by including the tag "[skip ci]" in your commit message.

```bash
$ git commit -m "[skip ci] commit message"
```
