# Hosting and deployment of development docs on CircleCI

The development docs are built using GitHub Actions. See [.github/workflows/README.md](/.github/workflows/README.md) for details.

CircleCI is used to download and host the artifacts from GitHub Actions from builds on pull requests and "main". It also runs the deploy job to deploy the development documentation when there is a merge on "main". These jobs are triggered by Github Actions using [trigger-hosting.yml](/.github/workflows/trigger-hosting.yml) after a successful doc build.
