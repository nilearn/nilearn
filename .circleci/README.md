# Documentation artifacts hosting and deployment on CircleCI

Docs are now built using GitHub Actions. See [.github/workflows/README.md](/.github/workflows/README.md) for details.

CircleCI is used to host the artifacts from builds on pull requests and "main". It also runs the deploy job to deploy the development documentation when there is a merge on "main". These jobs are triggered by Github Actions after a successful doc build.