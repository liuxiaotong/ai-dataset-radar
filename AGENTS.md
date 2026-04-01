# ai-dataset-radar Release Contract

## Canonical path

- canonical repo: `/Users/liukai/ai-dataset-radar`
- canonical branch: `main`
- release mode: `indirect_publish`

`ai-dataset-radar` does not own a direct production website deploy path.

## What counts as publishing

- running `run_weekly.sh`, package release, or tag creation does **not** by itself publish the production insights pages
- production only changes after radar output is pulled into `knowlyr-website` and the downstream website deploy succeeds

## Release handoff

- treat this repo as an upstream content/data generator
- when users ask “has radar gone live,” verify the downstream `knowlyr-website` release rather than only checking this repo

## Forbidden shortcut

- do not describe tag push or package publish as the final production release path
