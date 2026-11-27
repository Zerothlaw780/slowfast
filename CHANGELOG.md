# Changelog

## 0.1.0 (2026-11-27)

### Features

* Action Recognition (Kinetics): slowfast_r50 on Kinetics-400 with a per-stream clip buffer and preprocessing
* Person Actions (AVA): actions of each detected person with slowfast_r50_detection; results are matched to people by trackerID, otherwise by box overlap
* Custom fine-tuned model option and a fine-tuning notebook (`notebooks/train_slowfast.ipynb`)
* `action` and `actionScore` fields for the best action of each person; one label per person on the image
* Use cases in the README

### Bug Fixes

* Cache model weights in `/storage` so they survive container restarts and package updates
* Remove `target` from the task selector (the executor was empty in app.json)
* Type the detection lists and add the restart flag to the task selector
* Skip frames missing in Redis instead of raising an error
