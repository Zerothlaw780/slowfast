# SlowFast

> Recognizes the action in a video (e.g. "running", "playing guitar") with the SlowFast network.

**Category:** Capsule

**Status:** Experimental

**Last Updated:** 10.11.2026

---

## 1. Overview

SlowFast is a video action recognition model. It looks at a short clip through two pathways:
a **Slow** pathway that reads few frames in detail (what is in the scene) and a **Fast** pathway
that reads many frames with fewer channels (how things move). This package uses the pretrained
`slowfast_r50` model from PyTorchVideo, trained on Kinetics-400 (400 action classes).

**How it works:** every frame is resized (short side 256, center crop 256x256) and added to a buffer.
When the buffer holds *Clip Frames* frames, the model runs once every *Inference Interval* frames on
32 evenly spaced frames from the buffer (8 of them go to the Slow pathway). Frames in between reuse
the latest prediction. Each video stream has its own buffer.

**Typical use cases:**

- Recognizing what people are doing in a camera stream
- Tagging video clips by action for search or dataset creation
- Triggering an alert when a specific action is seen

---

## 2. Inputs

| Field Name | Kind/Type | Required | Default | Description |
|---|---|---|---|---|
| `inputImage` | Image (object or list) | Yes | — | Video frames, one per request, in order |

---

## 3. Configuration Parameters

| Parameter | Type | Field Type (UI control) | Allowed Values / Range | Default | Description |
|---|---|---|---|---|---|
| `Device` | string | dropdownlist | CPU / GPU | — | Where the model runs. GPU falls back to CPU (with a log warning) if no CUDA GPU is found |
| `ClipFrames` | number | textInput | [8, 128] | 32 | Recent frames that form one clip; resampled to the 32 frames the model expects |
| `InferenceInterval` | number | textInput | [1, 300] | 16 | Run the model once every N new frames; frames in between reuse the latest prediction |
| `TopK` | number | textInput | [1, 5] | 3 | How many of the most likely actions are reported |
| `ScoreThreshold` | number | textInput | [0.0, 1.0] | 0.2 | Actions below this score are not reported (the best one is always reported) |
| `DrawResult` | bool | dropdownlist | Enable / Disable | — | Write the latest predicted actions and scores on the output image |

---

## 4. Outputs

| Field Name | Kind/Type | Description |
|---|---|---|
| `outputImage` | Image (object or list) | The input frame, with the prediction drawn on it if Draw Result is enabled |
| `actionResult` | list of dict | One row per action, best first: `{rank, action, score}`. Until the first clip is full: `[{rank: 0, action: "warming up", score: 0.0}]` |

---

## 5. Use Case Examples

To be added.

---

## 6. Limitations and Notes

- Runs in the `diginova-pytorch` image (torch 2.6.0, torchvision 0.21.0) and needs `pytorchvideo==0.1.5`.
- Model weights (~264 MB) are downloaded on the first start into `src/weights/checkpoints/` and reused afterwards; they are not stored in the repository. Set `SLOWFAST_WEIGHTS_DIR` to use another folder.
- Predictions are Kinetics-400 classes (sports, daily activities, playing instruments...). Classes such as "walking", "fighting" or "falling" are not in the list; recognizing them needs fine-tuning.
- `pytorchvideo.transforms` does not work with torchvision >= 0.17, so preprocessing is implemented in `src/utils/clip.py`.
- On an M1 CPU one clip takes about 2.2 s, so predictions are made every few seconds, not every frame.

---

## 7. Development

**Tests:** `python -m pytest tests` (needs torch, opencv and pytorchvideo; the model test uses random weights, so no download is needed).


**Setup (once per clone):**

```bash
pip install pre-commit
pre-commit install
```

From then on, each commit runs the same checks as CI:

- **Spec check:** `PackageModel.py` and the executors follow the package development specification.
- **Lint and format:** ruff fixes and formats `src/` automatically. Stage the changes and commit again.
- **Commit message:** must follow [Conventional Commits](https://www.conventionalcommits.org).

To run all checks at any time, use `pre-commit run --all-files`.

**Workflow:**

1. Branch off `develop` and open a PR back into `develop`. CI must pass.
2. Write commit messages as `<type>[(scope)]: <description>`, for example `feat: add threshold config` or `fix(executor): handle empty input`. The type sets the next version:

   | Type | Version bump |
   |---|---|
   | `fix`, `perf` | patch |
   | `feat` | minor |
   | `feat!` or a `BREAKING CHANGE:` footer | major |
   | `refactor`, `docs`, `style`, `test`, `build`, `ci`, `chore` | none |

3. Merge `develop` into `main` to release. A release PR is then opened that bumps the version in `setup.py` and updates `CHANGELOG.md`. Merging it creates the tag and the GitHub Release. Don't edit the version yourself.
