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

> Package Init: the package currently passes the input image through unchanged.
> The model, configs and outputs are added in the next steps.

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
| `DrawResult` | bool | dropdownlist | Enable / Disable | — | Write the predicted action and its score on the output image |

---

## 4. Outputs

| Field Name | Kind/Type | Description |
|---|---|---|
| `outputImage` | Image (object or list) | The input frame, with the prediction drawn on it if Draw Result is enabled |

---

## 5. Use Case Examples

To be added.

---

## 6. Limitations and Notes

- Runs in the `diginova-pytorch` image (torch 2.6.0, torchvision 0.21.0) and needs `pytorchvideo==0.1.5`.
- Model weights (~264 MB) are downloaded at runtime and are not stored in the repository.
- On an M1 CPU one clip takes about 2.2 s, so predictions are made every few seconds, not every frame.

---

## 7. Development

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
