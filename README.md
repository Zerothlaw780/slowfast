# SlowFast

Video action recognition package for NovaVision, based on the SlowFast network
(`slowfast_r50` from PyTorchVideo, pretrained on Kinetics-400).

> Status: Package Init. The package currently passes the image through unchanged.
> Model, configs and usage are documented in later steps.

## Structure

```
src/
├── executors/SlowFast.py      # executor (Capsule): __init__, bootstrap, run
├── models/PackageModel.py     # inputs, outputs, configs (Pydantic)
└── utils/response.py          # builds the response model
```

## Requirements

- Image: `diginova-pytorch` (torch 2.6.0, torchvision 0.21.0)
- Extra: `pytorchvideo==0.1.5`
