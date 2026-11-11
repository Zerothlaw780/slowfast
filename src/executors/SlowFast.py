"""
SlowFast: video action recognition (slowfast_r50, Kinetics-400).

Each frame is resized and added to a per-stream buffer. When the buffer holds Clip Frames frames,
the model runs on it once every Inference Interval frames; frames in between reuse the latest
prediction. The executor object is created again for every frame, so the model and the buffers
live in the bootstrap dict, which the SDK creates once per node process.
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.helper.executor import Executor
from capsules.SlowFast.src.utils.response import build_response
from capsules.SlowFast.src.models.PackageModel import PackageModel
from capsules.SlowFast.src.utils.clip import ClipBuffer, resize_crop, to_inputs, top_actions, WARMING_UP
from capsules.SlowFast.src.utils.model import load_labels, load_model, resolve_device, predict
from capsules.SlowFast.src.utils.draw import draw_actions, result_lines

logger = LoggerManager()
# /storage is a volume that survives container restarts and package updates; outside NovaVision use src/weights
_DEFAULT_WEIGHTS = "/storage/model/slowfast" if os.path.isdir("/storage") else os.path.join(os.path.dirname(__file__), "..", "weights")
WEIGHTS_DIR = os.environ.get("SLOWFAST_WEIGHTS_DIR", _DEFAULT_WEIGHTS)


class SlowFast(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        configs = self.request.model.configs.executor.value.value.configs

        self.image = self.request.get_param("inputImage")
        self.device_name = configs.device.value.value  # "CPU" / "GPU"
        self.clip_frames = configs.clipFrames.value
        self.interval = configs.inferenceInterval.value
        self.top_k = configs.topK.value
        self.score_threshold = configs.scoreThreshold.value
        self.draw = configs.drawResult.value.value  # True / False

    @staticmethod
    def bootstrap(config: dict = None) -> dict:
        state = {"models": {}, "labels": load_labels(), "streams": {}}
        # The node process calls bootstrap(config=<node>) once at start-up, before any frame arrives,
        # so the first-time checkpoint download (~264 MB) does not block frames in the queue.
        # The main service may call bootstrap() without a config; the model is not needed there.
        if config is not None:
            state["models"]["cpu"] = SlowFast._load()
        return {"state": state}

    @staticmethod
    def _load():
        start = time.time()
        model = load_model(WEIGHTS_DIR)
        logger.info(f"SlowFast - slowfast_r50 loaded from {WEIGHTS_DIR} in {time.time() - start:.1f} s")
        return model

    def _model(self):
        """Model on the selected device. GPU falls back to CPU when CUDA is not available."""
        device, fell_back = resolve_device(self.device_name)
        if fell_back:
            logger.warning("SlowFast - Device is GPU but no CUDA GPU was found; running on CPU")
        models = self.bootstrap["state"]["models"]
        if "cpu" not in models:  # fallback, e.g. if bootstrap ran without a config
            models["cpu"] = self._load()
        if device not in models:
            models[device] = self._load().to(device)
        return models[device], device

    def _buffer(self):
        """One buffer per stream; rebuilt (emptied) when Clip Frames changes."""
        key = self.matchedID or self.uID
        streams = self.bootstrap["state"]["streams"]
        buf = streams.get(key)
        if buf is None or buf.frames.maxlen != self.clip_frames:
            buf = ClipBuffer(self.clip_frames)
            streams[key] = buf
        return buf

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img is None:
            raise ValueError(
                "SlowFast - input image could not be read "
                "(frame missing in Redis: it may have expired because processing fell behind)"
            )

        buf = self._buffer()
        buf.add(resize_crop(img.value))

        if buf.due(self.interval):
            model, device = self._model()
            start = time.time()
            buf.mark_run(predict(model, to_inputs(list(buf.frames), device)))
            logger.info(f"SlowFast - inference on {device} took {time.time() - start:.2f} s")

        if buf.probs is None:
            self.result = WARMING_UP
        else:
            labels = self.bootstrap["state"]["labels"]
            self.result = top_actions(buf.probs, labels, self.top_k, self.score_threshold)

        if self.draw:
            drawn = img.model_copy(deep=True)
            drawn.value = draw_actions(img.value, result_lines(self.result, len(buf.frames), self.clip_frames))
            self.output_image = Image.set_frame(img=drawn, package_uID=self.uID, redis_db=self.redis_db)
        else:
            self.output_image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
