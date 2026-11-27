"""
Person Actions (AVA): what each detected person is doing (stand, sit, walk, fall down, talk to...).

Each frame is resized (short side 256, no crop) and buffered together with its person detections.
When the buffer holds Clip Frames frames, the AVA model (slowfast_r50_detection) runs once every
Inference Interval frames on the clip, using the person boxes of the middle frame (AVA labels the
centre of a clip). Between runs, each current person gets the result of the same person: matched by
trackerID (from Object Tracking) when available, otherwise by box overlap.
"""

import copy
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.helper.executor import Executor
from capsules.SlowFast.src.utils.response import build_person_response
from capsules.SlowFast.src.models.PackageModel import PackageModel
from capsules.SlowFast.src.utils.clip import resize_short, to_inputs
from capsules.SlowFast.src.utils.model import load_ava_model, predict_boxes, resolve_device
from capsules.SlowFast.src.utils.persons import (
    PersonClipBuffer,
    actions_from_probs,
    box_xyxy,
    match_actions,
    person_detections,
)
from capsules.SlowFast.src.utils.draw import draw_persons

logger = LoggerManager()
# /storage is a volume that survives container restarts and package updates; outside NovaVision use src/weights
_DEFAULT_WEIGHTS = "/storage/model/slowfast" if os.path.isdir("/storage") else os.path.join(os.path.dirname(__file__), "..", "weights")
WEIGHTS_DIR = os.environ.get("SLOWFAST_WEIGHTS_DIR", _DEFAULT_WEIGHTS)


class PersonActions(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        configs = self.request.model.configs.executor.value.value.configs

        self.image = self.request.get_param("inputImage")
        self.detections = self.request.get_param("inputDetections") or []
        self.device_name = configs.device.value.value  # "CPU" / "GPU"
        self.clip_frames = configs.clipFrames.value
        self.interval = configs.inferenceInterval.value
        self.person_label = configs.personLabel.value
        self.top_k = configs.topK.value
        self.threshold = configs.actionThreshold.value
        self.draw = configs.drawResult.value.value  # True / False

    @staticmethod
    def bootstrap(config: dict = None) -> dict:
        state = {"models": {}, "streams": {}}
        # Loaded once per node process before frames arrive (the first start downloads the AVA checkpoint).
        if config is not None:
            state["models"]["cpu"] = PersonActions._load()
        return {"state": state}

    @staticmethod
    def _load():
        start = time.time()
        model = load_ava_model(WEIGHTS_DIR)
        logger.info(f"SlowFast - slowfast_r50_detection (AVA) loaded from {WEIGHTS_DIR} in {time.time() - start:.1f} s")
        return model

    def _model(self):
        """AVA model on the selected device. GPU falls back to CPU when CUDA is not available."""
        device, fell_back = resolve_device(self.device_name)
        if fell_back:
            logger.warning("SlowFast - Device is GPU but no CUDA GPU was found; running on CPU")
        models = self.bootstrap["state"]["models"]
        if "cpu" not in models:  # fallback, e.g. if bootstrap ran without a config
            models["cpu"] = self._load()
        if device not in models:
            models[device] = copy.deepcopy(models["cpu"]).to(device)
        return models[device], device

    def _buffer(self):
        """One buffer per stream; rebuilt (emptied) when Clip Frames changes."""
        key = self.matchedID or self.uID
        streams = self.bootstrap["state"]["streams"]
        buf = streams.get(key)
        if buf is None or buf.frames.maxlen != self.clip_frames:
            buf = PersonClipBuffer(self.clip_frames)
            streams[key] = buf
        return buf

    def _infer(self, buf):
        persons = buf.middle_persons()
        if not persons:
            buf.results = []
            return
        model, device = self._model()
        boxes = [[c * buf.scale for c in box_xyxy(p)] for p in persons]
        start = time.time()
        probs = predict_boxes(model, to_inputs(list(buf.frames), device), boxes)
        logger.info(f"SlowFast - AVA inference for {len(persons)} person(s) on {device} took {time.time() - start:.2f} s")
        buf.results = [
            (p.get("trackerID"), box_xyxy(p), actions_from_probs(row, self.top_k, self.threshold))
            for p, row in zip(persons, probs)
        ]

    @staticmethod
    def _label(persons, buf):
        """Copy of each person with the action of the matching person from the latest run."""
        out = []
        for p in persons:
            d = dict(p)
            actions = match_actions(p, buf.results) if buf.last_run is not None else []
            d["action"] = actions[0]["action"] if actions else ""
            d["actionScore"] = actions[0]["score"] if actions else 0.0
            d["actions"] = actions
            out.append(d)
        return out

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        persons = person_detections(self.detections, self.person_label)
        buf = self._buffer()
        if img is None:
            # Frame expired in Redis (processing fell behind): skip it and keep the latest actions
            logger.warning("SlowFast - input frame missing in Redis (expired); frame skipped")
            self.output_detections = self._label(persons, buf)
            self.output_image = self.image
            return build_person_response(context=self)

        frame_rgb, scale = resize_short(img.value)
        buf.add(frame_rgb, scale, persons)

        if buf.due(self.interval):
            self._infer(buf)
            buf.last_run = buf.count

        self.output_detections = self._label(persons, buf)

        if self.draw:
            header = None if buf.last_run is not None else f"SlowFast: warming up {len(buf.frames)}/{self.clip_frames}"
            drawn = img.model_copy(deep=True)
            drawn.value = draw_persons(img.value, self.output_detections, header)
            self.output_image = Image.set_frame(img=drawn, package_uID=self.uID, redis_db=self.redis_db)
        else:
            self.output_image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_person_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
