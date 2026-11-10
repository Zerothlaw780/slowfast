"""
SlowFast: video action recognition.

Model Design version: the image is passed through unchanged so the package can be
registered and tested in a workflow. The model (bootstrap) and the frame buffer are
added in the Executor Implement step.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.helper.executor import Executor
from capsules.SlowFast.src.utils.response import build_response
from capsules.SlowFast.src.models.PackageModel import PackageModel

logger = LoggerManager()


class SlowFast(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        configs = self.request.model.configs.executor.value.value.configs

        self.image = self.request.get_param("inputImage")
        self.device = configs.device.value.value                  # "CPU" / "GPU"
        self.clip_frames = configs.clipFrames.value
        self.interval = configs.inferenceInterval.value
        self.top_k = configs.topK.value
        self.score_threshold = configs.scoreThreshold.value
        self.draw = configs.drawResult.value.value                # True / False

    @staticmethod
    def bootstrap(config: dict = None) -> dict:
        # The SlowFast model will be loaded here once per node process (Executor Implement step).
        return {"state": {}}

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if img is None:
            raise ValueError("SlowFast - input image could not be read (frame missing in Redis)")

        # Model Design: outputs are wired, the model itself comes in Executor Implement.
        self.result = [{"rank": 0, "action": "warming up", "score": 0.0}]
        self.output_image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
