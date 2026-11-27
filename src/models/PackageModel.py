from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package,
    Image,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
    Detection,
)


# ============ INPUTS ============


class InputImage(Input):
    """
    Video frames, one per request, in order. The package keeps the most recent frames
    in a buffer and runs the model on them as one clip.
    """

    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


# ============ OUTPUTS ============


class OutputImage(Output):
    """
    Every frame, passed through. If Draw Result is enabled, the latest prediction is written on it.
    """

    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class ActionResult(Output):
    """
    Sent for every frame as a list with one row per predicted action, best first:
    [{rank (int), action (class name: Kinetics-400 or your custom classes), score (0-1)}].
    Between model runs the latest prediction is repeated. Until the first clip is full
    a single row is sent: [{rank: 0, action: "warming up", score: 0.0}].
    """

    name: Literal["actionResult"] = "actionResult"
    value: List[dict]
    type: Literal["list"] = "list"

    class Config:
        title = "Action Result"


class InputDetections(Input):
    """
    Person detections for the same frame, for example from YOLO (and optionally Object Tracking).
    Each item needs boundingBox {left, top, width, height} and classLabel; trackerID is used when present.
    """

    name: Literal["inputDetections"] = "inputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class OutputDetections(Output):
    """
    The person detections of the current frame with the predicted actions added:
    [{boundingBox, classLabel, ..., action, actionScore, actions: [{action, score}]}].
    action / actionScore are the best action and its probability (0-1), handy for tables and filters;
    actions lists up to Top K actions, best first (a person can have several, e.g. "stand" and "talk to").
    Until the first clip is full, or when no action passes Action Threshold, action is "" and actions is empty.
    """

    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Person Actions"


# ============ CONFIGS ============


class CustomWeightsId(Config):
    """
    The fine-tuned checkpoint (.pth) created with notebooks/train_slowfast.ipynb, uploaded to Assets.
    It holds the weights and the class names, so the package reports your own classes.
    """

    name: Literal["WeightsId"] = "WeightsId"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"

    class Config:
        title = "Weights File"
        json_schema_extra = {
            "shortDescription": "Fine-tuned checkpoint (.pth)",
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {"multiple": 0, "returnAttribute": ["name"], "name": "app::logo_wide"},
        }


class ModelKinetics(Config):
    name: Literal["Kinetics400"] = "Kinetics400"
    value: Literal["Kinetics400"] = "Kinetics400"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Kinetics-400 (pretrained)"


class ModelCustom(Config):
    name: Literal["Custom"] = "Custom"
    weightsId: CustomWeightsId
    value: Literal["Custom"] = "Custom"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom (fine-tuned)"


class Model(Config):
    """
    Which weights the package uses:
    - Kinetics-400 (pretrained): 400 everyday and sports actions, no training needed.
    - Custom (fine-tuned): your own action classes. Train with notebooks/train_slowfast.ipynb
      (the Kinetics model is used as the backbone), upload the .pth file to Assets and select it here.
    """

    name: Literal["Model"] = "Model"
    value: Union[ModelKinetics, ModelCustom]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Model"
        json_schema_extra = {"shortDescription": "Pretrained or fine-tuned"}



class DeviceCPU(Config):
    name: Literal["CPU"] = "CPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class DeviceGPU(Config):
    name: Literal["GPU"] = "GPU"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class Device(Config):
    """
    Where the model runs:
    - CPU: works on every device. One clip takes about 2 s, so set Inference Interval accordingly.
    - GPU: much faster, needs an NVIDIA GPU with CUDA. If no GPU is found the package falls back to CPU
      and writes a warning to the log.
    """

    name: Literal["Device"] = "Device"
    value: Union[DeviceCPU, DeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Device"
        json_schema_extra = {"shortDescription": "Run on CPU or GPU"}


class ClipFrames(Config):
    """
    How many recent frames form one clip. The clip is resampled to the 32 frames the model expects
    (8 of them go to the Slow pathway). Clip duration = Clip Frames / frame rate; the model was trained
    on clips of about 2 seconds (64 frames at 30 fps, 20 frames at 10 fps).
    """

    name: Literal["ClipFrames"] = "ClipFrames"
    value: int = Field(ge=8, le=128, default=32)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[8, 128]"] = "[8, 128]"

    class Config:
        title = "Clip Frames"
        json_schema_extra = {"shortDescription": "Frames per clip"}


class InferenceInterval(Config):
    """
    Run the model once every N new frames; frames in between reuse the latest prediction.
    Lower values react faster but cost more compute. On CPU keep N x (1 / frame rate) above ~2 s.
    """

    name: Literal["InferenceInterval"] = "InferenceInterval"
    value: int = Field(ge=1, le=300, default=16)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[1, 300]"] = "[1, 300]"

    class Config:
        title = "Inference Interval"
        json_schema_extra = {"shortDescription": "Run every N frames"}


class TopK(Config):
    """
    How many of the most likely actions are reported, best first.
    """

    name: Literal["TopK"] = "TopK"
    value: int = Field(ge=1, le=5, default=3)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[1, 5]"] = "[1, 5]"

    class Config:
        title = "Top K"
        json_schema_extra = {"shortDescription": "Number of actions reported"}


class ScoreThreshold(Config):
    """
    Actions with a score below this value are not reported. 0 reports all Top K actions.
    If no action passes, the best one is still reported so the output is never empty.
    """

    name: Literal["ScoreThreshold"] = "ScoreThreshold"
    value: float = Field(ge=0.0, le=1.0, default=0.2)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Score Threshold"
        json_schema_extra = {"shortDescription": "Minimum action score"}


class DrawResultTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class DrawResultFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class DrawResult(Config):
    """
    Write the latest predicted actions and their scores on the output image.
    """

    name: Literal["DrawResult"] = "DrawResult"
    value: Union[DrawResultTrue, DrawResultFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Draw Result"
        json_schema_extra = {"shortDescription": "Show prediction on image"}


class PersonLabel(Config):
    """
    classLabel of the detections to analyse, as written by the detector (YOLO uses "person").
    Leave empty to use every detection.
    """

    name: Literal["PersonLabel"] = "PersonLabel"
    value: str = Field(default="person")
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Person Class"
        json_schema_extra = {"shortDescription": "Detections to analyse"}


class ActionThreshold(Config):
    """
    Minimum probability for an action to be reported for a person (0-1). Each action is scored on its own,
    so several actions can pass. Lower values report more (and less certain) actions.
    """

    name: Literal["ActionThreshold"] = "ActionThreshold"
    value: float = Field(ge=0.0, le=1.0, default=0.3)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Action Threshold"
        json_schema_extra = {"shortDescription": "Minimum action probability"}


# ============ EXECUTOR ============


class SlowFastInputs(Inputs):
    inputImage: InputImage


class SlowFastConfigs(Configs):
    model: Model
    device: Device
    clipFrames: ClipFrames
    inferenceInterval: InferenceInterval
    topK: TopK
    scoreThreshold: ScoreThreshold
    drawResult: DrawResult


class SlowFastOutputs(Outputs):
    outputImage: OutputImage
    actionResult: ActionResult


class SlowFastRequest(Request):
    inputs: Optional[SlowFastInputs]
    configs: SlowFastConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class SlowFastResponse(Response):
    outputs: SlowFastOutputs


class SlowFastExecutor(Config):
    """
    Recognizes the action in the most recent frames with the SlowFast network
    (slowfast_r50; Kinetics-400 classes or your own fine-tuned classes).
    """

    name: Literal["SlowFast"] = "SlowFast"
    value: Union[SlowFastRequest, SlowFastResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Action Recognition (Kinetics)"
        json_schema_extra = {"target": {"value": 0}}


class PersonActionsInputs(Inputs):
    inputImage: InputImage
    inputDetections: InputDetections


class PersonActionsConfigs(Configs):
    device: Device
    clipFrames: ClipFrames
    inferenceInterval: InferenceInterval
    personLabel: PersonLabel
    topK: TopK
    actionThreshold: ActionThreshold
    drawResult: DrawResult


class PersonActionsOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class PersonActionsRequest(Request):
    inputs: Optional[PersonActionsInputs]
    configs: PersonActionsConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class PersonActionsResponse(Response):
    outputs: PersonActionsOutputs


class PersonActionsExecutor(Config):
    """
    Recognizes what each detected person is doing (stand, sit, walk, fall down, talk to... 60 AVA actions)
    with the SlowFast network trained on AVA (slowfast_r50_detection). Needs person boxes from a detector.
    """

    name: Literal["PersonActions"] = "PersonActions"
    value: Union[PersonActionsRequest, PersonActionsResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Person Actions (AVA)"
        json_schema_extra = {"target": {"value": 0}}


# ============ PACKAGE ============


class ConfigExecutor(Config):
    """
    Choose the task:
    - Action Recognition (Kinetics): one label for what happens in the whole clip (400 activities and sports).
    - Person Actions (AVA): actions of each person, using person boxes from a detector such as YOLO.
    """

    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SlowFastExecutor, PersonActionsExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Select the task"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["SlowFast"] = "SlowFast"
