from pydantic import validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


# ============ INPUTS ============

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


# ============ OUTPUTS ============

class OutputImage(Output):
    """
    Every frame, passed through. If Draw Result is enabled, the predicted action is written on it.
    """
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


# ============ CONFIGS ============
# Package Init: only Draw Result for now. Device, inference interval and top-k are added in Model Design.

class DrawResultEnable(Config):
    name: Literal["Enable"] = "Enable"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class DrawResultDisable(Config):
    name: Literal["Disable"] = "Disable"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class DrawResult(Config):
    """
    Write the predicted action and its score on the output image.
    """
    name: Literal["DrawResult"] = "DrawResult"
    value: Union[DrawResultEnable, DrawResultDisable]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Draw Result"
        json_schema_extra = {
            "shortDescription": "Show prediction on image"
        }


# ============ EXECUTOR ============

class SlowFastInputs(Inputs):
    inputImage: InputImage


class SlowFastConfigs(Configs):
    drawResult: DrawResult


class SlowFastOutputs(Outputs):
    outputImage: OutputImage


class SlowFastRequest(Request):
    inputs: Optional[SlowFastInputs]
    configs: SlowFastConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SlowFastResponse(Response):
    outputs: SlowFastOutputs


class SlowFastExecutor(Config):
    """
    Recognizes the action in a short video clip with the SlowFast network (Kinetics-400 classes).
    """
    name: Literal["SlowFast"] = "SlowFast"
    value: Union[SlowFastRequest, SlowFastResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SlowFast Action Recognition"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ============ PACKAGE ============

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SlowFastExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["SlowFast"] = "SlowFast"
