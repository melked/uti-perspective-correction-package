
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


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


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
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


class KeepSideFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class KeepSideTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class KeepSideBBox(Config):
    """
        Rotate image without catting off sides.
    """
    name: Literal["KeepSide"] = "KeepSide"
    value: Union[KeepSideTrue, KeepSideFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Keep Sides"


class Degree(Config):
    """
        Positive angles specify counterclockwise rotation while negative angles indicate clockwise rotation.
    """
    name: Literal["Degree"] = "Degree"
    value: int = Field(ge=-359.0, le=359.0,default=0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[-359, 359]"] = "[-359, 359]"

    class Config:
        title = "Angle"


class PerspectiveCorrectionInputs(Inputs):
    inputImage: InputImage


class PerspectiveCorrectionConfigs(Configs):
    degree: Degree
    drawBBox: KeepSideBBox


class PerspectiveCorrectionOutputs(Outputs):
    outputImage: OutputImage


class PerspectiveCorrectionRequest(Request):
    inputs: Optional[PerspectiveCorrectionInputs]
    configs: PerspectiveCorrectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PerspectiveCorrectionResponse(Response):
    outputs: PerspectiveCorrectionOutputs


class PackageExecutor(Config):
    name: Literal["PerspectiveCorrection"] = "PerspectiveCorrection"
    value: Union[PerspectiveCorrectionRequest, PerspectiveCorrectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "PerspectiveCorrection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PackageExecutor]
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
    type: Literal["component"] = "component"
    name: Literal["PerspectiveCorrection"] = "PerspectiveCorrection"
