from sdks.novavision.src.helper.package import PackageHelper
from components.PerspectiveCorrection.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    PerspectiveCorrectionOutputs,
    PerspectiveCorrectionResponse,
    PerspectiveCorrectionExecutor,
    OutputImage
)

def build_response(context):
    output_image = OutputImage(value=context.image)
    outputs = PerspectiveCorrectionOutputs(outputImage=output_image)
    perspective_response = PerspectiveCorrectionResponse(outputs=outputs)
    perspective_executor = PerspectiveCorrectionExecutor(value=perspective_response)
    executor = ConfigExecutor(value=perspective_executor)
    package_configs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    return package.build_model(context)