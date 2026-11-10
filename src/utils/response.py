from sdks.novavision.src.helper.package import PackageHelper
from capsules.SlowFast.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SlowFastExecutor,
    SlowFastResponse,
    SlowFastOutputs,
    OutputImage,
    ActionResult,
)


def build_response(context):
    output_image = OutputImage(value=context.output_image)
    action_result = ActionResult(value=context.result)

    outputs = SlowFastOutputs(outputImage=output_image, actionResult=action_result)
    response = SlowFastResponse(outputs=outputs)
    executor = SlowFastExecutor(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=package_configs)
    return package.build_model(context)
