from sagemaker.estimator import Framework
from sagemaker.predictor import Predictor
from sagemaker.mxnet import MXNetModel
from sagemaker.mxnet.model import MXNetPredictor
from sagemaker import utils
from sagemaker import image_uris
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import StringDeserializer

REGION_MAPPING = {
    "af-south-1": "626614931356",
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ca-central-1": "763104351884",
    "eu-central-1": "763104351884",
    "eu-north-1": "763104351884",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "eu-south-1": "692866216735",
    "me-south-1": "217643126080",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884"
}


class AutoGluonTraining(Framework):
    def __init__(
        self,
        entry_point,
        region,
        framework_version,
        instance_type,
        source_dir=None,
        hyperparameters=None,
        **kwargs,
    ):
        account = REGION_MAPPING[region]
        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/autogluon-training:{framework_version}-cpu-py37-ubuntu18.04"
        super().__init__(
            entry_point,
            source_dir,
            hyperparameters,
            instance_type=instance_type,
            image_uri=image_uri,
            **kwargs,
        )

    def _configure_distribution(self, distributions):
        return

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=None,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_name=None,
        **kwargs,
    ):
        return None


class AutoGluonTabularPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, serializer=CSVSerializer(), deserializer=StringDeserializer(), **kwargs
        )


class AutoGluonInferenceModel(MXNetModel):
    def __init__(
        self, model_data, role, entry_point, region, framework_version, instance_type, **kwargs
    ):
        account = REGION_MAPPING[region]
        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/autogluon-inference:{framework_version}-cpu-py37-ubuntu16.04"
        super().__init__(
            model_data, role, entry_point, image_uri=image_uri, framework_version="1.8.0", **kwargs
        )
