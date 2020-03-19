# Sagemaker deploy endpoint with cloudformation
This deploys a SageMaker endpoint with autoscaling.

## Deploy
1. Use cloudformation to deploy the stack using template [SageMakerCloudFormation.json](SageMakerCloudFormation.json).

2. Once the stack is successfully deployed, you can use the notebook  [InvokeEndpoint.ipynb](InvokeEndpoint.ipynb) to invoke your endpoint

3. Finally to loadtest your endpoint, see [loadtest/Readme.md](loadtest/Readme.md)