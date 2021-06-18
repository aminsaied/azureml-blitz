# Azure ML Blitz

|Part|Description|
|-|-|
|0-welcome-to-azureml|Run hello world!|
|1-bring-your-own-data|Upload local data to Azure ML|
|2-custom-environments|Use third-party Python libraries|
|3-train-a-model|Train a PyTorch CNN on CIFAR10|
|4-deploy-a-model|Create endpoint to serve model|
|5-hyperdrive|Optimize model|

## Azure ML SDK

This repo uses the Azure ML Python SDK:

```
pip install azureml-core
```

We recommend creating an isolated Conda environment:

```
conda create -n aml python=3.8 pip -y && conda activate aml
pip install azureml-core
python -c "import azureml.core; print(azureml.core.__version__)"
```