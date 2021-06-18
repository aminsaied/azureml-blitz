# Run local training

Create environment for local testing:

```
conda create -n aml-pytorch python=3.8 pip -y && conda activate aml-pytorch
conda install pytorch torchvision cpuonly -c pytorch-lts
pip install azureml-core numpy matplotlib
```