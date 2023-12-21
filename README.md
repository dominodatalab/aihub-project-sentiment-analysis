# Sentiment analysis on product review data

In this project we demonstrate the use of a pre-trained Masked Language Model (MLM) known as BERT in Domino and the process of fine-tunning the model for a specific task.

## License
This template is licensed under Apache 2.0 and contains the following components: 
* distilbert [Apache 2.0](https://huggingface.co/distilbert-base-uncased)
* NVIDIA [EULA](https://docs.nvidia.com/cuda/eula/index.html#license-grant)
* pandas [BSD 3](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
* MLFlow [Apache 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
* Transformer [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
* pytorch [Caffe 2](https://github.com/pytorch/pytorch/blob/main/LICENSE)

## About this template

Fine-tuning a pre-trained transformer an effective and state-of-the-art technique for solving NLP problems.

1. The process typically begins with a pre-trained model, which is not task specific. This model is trained on a large corpora of unlabelled data (e.g. Wikipedia) using masked-language modelling loss. Bidirectional Encoder Representations from Transformers (BERT) is an example for such a model [1].

2. Optionally, the model can undergo a process of domain specific adaptive fine-tuning, which produces a new model with narrower focus. This new model may be better prepared to address domain-specific challenges as it is now closer to the expected distribution of the target data. An example for such a model is FinBERT [2].

3. The final stage of the model building pipeline is the task-based fine-tuning. This is the step where the model is further fine-tuned using a specific supervised learning dataset and task (like text classification or NER). The resulting model is now able to perform the specific task on a similarly distributed dataset.

The advantage of using the process outlined above is that pre-training is typically computationally expensive and requires the processing of large volumes of data. Task fine-tuning, on the other hand, is relatively cheap and quick to accomplish, especially if distributed and GPU-accelerated compute can be employed in the process.

In this demo project we use the [Sentiment Analysis for Amazon Reviews](https://huggingface.co/datasets/amazon_polarity), which provides 3.5M samples of Amazon product reviews, and their corresponding sentiments (positive, negative). Because of how large this dataset is (4GB), for demonstration purposes you'll see us use only a subset. 
But feel free to change that to use the entire dataset in full.

The assets available in this project are:

* **finetune.ipynb** - A notebook, illustrating the process of getting distilbert from [Huggingface 🤗](https://huggingface.co/distilbert-base-uncased) into Domino, and using GPU-accelerated backend for the purposes of fine-tuning it with the Sentiment Analysis from Amazon Reviews dataset
* **finetune.py** - A script, which performs the same fine-tuning process but can be scheduled and executed as a Domino job. The script also accepts the following optional command line arguments, with sensible defaults:
    * *lr* - learning rate for the fine-tuning process
    * *epochs* - number of training epochs
    * *train_batch_size* - how large of a batch size for training 
    * *eval_batch_size* - how large of a batch size for evaluation 
    * *dataset_name* - which dataset to fine-tune with 
    * *text_col* - the name of the column in the dataset representing the text 
    * *label_col* - the name of the column in the dataset representing the label 
    * *distilbert_model* - the specific Distil-BERT model to use 
* **score.py** - A scoring function, which is used to deploy the fine-tuned model as a [Domino API](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/host-models-as-rest-apis/)
* **app.sh** - Launch instructions for the accompanying Streamlit app

# Model API calls

The **score.py** provides a scoring function with the following signature: `predict_sentiment(sentence)`. To test it, you could use the following JSON payload:

```
{
  "data": {
    "sentence": "The item came damanged, 1 star."
  }
}
```

# Set up instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

## Environment Requirements

`quay.io/domino/pre-release-environments:project-hub-gpu.main.latest`

**Pluggable Workspace Tools** 
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [ "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/workspaces/vscode/start" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```

Please change the value in `start` according to your Domino version.

**Important !!** The version of PyTorch included in the above image is not compatible with GPUs with the Ampere architecture. Please check the PyTorch compatibility with your GPU before running this code. 

The following environment variables should also be set at the project level:

```
DISABLE_MLFLOW_INTEGRATION=TRUE	
TF_ENABLE_ONEDNN_OPTS=0
```

## Hardware Requirements
You also need to make sure that the hardware tier running the notebook or the fine-tuning script has sufficient resources. An *nvidia-low-g4dn-xlarge* hardware tier is recommended, as it provides GPU-acceleration that the fine-tunning can take advantage of.

# References

[1] Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 11 October 2018, [arXiv:1810.04805v2](https://arxiv.org/abs/1810.04805)

[2] Dogu Araci, FinBERT: Financial Sentiment Analysis with Pre-trained Language Models, 2019, [arXiv:1908.10063](http://arxiv.org/abs/1908.10063)
