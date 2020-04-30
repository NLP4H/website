---
title: "MS-BERT, Using Neurological Examination Notes to Further Pre-train BlueBERT for Multiple Sclerosis Severity Classification"
date: 2020-04-28T20:11:53-04:00
draft: true
---

## Motivation

Language models are evolving at an unprecedented rate. This can be observed through the development of models such as: [Transformers](https://github.com/huggingface/transformers), [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers), [ELMo](https://arxiv.org/abs/1802.05365), [BioBERT](https://arxiv.org/abs/1901.08746), [BlueBERT](https://github.com/ncbi-nlp/bluebert), [XL-Net](https://arxiv.org/abs/1906.08237), and [Transformers-XL](https://arxiv.org/abs/1901.02860). These language models have created new possibilities by achieving strong results with moderate amounts of data in many Natural Language Processing (NLP) tasks. 

However, the performance of these general language models can be impacted if they are applied to a more specific domain, such as a clinical domain. This is because specific domains use specific vocabulary, syntax and semantics, which differ substantially from the general language. For this reason, further pre-training a general language model on specific domain language can improve performance. For example, BlueBERT language model is developed for clinical NLP tasks. It is built upon BERT and is further pre-trained on ~4,000 million words from PubMed abstracts and ~500 million words from clinical notes [MIMIC-III](https://mimic.physionet.org/).

## What is MS-BERT

While BlueBERT is a strong language model for healthcare applications, we decided to further pre-train it on ~35 million words originating from Multiple Sclerosis (MS) examinations. By further pre-training BlueBERT on a large corpus of consult notes, we provide a language model which aims to provide a deeper understanding of clinical texts, particularly those pertaining to Multiple Sclerosis.

Hence, in this article we look at further pre-training BlueBERT, to develop what we call Multiple Sclerosis-BERT (or MS-BERT for short), and how this language model may be used for clinical prediction tasks with an [AllenNLP](https://allennlp.org/) framework.

MS-BERT is a model developed by students at the University of Toronto along with the Data Science and Advanced Analytics (DSAA) department at St. Michael's Hospital. Our model was able to beat previous baselines such as a Word2Vec CNN on numerous MS severity prediction tasks by up to almost 30%.

## Tutorial

In this section we take you through pre-training MS-BERT and using MS-BERT (with an AllenNLP Framework) for Multiple Sclerosis Severity Classification.

### Step 1: Data Pre-Processing and De-identification

As we were using raw clinical notes, specifically consult notes, there were many identifiable attributes such as patient names, dates, locations and identification numbers. Removal of identifiable information is important not only to protect patient privacy but to also to help the model generalize across patients. 

We processed the notes to remove footers and signature information. The footer and signature information contained no patient information and were a standard signature block that was common among all consult notes. Then, we collected a database of identifiable patient information. This information was combined with regular expression (regex) rules to find and replace identifiable information within the remaining text. We replaced the identifiable information with a contextually similar token from the BERT vocab. These tokens were chosen as they did not previously appear in the note cohort and retained similar contextual meaning in the note after replacement to the original identifiable information. For example, we replaced all male patient names to a male name that was not found within the dataset but was present as a token from the BERT vocab.

| ![de_id_dict](/figures/de_id_dict.png) | 
|:--:| 
| *The tokens in the BERT vocab we used for text replacement as they were not originally found within the consult notes and have similar semantic meaning.* |
 
Next, the de-identified notes were pre-tokenized to the BERT vocabulary. This was done to speed up performance of downstream tasks as tokens could be read in directly vs repeatedly tokenizing each note for each task. We then split the note cohort into test train and validation sets.

### Step 2: Pre-Training MS-BERT

Once we had a de-identified note cohort, we could proceed with pre-training. Given the bi-directional nature of BERT and the size and nature of our notes, we used a masked language modeling pre-training task. We used BlueBERT as a starting point to train our model. Using our de-identified notes, 15% of the tokens from the notes were randomly masked with the task of predicting them based only on the context before and after each masked token. This process used code from the Transformers library and was based on the procedure outlined in [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers), [BlueBERT](https://github.com/ncbi-nlp/bluebert), and [XL-Net](https://arxiv.org/abs/1906.08237). We trained our model over 50 epochs using 125000 training steps for each epoch.
The masked language modeling pre-training task allowed our model to be better adapted to the MS consult notes by adjusting the internal weights of the BlueBERT model to better fit our note cohort. This pre-training results in a unique language model which we call MS-BERT. 

_You can run pre-training as follows:_

```sh {linenos=table}
git clone https://github.com/NLP4H/MS-BERT.git
cd MS-BERT
python pre_training.py --output_dir=output --model_type=bert --model_name_or_path=<path_to_blue_bert> --do_train --train_data_file=<path_to_notes_text_file> --mlm
```

_Here is how you can load MS-BERT:_

```py {linenos=table}
import transformers
tokenizer = AutoTokenizer.from_pretrained("NLP4H/ms_bert")
model = AutoModel.from_pretrained("NLP4H/ms_bert")
```

### AllenNLP Pipeline

In order to use our MS-BERT model in a relevant clinical task, we developed a model using the  [AllenNLP](https://allennlp.org/) framework. AllenNLP is an open-source NLP library that offers a variety of state of the art models and tools built on top of a PyTorch implementation.

_For a more in-depth guide on this process check out our [tutorial](MEDIUM POST TO COME)._

### Step 3: From Clinical Note to Chunk-Level Embedding(s)

Most transformer models have a context length limited to a number of sub-word tokens (512 in case of BlueBERT and MS-BERT). However, the consult notes are often significantly longer than that. In order to address this, we split each tokenized note into chunks of the maximum context length, with the last one potentially being smaller. We use our MS-BERT model to generate chunk-level embeddings which results in a variable length output sequence of 768 dimensional chunk embedding vectors. Note that this chunking process is automated by AllenNLP as demonstrated in Step 5. 

_To tokenize your data use the following code but on your notes:_

```py {linenos=table}
import transformers
from transformers import BertModel, BertTokenizer

text = "Your Clinical Notes"

tokenizer = BertTokenizer.from_pretrained('~/MS_BERT/vocab.txt')
tokenized_text = tokenizer.encode(text, add_special_tokens=True)
```

_Now that your text is tokenized, you can use our dataset reader:_

```py {linenos=table}
import os
import pandas as pd
from typing import Dict, List, Iterator, Tuple, Union
import logging
import torch

from overrides import overrides

from transformers import BertTokenizerFast

# AllenNLP imports
from allennlp.data import Instance
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

@DatasetReader.register('ms_edss19_reader')
class ms_edss19_reader(DatasetReader):
	def __init__(self, tokenizer:str = "BertTokenizerFast", token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
		super().__init__(lazy=False)
		self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer}
		if tokenizer == "BertTokenizerFast":
			self.tokenizer = BertTokenizerFast("/models/base_blue_bert_pt/vocab.txt")
		else:
			raise NotImplementedError

	def text_to_instance(self, text: str, ids: int, labels: float = None) -> Instance:
		text_ids = []
		for t in text[1:-1].split(','):
			text_ids.append(int(t))
		tokens = [Token(text_id=x) for x in text_ids]
		note_field = TextField(tokens, self.token_indexers)
		fields = {"tokens": note_field}

		id_field = MetadataField([ids])
		fields["ids"] = id_field
		if labels:
			label_field = LabelField(str(labels), label_namespace="edss19_labels")
			fields["label"] = label_field
		else:	
			label_field = LabelField(str(0.0), label_namespace="edss19_labels")
			fields["label"] = label_field
		return Instance(fields)

	def _read(self, file_path: str) -> Iterator[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			if row["tokenized_text"] == "[101, 102]" or row["edss_19"] == '' or row["edss_19"] is None:
				continue
			if row["edss_19"] < 0 :
				continue
			label = row["edss_19"]
			yield self.text_to_instance(text=row["tokenized_text"], ids=row["patient_id"], labels = label)
```

_And include it in the config:_

```json {linenos=table, linenostart=13}
	"dataset_reader": {
            "type": "data_scripts.dataset_reader.ms_edss19_reader",
            "token_indexers": { 
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": ms-bert,
                    "namespace": "tokens",
                    "max_length": 512,
                }
            },
```

### Step 4: Generating Note-Level Embeddings

The next part of the architecture is meant to create a note-level embedding by combining the sequence of chunk-level embeddings. We used a CNN encoder provided in the AllenNLP library. This CNN encoder consists of 6 1D convolutions with kernels of size [2, 3, 4, 5, 6, 10] and 128 filters each for a total of 768 dimensions in the output. This output is our final note embedding. The CNN encoder is an implementation of Zhang & Wallace's method from [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820) included in the AllenNLP library.

| ![note_level_embeddings](/figures/note_level_embeddings.PNG) |
|:--:| 
| *Inspired by [A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820).* |

In our case the sentence matrix is 768 x num_chunks and represents the encounter note as a sequence of chunk embeddings. We have 128 filters for each kernel size for a total of 768 filters. We do not predict directly from this feature space but rather feed the concatenated feature vector (our note-level embedding) as input to the next part of our model.

_It is as simple as including this in the config of your model:_

```json {linenos=table,linenostart=39}
 "seq2vec_encoder": {
    "type": "cnn",
    "embedding_dim": 768,
    "num_filters": 128,
    "ngram_filter_sizes": [2, 3, 4, 5, 6, 10]
```

### Step 5: Training a Classification Model

Using AllenNLP's “Basic Classifier" as a starting point, we implemented a custom classifier that used our consult notes as input in order to predict MS severity scores. Our classifier used a custom dataset reader to; read in the variable label, read in the text for each de-identified note, tokenize the words using the MS-BERT vocab, and encode notes into 512 token chunks. Next, the token chunks for each note are passed to a “Text Field" embedder, where MS-BERT weights are applied (i.e. Step 3). The output is then passed into the CNN based encoder described in Step 4 to pool the chunks and generate a note level embedding - a 1D vector of 768. This note level embedding is passed through 2 linear feed forward layers with output dimensions 500 and 250 respectively before finally being passed to a linear classification layer to predict a label for the note.

| ![model_pipeline](/figures/model_pipeline.png) | 
|:--:| 
| *A visualization of the Classification Model using MS-BERT and AllenNLP.* |

Lastly, we used AllenNLP’s training module to train and optimize our classifier for our given prediction task.

## Performance and Outcomes

A common measurement of multiple sclerosis (MS) severity is EDSS or the Expanded Disability Status Scale. This is a scale that increases from 0 to 10 depending on the severity of MS symptoms. It also consists of eight functional sub-scores that relate to how well specific systems or functions in your body work, such as bowel bladder, visual, etc. These were our main targets of prediction. 

We can see a significant improvement by MS-BERT over the baseline in prediction of EDSS, raising Weighted-F1 from 0.897 to 0.941. Interestingly, our model performed better alone than when it was combined with rule based functions through a simple if statement or through [Snorkel](https://www.snorkel.org/). _Want to learn more about Snorkel and how we used it? Check out our Snorkel [tutorial](COMING SOON)!_ 

Additionally, we see a very large improvement over baseline when looking at performance on sub-score prediction. Improving the mean accuracy (or Micro-F1)  by a massive 29.3%. This large gain is interesting because sub-score prediction is a much harder task. Sub-scores are not directly stated within the notes like EDSS. Instead they are often referenced, or symptoms for each sub-score are described. Thus, the significant improvement may come from MS-BERT’s ability to better capture the contextual information in order to determine sub-scores. 

## What We Would Have Done Differently in Retrospect

Our model was trained on notes that were de-identified by replacing both doctor and patient names to the same name -> Ezekiel / Lucie Salamanca. The performance was still quite good as that information is not incredibly relevant to the severity of MS, but for other tasks, our embeddings might be sub-optimal.

We used a pre-trained BERT model with the original vocabulary which does not include many clinical specific tokens, and includes many tokens which are virtually impossible to encounter in clinical notes (non-latin alphabet tokens, names). The next step in our pipeline is to rework the vocabulary and re-train our model on all of MIMIC + Pubmed (following the BlueBERT implementation) and our own notes with this modified vocabulary.

Because the notes are significantly longer then the model's context window (of 512 tokens), our model may not be able to pick up information that is scattered throughout the note. Therefore, the current method of combining embeddings may misrepresent what is actually contained in the note. This can be resolved by training a model like [Transformers-XL](https://arxiv.org/abs/1901.02860), which may come after changing the vocabulary.

## Full Config:

We include our full allennlp config that includes our custom dataset reader, model etc. _If you want to know what each part does please take a look at our in-depth [tutorial](MEDIUM POST TO COME)._

```json {linenos=table}
local experiment_name = "cnn_edss19";

{	"train_options": { 
        "serialization_dir": "/results/dev/" + experiment_name,
        "file_friendly_logging": false,
        "recover": false,
        "force": true,
        "node_rank": 0,
        "batch_weight_key": "",
        "dry_run": false,
    },
	"params":{
	"dataset_reader": {
            "type": "data_scripts.datasetreaders.ms_edss19_reader",
            "token_indexers": { 
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": "/models/base_blue_bert_pt",
                    "namespace": "tokens",
                    "max_length": 512,
                }
            },
        },
        "train_data_path": "/data_dir/train_data.csv",
        "validation_data_path": "/data_dir/val_data.csv",
        "test_data_path": "/data_dir/test_data.csv",
        "unlabeled_data_path": "/data_dir/unlabeled_data.csv",
        "model": {
            "type": "models.ms_classifiers.ms_classifier",
            "text_field_embedder": {
                "token_embedders": {
                    "tokens": {
                        "type": "pretrained_transformer",
                        "model_name": "/models/base_blue_bert_pt",
                        "max_length": 512,
                    }
                }
            },
            "seq2vec_encoder": {
                "type": "cnn",
				"embedding_dim": 768,
				"num_filters": 128,
				"ngram_filter_sizes": [2, 3, 4, 5, 6, 10],
            },
            "feedforward": {
                "input_dim": 768,
                "num_layers": 2,
                "hidden_dims": [500, 250],
                "activations": ["relu","relu"]
            },
            "dropout": 0.1,
            "label_namespace": "edss19_labels"
        },
        "data_loader": {
            "batch_sampler": {
                "type": "bucket",
                "batch_size": 5,
                "padding_noise": 0,
            },
        },
        "validation_data_loader": {
            "batch_sampler": {
                "type": "bucket",
                "batch_size": 5,
                "padding_noise": 0,
            },
        },
        "evaluate_on_test": true,
        "trainer": {
            "optimizer": {
                "type": "huggingface_adamw",
                "lr": 5e-4,
                "weight_decay": 0.01,
                "correct_bias": true
            },
            "learning_rate_scheduler": {
                "type": "reduce_on_plateau",
                "min_lr" : 5e-5
            },
            "patience": 5,
            "validation_metric": "+f1",
            "num_epochs": 50,
            "checkpointer": {
                 "num_serialized_models_to_keep": 1,
                 "keep_serialized_model_every_num_seconds": null,
            },
            "model_save_interval": null,
            "grad_norm": 1.0,
            "no_grad": ["embedder"],
            "grad_clipping": 1.0,
            "summary_interval": 1,
            "histogram_interval": 10,
            "should_log_parameter_statistics": true,
            "should_log_learning_rate": true,
            "log_batch_size_period": 100,
            "moving_average": null,
            "distributed": false,
            "local_rank": 0,
            "cuda_device": 3, 
            "world_size": 1,
            "num_gradient_accumulation_steps": 4,
        }
    }
}
```

## Thank You!
Thanks for reading everyone! If you have any questions please do not hesitate to contact us at nlp4health (at gmail dot) com. :)

## Acknowledgements

We would like to thank the researchers and staff at the Data Science and Advanced Analytics (DSAA) department, St. Michael’s Hospital, for providing consistent support and guidance throughout this project. We would also like to thank Dr. Marzyeh Ghassemi, and Taylor Killan for providing us the opportunity to work on this exciting project. Lastly, we would like to thank Dr. Tony Antoniou and Dr. Jiwon Oh from the MS clinic at St. Michael's Hospital for their support on the neurological examination notes.

