---
title: "MS-BERT, Using Neurological Examination Notes to Further Pre-train BlueBERT for Multiple Sclerosis Severity Classification"
date: 2020-04-28T20:11:53-04:00
draft: false
---

## Motivation

Language models are evolving at an unprecedented rate. This can be observed through the development of models such as: [Transformers](https://github.com/huggingface/transformers), [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers), [ELMo](https://arxiv.org/abs/1802.05365), [BioBERT](https://arxiv.org/abs/1901.08746), [BlueBERT](https://github.com/ncbi-nlp/bluebert), [XL-Net](https://arxiv.org/abs/1906.08237), and [Transformers-XL](https://arxiv.org/abs/1901.02860). These language models have created new possibilities by achieving strong results with moderate amounts of data in many Natural Language Processing (NLP) tasks. 

However, the performance of these general language models can be impacted if they are applied to a more specific domain, such as a clinical domain. This is because specific domains use specific vocabulary, syntax and semantics, which differ substantially from the general language. For this reason, further pre-training a general language model on specific domain language can improve performance. For example, BlueBERT language model is developed for clinical NLP tasks. It is built upon BERT and is further pre-trained on ~4,000 million words from PubMed abstracts and ~500 million words from clinical notes [MIMIC-III](https://mimic.physionet.org/).

## What is MS-BERT

While BlueBERT is a strong language model for healthcare applications, we decided to further pre-trained it on ~35 million words originating from Multiple Sclerosis (MS) examinations. By further pre-training the BlueBERT model on a large corpus of consult notes, we provide a language model which aims to provide a deeper understanding of clinical texts, particularly those pertaining to Multiple Sclerosis.

Hence, in this article we look at further pre-training BlueBERT, to develop what we call Multiple Sclerosis-BERT (or MS-BERT for short), and how this language model may be used for clinical prediction tasks with an [AllenNLP](https://allennlp.org/) framework.

MS-BERT is a model developed by students at the University of Toronto along with the Data Science and Advanced Analytics (DSAA) department at St. Michael's Hospital. Our model was able to beat previous baselines such as Word2Vec on numerous MS severity prediction tasks by up to almost 30%.

## Tutorial

In this section we take you through pre-training MS-BERT and using MS-BERT (with an AllenNLP Framework) for Multiple Sclerosis Severity Classification.

### Step 1: Data Pre-Processing and De-identification

As we were using raw clinical notes, specifically consult notes, there were many identifiable attributes such as patient names, dates, locations and identification numbers. Removal of identifiable information is important not only to protect patient privacy but to also to help the model generalize across patients. 

We processed the notes to remove footers and signature information. The footer and signature information contained no patient information and were a standard signature block that was common among all consult notes. Then, we collected a database of identifiable patient information. This information was combined with regular expression (regex) rules to find and replace identifiable information within the remaining text. We replaced the identifiable information with a contextually similar token from the BERT vocab. These tokens were chosen as they did not previously appear in the note cohort and retained similar contextual meaning in the note after replacement to the original identifiable information. For example, we would replace all male patient names to a male name that was not found within the dataset but was present as a token from the BERT vocab.

Figure 1: (The tokens in the BERT vocab we used for text replacement as they were not originally found within the consult notes and have similar semantic meaning.)
 
Next, the de-identified notes were pre-tokenized to the BERT vocabulary. This was done to speed up performance of downstream tasks as tokens could be read in directly vs repeatedly tokenizing each note for each task. We then split the note cohort into test train and validation sets.

### Step 2: Pre-Training MS-BERT

Once we had a de-identified note cohort, we could proceed with pre-training. Given the bi-directional nature of BERT and the size and nature of our notes, we used a masked language modeling pre-training task. We used BlueBERT as a starting point to train our model. Using our de-identified notes, 15% of the tokens from the notes were randomly masked with the task of predicting them based only on the context before and after each masked token. This process used code from the Transformers library and was based on the procedure outlined in [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) and [BlueBERT](https://github.com/ncbi-nlp/bluebert), [XL-Net](https://arxiv.org/abs/1906.08237). We trained our model over 50 epochs using 125000 training steps for each epoch.
The masked language modeling pre-training task allowed our model to be better adapted to the MS consult notes by adjusting the internal weights of the BlueBERT model to better fit our note cohort. This pre-training results in a unique language model which we call MS-BERT. 

Add code snipet to run pre-training.py and how to load MS-BERT


How to load MS-BERT:

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

### AllenNLP Pipeline

In order to use our MS-BERT model in a relevant clinical task, we developed a model using the  [AllenNLP](https://allennlp.org/) framework. AllenNLP is an open-source NLP library that offers a variety of state of the art models and tools built on top of a PyTorch implementation.

For a more in-depth guide on this process check out our [tutorial](MEDIUM POST TO COME)

### Step 3: From Clinical Note to Chunk-Level Embedding(s)

Most transformer models have a context length limited to a number of sub-word tokens (512 in case of BlueBERT and MS-BERT). However, the consult notes are often significantly longer than that. In order to address this, we split each tokenized note into chunks of the maximum context length, with the last one potentially being smaller. We use our MS-BERT model to generate chunk-level embeddings which results in a variable length output sequence of 768 dimensional chunk embedding vectors. Note that this chunking process is automated by AllenNLP as demonstrated in Step 5. 

Add code snippet for tokenizing 


Now that your text is tokenized, you can use our dataset reader:

<script src="https://gist.github.com/MichalMalyska/50387452d7eb842175d97a8a7d7601f9.js"></script>

And include it in the config:

<script src="https://gist.github.com/MichalMalyska/2fa760aed5b163f337b998585b11639a.js"></script>


### Step 4: Generating Note Level Embeddings

The next part of the architecture is meant to create a note-level embedding by combining the sequence of chunk-level embeddings. We used a CNN encoder provided in the AllenNLP library. This CNN encoder consists of 6 1D convolutions with kernels of size [2, 3, 4, 5, 6, 10] and 128 filters each for a total of 768 dimensions in the output. This output is our final note embedding. The CNN encoder is an implementation of Zhang & Wallace's method from [A Sensitivity Analysis of (and Practitioners’ Guide to) ConvolutionalNeural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820) included in the AllenNLP library.


Figure 2: (From [A Sensitivity Analysis of (and Practitioners’ Guide to) ConvolutionalNeural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)) 

In our case the sentence matrix is 768 x num_chunks and represents the encounter note as a sequence of chunk embeddings. Our kernel sizes correspond to the region sizes in the figure and we have 128 filters for each region size for a total of 768 filters. We do not predict directly from this feature space as in the figure but rather feed the concatenated feature vector as input to the next part of our model. 

It is as simple as including this in the config of your model:

<script src="https://gist.github.com/MichalMalyska/f454b9452a1bf8297a722abf33d73c90.js"></script>

### Step 5: Training a Classification Model

Using AllenNLP's “Basic Classifier" as a starting point, we implemented a custom classifier that used our consult notes as input in order to predict MS severity scores. Our classifier used a custom dataset reader to; read in the variable label, read in the text for each de-identified note, tokenize the words using the MS-BERT vocab, and encode notes into 512 token chunks. Next, the token chunks for each note are passed to a “Text Field" embedder, where MS-BERT weights are applied (i.e. Step 3). The output is then passed into the CNN based encoder described in Step 4 to pool the chunks and generate a note level embedding - a 1D vector of 768. This note level embedding is passed through 2 linear feed forward layers with output dimensions 500 and 250 respectively before finally being passed to a linear classification layer to predict a label for the note.

Include Figure here

Lastly, we used AllenNLP’s training module to train and optimize our classifier for our given prediction task.

## Performance and Outcomes 

A common measurement of multiple sclerosis (MS) severity is EDSS or the Expanded Disability Status Scale. This is a scale that increases from 0 to 10 depending on the severity of MS symptoms. It also consists of eight functional sub-scores that relate to how well specific systems or functions in your body work, such as bowel bladder, visual, etc. These were our main targets of prediction. 

We can see a significant improvement by MS-BERT over the baseline in prediction of EDSS, raising Weighted-F1 from 0.897 to 0.941. Interestingly, our model performed better alone than when it was combined with rule based functions through a simple if statement or through Snorkel. 

Additionally, we see a very large improvement over baseline when looking at performance on sub-score prediction. Improving the mean accuracy (or Micro-F1)  by a massive 29.3%. This large gain is interesting because sub-score prediction is a much harder task. Sub-scores are not directly stated within the notes like EDSS. Instead they are often referenced, or symptoms for each sub-score are described. Thus, the significant improvement may come from MS-BERT’s ability to better capture the contextual information in order to determine sub-scores. 

## Things we would have done differently in retrospect:

Our model was trained on notes that were de-identified by replacing both doctor and patient names to the same name -> Ezekiel / Lucie Salamanca. The performance was still quite good as that information is not incredibly relevant to the severity of MS, but for other tasks, our embeddings might be sub-optimal.

We used a pre-trained BERT model with the original vocabulary which does not include many clinical specific tokens, and includes many tokens which are virtually impossible to encounter in clinical notes (non-latin alphabet tokens, names). The next step in our pipeline is to rework the vocabulary and re-train our model on all of MIMIC + Pubmed (following the BlueBERT implementation) and our own notes with this modified vocabulary.

Having a shorter context length model - Quite a few of our errors come from the way we capture information - 512 tokens is often enough to capture information but sometimes the information is scattered around the note and the context length makes it impossible to have a complete picture of a note - the combined embeddings misrepresent what actually is in text. This can be resolved by training a model like transformer-xl, which might come after changing the vocabulary.

Thanks for reading everyone! If you have any questions please do not hesitate to contact us at nlp4health at gmail dot com. :)
