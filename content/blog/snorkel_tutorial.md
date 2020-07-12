---
title: "Using Snorkel and NLP Models to Predict Multiple Sclerosis Severity Scores"
date: 2020-04-29T20:11:53-04:00
draft: false
---
In this tutorial, we will walk through the process of using Snorkel to generate labels for an unlabelled dataset. We will provide you examples of basic Snorkel components by guiding you through a real clinical application of Snorkel. Specifically, we will use Snorkel to try to boost our results in predicting [Multiple Sclerosis (MS) severity scores](https://www.mstrust.org.uk/a-z/expanded-disability-status-scale-edss). Enjoy!

_Check out the [Snorkel Intro Tutorial](https://www.snorkel.org/use-cases/01-spam-tutorial) for a walk through on spam labelling. For more examples of high-performance in real-world uses of Snorkel, see [Snorkel's publication list](https://www.snorkel.org/resources/)._

_Check out our other work focused on NLP for MS severity classification [here](https://medium.com/@nlp4health/ms-bert-using-neurological-examination-notes-for-multiple-sclerosis-severity-classification-f75f13600d3e)._

## What is Snorkel? 

Snorkel is a system that facilitates the process of building and managing training datasets without manual labelling. The first component of a Snorkel pipeline includes labelling functions, which are designed to be weak heuristic functions that predict a label given unlabelled data. The labelling functions that we developed for MS severity score labelling were the following: 
- Multiple key-word searches (using regular expressions) within the text. For example, in finding a severity score we searched for the phrase in numeric format and in roman numeral format.
- Common baselines such as logistic regression, linear discriminant analysis, and support vector machines which were trained using term frequency–inverse document frequency (or tf-idf for short) features. 
- Word2Vec Convolutional Neural Network (CNN).
- Our MS-BERT classifier described [in this blog post](https://medium.com/@nlp4health/ms-bert-using-neurological-examination-notes-for-multiple-sclerosis-severity-classification-f75f13600d3e).

The second component of the Snorkel pipeline is a generative model that outputs a single confidence weighted training label per data point given predictions from all the labelling functions. It does this by learning to estimate the accuracy and correlations of the labelling functions based on their agreements and disagreements. 

## Snorkel Tutorial

To reiterate, in this article we demonstrate label generation for MS severity scores. A common measurement of MS severity is EDSS or the Expanded Disability Status Scale. This is a scale that increases from 0 to 10 depending on the severity of MS symptoms. We will refer to EDSS in general as the MS severity score but for our keen readers we thought we would provide this information. This score is further described [here](https://www.mstrust.org.uk/a-z/expanded-disability-status-scale-edss).

### Step 0: Acquire a Dataset
In our task, we worked with a dataset compiled by a leading MS research hospital, containing over *70,000* MS consult notes for about 5000 patients. Of the 70,000 notes only *16,000* are manually labeled by an expert for MS severity. This means that their are approximately *54,000* unlabelled notes. As you may or not be aware, having a larger dataset to train models generally lead to better model performance. Hence, we used Snorkel to generate what we call 'silver' labels for our *54,000* unlabelled notes. The *16,000* 'gold' labelled notes were used to train our classifiers before creating their respective labelling function.

### Step 1: Installing Snorkel 
To install Snorkel to your project, you can run the following:
```bash
# For pip users
pip install snorkel

# For conda users
conda install snorkel -c conda-forge
```

### Step 2: Adding Labelling Functions 
#### Setting up 

Labelling functions allow you to define weak heuristics and rules that predict a label given unlabelled data. These heuristics can be derived from expert knowledge or other labelling models. In the case of MS severity score prediction, our labelling functions included: key-word search functions derived from clinicians, baseline models trained to predict MS severity scores (tf-idf, word2vec cnn, etc.), and our MS-BERT classifier. 

As you will see below, you mark labelling functions by adding "@labeling_function()" above the function. For each labelling function, a single row of a dataframe containing unlabelled data (i.e. one observation/sample) is passed in. Each labelling function applies heuristics or models to obtain a prediction for each row. If the prediction is not found, the function abstains (i.e. returns -1).  

```python
from snorkel.labeling import labeling_function

# multiple keyword searches

@labeling_function()
def ms_severity_decimal_lf(df_row):

    # do something to predict score
    
    return score
    
@labeling_function()
def ms_severity_word_lf(df_row):

    # do something to predict score
    
    return score
    
# MS-BERT classifier    

@labeling_function()
def msbc_lf(df_row):

    # do something to predict score
    
    return score   

# common baselines

@labeling_function()
def logreg_tfidf_lf(df_row):

    # do something to predict score
    
    return score
# include as many labelling functions as you can...
```

When all labelling functions have been defined, you can make use of the "PandasLFApplier" to obtain a matrix of predictions given all labelling functions. 

```python
from snorkel.labeling import PandasLFApplier

# create a list of the labelling functions you have defined
lfs = [ms_severity_decimal_lf, ms_severity_word_lf, msbc_lf, logreg_tfidf_lf]

applier = PandasLFApplier(lfs=lfs)
L_predictions = applier.apply(df=df_unlabelled)
```
Upon running the following code, you will obtain a (N X num_lfs) L_predictions matrix, where N is number of observations in 'df_unlabelled' and 'num_lfs' is the number of labelling functions defined in 'lfs'.

#### Labelling Function Example #1: Key-Word Search

Below shows an example of a key-word search (using regular expressions) used to extract MS severity scores recorded in decimal form. The regular expression functions are applied to attempt to search for the MS severity score recorded in decimal form. If found, the funtion returns the score in the appropriate output format. Else, the function abstains (ie. returns -1) to indicate that the score is not found. 

```python
# note that the labeling function takes in only one observation called 'df_row'
@labeling_function()
def ms_severity_decimal_lf(df_row):
    
    # extract text
    note = df_row.text
    
    # perform regular expression search
    phrase = re.compile(r"ms severity score", re.IGNORECASE)
    # looking for a number in decimal format
    phrase_decimal = re.compile(r"\d\.\d")

    #default to -1 if nothing is found (ABSTAIN)
    score = -1 
    sentences = sent_tokenize(note)
    for sent in sentences:
        ms_severity_mentions = re.search(phrase, sent)

        # Find sentence with "ms severity score"
        if ms_severity_mentions != None:
            filtered_sentence = sent[ms_severity_mentions.end():]

            # If score mentioned in decimal form -> get first number that is mentioned after the ms severity score mention
            if len(re.findall(phrase_decimal, filtered_sentence)) > 0:
                score = float(re.findall(phrase_decimal, filtered_sentence)[0])
                break
    # make sure score makes sense given context of problem (score is between 0-10)
    if score not in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        score = -1
    
    # score converted to class category
    label_dict = {0.0:0,
            1.0:1,
            1.5:2,
            2.0:3,
            2.5:4,
            3.0:5,
            3.5:6,
            4.0:7,
            4.5:8,
            5.0:9,
            5.5:10,
            6.0:11,
            6.5:12,
            7.0:13,
            7.5:14,
            8.0:15,
            8.5:16,
            9.0:17,
            9.5:18,
            -1:-1}
        
    return label_dict[score]
```

#### Labelling Function Example #2: Trained Classifier 

Above we see an example using a key-word search. To integrate a trained classifier, you must perform one extra step. That is, you must train and export your model before creating your labelling function. Here is an example of how we trained a logistic regression that was built ontop of tf-idf features.
```python
# Sklearn models imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
# train tfidf model
def train_lr_model(notes_train, notes_val,Y_train,Y_val):

    # set save directory for logistic regression and tf-idf feature model
    save_dir = "path to saving directory"
    # load tfidf vectorizer
    tf = TfidfVectorizer( max_features=1500)

    # fit tfidfvectorizer to train and transform train and valid
    X_train = tf.fit_transform(notes_train)
    X_valid = tf.transform(notes_val)
    
    #save the tfidf transformer for prediction
    os.chdir(save_dir)
    model_name = "tf.pkl"
    with open(model_name, 'wb') as file:
        pickle.dump(tf, file)

    # hyper Param Tuning:
    tuned_params_LR = [{"C": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
        1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]

    print("hyperparam tuning for LR")
    best_c = 0
    best_score = 0
    for i, param in enumerate(tuned_params_LR[0]["C"]):
        # train model
        clf = LogisticRegression(C=param, solver='lbfgs')
        clf.fit(X_train,Y_train)
        # see if model performs best
        score = accuracy_score(clf.predict(X_valid),Y_val)
        # update best parameters if better
        if score > best_score:
            best_score = score
            best_c = param
    #save best model
    print("training model to best:", best_c)
    clf = LogisticRegression(C=best_c, solver='lbfgs')
    clf.fit(X_train,Y_train)
    os.chdir(save_dir)
    model_name = "log_reg_baseline.pkl"
    with open(model_name, 'wb') as file:
        pickle.dump(clf, file)
    return
```
With the model trained, implementing a labelling function is as simple as this:
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

@labeling_function()
def logreg_tfidf_lf(df_row):

    # load models for prediction
    model_path = "path to logistic regression and tf-idf model"
    os.chdir(model_path)
    model_name = "tf.pkl"
    with open(model_name, 'rb') as file:
        tf = pickle.load(file)
    with open("log_reg_baseline.pkl", 'rb') as file:
        model = pickle.load(file)
        
    # tfidf transform
    X = tf.transform(df_row.texts)
    
    # predict
    # model predicts from tfidf input X
    score = model.predict(X)

    return score
```

### Step 3(a): Using Snorkel's Majority Vote 

Some would say the simpliest function Snorkel uses to generate a label is 'Majority Vote'. Majority Vote, as the name implies, makes a prediction based on the most voted for class.

To implement Majority Vote you must specify the 'cardinality' (i.e. number of classes). 

```python
import snorkel
from snorkel.labeling import MajorityLabelVoter

majority_model = MajorityLabelVoter(cardinality=num_classes)
y_preds = majority_model.predict(L=L_unlabelled)
```

### Step 3(b): Using Snorkel's Label Model

To take advantange of Snorkel's full functionality, we used the 'Label Model' to generate a single confidence-weighted label given a matrix of predictions obtained from all the labelling functions (ie. L_unlabelled). The Label Model predicts by learning to estimate the accuracy and correlations of the labelling functions based on their agreements and disagreements. 

As shown below, you can define a Label Model and specify 'cardinality'. After you fit the Label Model with L_unlabelled, it will generate single predictions for the unlabelled data. 

```python
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=num_classes, verbose=True)     
label_model.fit(L_train=L_unlabelled)
# get our predictions for the unlabelled dataset
y_preds = label_model.predict(L=L_unlabelled)
```

### Step 4: Evaluation Tools

#### LF Analysis - Coverage, Overlaps, Conflicts

To better understand how your labelling functions are functioning, you can make use of Snorkel's LFAnalysis. The LF analysis reports the polarity, coverage, overlap, and conflicts of each labelling function. 

The definition of these terms are as follows and you can refer to the [Snorkel documentation](https://snorkel.readthedocs.io/en/v0.9.1/packages/_autosummary/labeling/snorkel.labeling.LFAnalysis.html#snorkel.labeling.LFAnalysis.lf_polarities) for more information:

* Polarity: Infer the polarities of each LF based on evidence in a label matrix.
* Coverage: Computes the fraction of data points with at least one label.
* Overlap: Computes the fraction of data points with at least two (non-abstain) labels.
* Conflicts: Computes the fraction of data points each labelling function disagrees with at least one other labelling function.

Upon running the following code, you will obtain a table providing an analysis of how your labelling functions performed relative to each other. 
```python
from snorkel.labeling import LFAnalysis

LFAnalysis(L_unlabelled, lfs).lf_summary()
```

#### 'get_label_buckets' 

Snorkel provides some more evaluation tools to help you understand the quality of your labelling functions. In particular, 'get_label_buckets' is a handy way to combine labels and make comparisons. For more information, read the [Snorkel documentation](https://snorkel.readthedocs.io/en/v0.9.1/packages/_autosummary/analysis/snorkel.analysis.get_label_buckets.html).

The following code allows you to compare the true labels (y_gold) and predicted labels (y_preds) to view data points that were correctly or incorrectly labelled by Snorkel. This will allow you to pin-point which data points are difficult to correctly label, so that you can fine-tune your labelling functions to cover these edge cases.  

_Note, for this analysis we went back and created a L_train matrix which contains our labelling function predictions for our 'gold' labelled dataset._

```python
from snorkel.analysis import get_label_buckets
import pandas

buckets = get_label_buckets(y_gold, y_preds)

# eg. obtain datapoints with true label of 1 and predicted label of 1
df_labelled.iloc[buckets[(1, 1)]].sample(10, random_state=1)

# eg. obtain datapoints with true label of 1 and predicted label of -1 (abstain)
df_labelled.iloc[buckets[(1, -1)]].sample(10, random_state=1)

```

Alternatively, you can use 'get_label_buckets' to make comparisons between labelling functions. The following code allows you to compare the label predictions in L_unlabelled to observe how different labelling functions label datapoints differently. 

```python
from snorkel.analysis import get_label_buckets
import pandas

# Remeber that lfs was defined as follows:
    # lfs = [ms_severity_decimal_lf, ms_severity_word_lf, msbc_lf, logreg_tfidf_lf]
# eg. obtain data points where 'ms_severity_decimal_lf' abstained but 'ms_severity_word_lf' predicted 1  
buckets = get_label_buckets(L_unlabelled[:, 0], L_unlabelled[:, 1])

df_unlabelled.iloc[buckets[(-1, 1)]].sample(10, random_state=1)
```

### Step 5: Deployment 

#### Choosing the Best Labelling Model to Label Unlabelled Data

Following the procedure outlined above, we developed various labelling functions based on key-word searches, baseline models, and our MS-BERT classifier. We experimented with various ensembles of labelling functions and used Snorkel's Label Model to obtain predictions for a held-out labelled dataset. This allowed us to determine which ensemble of labelling functions would be best to label our unlabelled dataset. 

As shown in the table below, we observed that the MS-BERT classifier (MSBC) alone outperformed all ensembles that contain itself by at least 0.02 on Macro-F1. The addition of weaker heuristics and classifiers consistently decreased the ensemble's performance. Furthermore, we observed that the amount of conflict for the MS-BERT classifier increased as weaker classifiers and heuristics were added to the ensemble. 

![](https://i.imgur.com/vrSJp1t.png)
_Note, Rule Based (RB) refers to our key-word searches. LDA refers to linear discriminant analysis. TFIDFs refer to all models built ontop of tf-idf features (i.e. logistic regression, linear discriminant analysis, and support vector machines)._

To understand our findings, we have to remind ourselves that Snorkel's label model learns to predict the accuracy and correlations of the labelling functions based on agreements and disagreements amongst each other. Therefore in the presence of a strong labelling function, such as our MS-BERT classifier, the addition of weaker labelling functions introduces more disagreements with the strong labelling functions and therefore decreases performance. From these findings, we learned that Snorkel may be more suited for situations where you _only_ have weak heuristics and rules. However, if you already have a strong labelling function, developing a Snorkel ensemble with weaker heuristics may compromise performance. 

Therefore, the MS-BERT classifier alone was chosen to label our unlabelled dataset. 

#### Semi-Supervised Labelling Results

The MS-BERT classifier was used to obtain 'silver' labels for our unlabelled dataset. These 'silver' labels were combined with our 'gold' labels to obtain a silver+gold dataset. To infer the quality of the silver labels, new MS-BERT classifiers were developed: 1) MS-BERT+ (trained on silver+gold labelled data); and 2) MS-BERT-silver (trained on silver labelled data). These classifiers were evaluated on a held-out test dataset that was previously used to evaluate our original MS-BERT classifier (trained on gold labelled data). MS-BERT+ achieved a Macro-F1 of 0.86238 and a Micro-F1 of 0.92569, and MS-BERT-silver achieved a Macro-F1 of 0.82922 and a Micro-F1 of 0.91442. Although their performance was slightly lower that our original MS-BERT classifier (Macro-F1 of 0.88296, Micro-F1 of 0.94177), they still outperformed the previous best baseline models for MS severity prediction. The strong results of MS-BERT-silver helps show the effectiveness of using our MS-BERT classifier as a labelling function. It demonstrates potential to reduce tedious hours required by a professional to read through a patient's consult note and manually generate MS severity scores.

# Thank You!
Thanks for reading everyone! If you have any questions please do not hesitate to contact us at nlp4health (at gmail dot) com. :)

# Acknowledgements

We would like to thank the researchers and staff at the Data Science and Advanced Analytics (DSAA) department, St. Michael’s Hospital, for providing consistent support and guidance throughout this project. We would also like to thank Dr. Marzyeh Ghassemi, and Taylor Killan for providing us the opportunity to work on this exciting project. Lastly, we would like to thank Dr. Tony Antoniou and Dr. Jiwon Oh from the MS clinic at St. Michael’s Hospital for their support on the neurological examination notes.