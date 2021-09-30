# XAI-emotion
Code repository for MSc Artificial Intelligence individual project

**Instructions for running the code**

There are 8 Jupyter Notebooks (see list below), each of them contains specific running instructions in the first cell.

There is a link to a zipped file on OneDrive containing 4  model weights and 2 pre-created LIME and SHAP objects. These are not available on GitHub due to file size constraints.

There are 4 py module files containing functions that are imported by the notebooks

There are two training history numpy files which are loaded by the notebooks

FER data is contained in the folder 'data'. The original source for this is https://www.kaggle.com/msambare/fer2013

text_emotion_cleaned.csv contains the pre processed data for the NLP model

text_emotion.csv is the uncleaned NLP data sourced from https://www.kaggle.com/manuelbenedicto/figure-eight-labelled-textual-dataset

The notebooks are:

1. FER 2013 VGG Face Model Training and Evaluation FINAL.ipynb  - FER Model (FER 2013 dataset)

2. FER 2013 LIME FINAL.ipynb - see the LIME explanations for FER 2013 

3. FER 2013 SHAP FINAL.ipynb - see the SHAP explanations for FER 2013

4. FELTD Data Pre-Processing FINAL.ipynb - Text Model (FELD Dataset) - see how the pre-processing was done

5. FELTD Embedding Bag Model FINAL.ipynb - Embedding Bag Model (FELD Dataset) 

6. FELTD BERT Model Final.ipynb  - BERT Model (FELD Dataset) 

7. FELTD Embedding Bag Model LIME Explanations FINAL.ipynb  - see the LIME explanations for FELTD

8. FELTD BERT Model SHAP Explanations FINAL.ipynb  - see the SHAP explanations for FELTD


Each notebook contains specific running instructions in the first cell. Observe the outputs found in the project preloaded, or train your own models from scratch and create your own SHAP and LIME explainations from scratch.
