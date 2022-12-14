# Project Title

* **Summary** This repository holds an attempt to apply regression models to predict enzyme stability using data from
"Novozymes Enzyme Stability Prediction" Kaggle challenge (https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction). 

## Overview

* challenge:
   The task, as defined by the Kaggle challenge is to predict the thermostability of the enzymes variant using the experimentally measured thermostability which is the melting point data 
    that includes natural and engineered sequences with single or multiple mutations upon the natural sequence. 
*Approach:
   The approach in this repository formulates the problem as regression task, using regression model with protien sequence as input. I compared the performance of 3 different regressors 
   namely "Random forest regressor", "Linear regression", and "Least Angle regression" 
 *Summary of the performance achieved** Our best model was able to predict the melting point with the least value of RMSE

## Summary of Workdone:
The given protien sequence was converted to numerical values to make it suitable for regression models. Each letter (A,B etc) couts for every sequence are provided as features or predictors
and the melting point is the target. I used training set for the validation as the given test set didn't have any target column. To analayze and conclude, prediction I used RMSE, MAE, MSE
as metric. 

### Data

* Data:
  * Type: 
    * Input: CSV file of features (Protien sequence), output: Meltingpoint(Target).
  * Size: 16.36 Mb
  * Instances (Train, Test, Validation Split): 20349 amino_sequence for training, taining set was used for validation.

#### Preprocessing / Clean up

* target was histogrammed
* outliers were detected
* eliminated the outliers
* converted alpahbetic categories to numerical
* listed number of letters and made them my feeatures

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input: Protien sequence
  * Output: Melting point
  * Models
    * Random forest regressors, Linear regression, Least angle regression. Reason: to determine the strenght of the prediction by comparing the models based on perfomance evaluaton metrics
    * Hyperparameter tuning

### Training

* Describe the training:
  * How you trained: software- Anaconda, Ide- Jupyter notebook and hardware- Personal laptop.
                   : randomserachCv was used to find the best parameters to improve model
                   : model was fit based on the best case of hyperparameter tuning(best value of hyperparameters)
  * How long did training take- more than 20 mins
  * limited the hyperparameter due to less computation power.or else grid search
  * No difficulties

### Performance Comparison

* key performance metrics: RMSE, MSE, MAE
                         : best model determined based on the least value produced by RMSE among the models (better strenght in the prediction)
* Show/compare results in one table.
*

### Conclusions

* Random forest regressor works  better than Linear regression and Least angle regression in this case.

### Future Work

* I would try using the same approach with neuarl networs to come out with even more clear insights .
* What are some other studies that can be done starting from here: Reasearchers who does artificial protien sequencing can use the combinations of sequnece
                                                                  that produced best thermostability 

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
            pandas
            matplotlib
            seaborn
            numpy


### Data

* Point to where they can download the data:  https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data

#### Performance Evaluation

* create a function that incorporates all the metric tasks and use it to evaluate the performance


## Citations

* https://scikit-learn.org/stable
*https://www.kaggle.com/code/pragyanbeuria/enzyme-stability-prediction-rand-forest-approch
*https://towardsdatascience.com/protein-sequence-classification-99c80d0ad2df
