# Project Title

* **Summary**:  This repository holds an attempt to apply regression models to predict enzyme stability using data from
"Novozymes Enzyme Stability Prediction" Kaggle challenge (https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction). 

## Overview

* challenge:
   The task, as defined by the Kaggle challenge is to predict the thermostability of the enzyme variants using the experimentally measured thermostability which is the melting point data 
    that includes natural and engineered sequences with single or multiple mutations upon the natural sequence. 

*Approach:
   The approach in this repository formulates the problem as regression task, using regression model with protein sequence as input. I compared the performance of 3 different regressors 
   namely "Random forest regressor", "Linear regression", and "Least Angle regression" 
   
 *Summary of the performance achieved** Our best model was able to predict the melting point with the least value of RMSE

## Summary of Workdone:
The given protein sequence was converted to numerical values to make it suitable for regression models. Each letter (A,B etc) counts for every sequence are provided as features or predictors
and the melting point is the target. I used training set for the validation as the given test set didn't have any target column. To analayze and conclude, prediction I used RMSE, MAE, MSE
as metric. 

### Data

* Data:
  * Type: 
    * Input: CSV file of features (Protein sequence), output: Melting point(Target).
  * Size: 16.36 Mb
  * Instances (Train, Test, Validation Split): 20349 amino_sequence for training, taining set was used for validation.
  
  ![image](https://user-images.githubusercontent.com/112579358/207649679-a290be8f-854e-4bea-89b7-8d406b456fe3.png)


#### Preprocessing / Clean up

* target was histogrammed
* outliers were detected
* eliminated the outliers
* converted alpahbetic categories to numerical
* listed number of letters and made them my feeatures

#### Data Visualization

![image](https://user-images.githubusercontent.com/112579358/207647770-c8b884a2-abd1-4227-a702-e8faee0e4f23.png)

![image](https://user-images.githubusercontent.com/112579358/207647987-6164e144-ec63-4669-89b1-c5b15d2d59a4.png)

![image](https://user-images.githubusercontent.com/112579358/207648390-f805349a-7b02-4b24-a6e1-0115dd465385.png)



### Problem Formulation

* Define:
  * Input: Protein sequence
  * Output: Melting point
  * Models
    * Random forest regressors, Linear regression, Least angle regression. Reason: to determine the strength of the prediction by comparing the models based on perfomance evaluaton metrics
    * Hyperparameter tuning

### Training

* Description of training:
  * How I trained: software- Anaconda, Ide- Jupyter notebook and hardware- Personal laptop.
                   : randomserachCv was used to find the best parameters to improve the model
                   : model was fit based on the best case of hyperparameter tuning(best value of hyperparameters)
  * Time take for training- more than 20 mins
  * limited the hyperparameter due to less computation power or else grid search would have been used 
  ![image](https://user-images.githubusercontent.com/112579358/207648845-4080ac47-bab0-4f38-9c49-e3d830ff18c2.png)

  * No difficulties

### Performance Comparison

* key performance metrics: RMSE, MSE, MAE
                         : best model determined based on the least value produced by RMSE among the models (better strenght in the prediction)
 
 ![image](https://user-images.githubusercontent.com/112579358/207649021-16d13feb-244d-4a1b-acbc-92dd72178c5b.png)
 
 ![image](https://user-images.githubusercontent.com/112579358/207649168-a9be410e-9a4a-4035-9448-9465bd5e1e82.png)

![image](https://user-images.githubusercontent.com/112579358/207649296-b14a5ec7-d5fe-4841-afa4-e726b5253e21.png)




### Conclusions

* Random forest regressor works  better than Linear regression and Least angle regression in this case.

### Future Work

* I would try using the same approach with neuarl networs to come out with even more clear insights .
* What are some other studies that can be done starting from here: Reasearchers who does artificial protien sequencing can use the combinations of sequnece
                                                                  that produced best thermostability to bring out an enhanced performance 


### Software Setup
* List all of the required packages.
            *pandas
            *matplotlib
            *seaborn
            *numpy
            *scikit learn


### Data

* download the data using this link:  
    https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data

#### Performance Evaluation

* create a function that incorporates all the metric tasks and use it to evaluate the performance


## Citations

* https://scikit-learn.org/stable

*https://www.kaggle.com/code/pragyanbeuria/enzyme-stability-prediction-rand-forest-approch

*https://towardsdatascience.com/protein-sequence-classification-99c80d0ad2df
