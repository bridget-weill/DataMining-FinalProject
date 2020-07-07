# __Online Shopper's Intentions__
#### BA540 - Final Report
##### Bridget Weill
---------
E-commerce has exploded over the past few years and has created immense potential in the market, but the fact that the conversion rates have not increased at the same rate introduces the need for e-commerce companies to explore solutions that present customized promotions to the online shoppers. In traditional brick and mortar retailing, a salesperson can offer a range of customized alternatives to shoppers based on the experience he or she has gained over time. This experience has an important influence on the effective use of time, purchase conversion rates, and sales figures. Many e-commerce companies invest in early detection and behavioral prediction systems which imitate the behavior of a salesperson in virtual shopping environment. Additionally, there are academic studies that address the problem from different perspectives using machine learning methods. While some of these studies deal with categorization of visits based on the user’s navigational patterns, others aim to predict the behavior of users in real time and take actions accordingly to improve the shopping cart abandonment and purchase conversion rates. In this project, the focus is to product a real-time online shopper behavior analysis system.

[Link to Reference](https://www.semanticscholar.org/paper/Real-time-prediction-of-online-shoppers%E2%80%99-purchasing-Sakar-Polat/747e098f85ca2d20afd6313b11242c0c427e6fb3)

## Analytic Focus
#### What drives potential customers to make a purchase?

*Additional Research Questions:*
1. What are the most common traits that lead to a purchase on the site?
2. What does the monthly/seasonal data tell us?
3. How can the findings be applied to a real world problem?
4. Which modeling technique works best with the dataset?
5. Which variable predicts Revenue better, Weekend or Special Day?

## Exploratory Data Analysis

*This is an overview of the exploratory data analysis, use this link to see the EDA notebook for further details: [Link to EDA](ExploratoryDataAnalysis.ipynb)*

The EDA consists of descriptive statistics, data separation, visualizations, binning, encoding & correlation of the data before further preparation. 

__Data Dictionary:__ Defintions of the variables & further understanding of the dataset used throughout this analysis can be found [here](DataDictionary.ipynb)

#### Descriptive Statistics
- The dataframe has 12330 rows and 18 columns
- There are no missing values
- There are 6 columns with a median of 0, meaning at least half the values are 0
- The datatypes of the columns vary:
    - _int64:_ 7 columns
    - _float64:_ 7 columns
    - _object:_ 2 columns
    - _bool:_ 2 columns

#### Data Separation
- Data was seperated into continuous & categorical dataframes as follows:
 
<table>
<tr><td>

|__Continuous__|
|:------:|
|Administrative|
|Administrative_Duration|
|Informational|
|Informational_Duration|
|ProductRelated|
|ProductRelated_Duration|
|BounceRates|
|ExitRates|
|PageValues|

</td><td>

|__Categorical__|
|:-------------:|
|SpecialDay|
|Month|
|OperatingSystems|
|Browser|
|Region|
|TrafficType|
|VisitorType|
|Weekend|
        
</td></tr> </table>

- The __target is revenue__; also seperated into another dataframe

#### Visualizations
*Continuous Data:*
- Histograms:
    - Histograms help visualize the distribution of the values among the columns along with the the skewness.
    - The histograms on this dataset show that many columns contain low values and and are skewed. It is clear that the data is not balanced, having a majority of zero vales.
   
<img width="350" alt="Screen Shot 2020-05-05 at 3 41 32 PM" src="https://user-images.githubusercontent.com/54870844/81108634-fc9d2200-8ee6-11ea-876f-66edf3d15c3c.png"> <img width="350" alt="Screen Shot 2020-05-05 at 3 41 46 PM" src="https://user-images.githubusercontent.com/54870844/81108729-10488880-8ee7-11ea-9f24-9389c3cf3a81.png"> <img width="180" alt="Screen Shot 2020-05-05 at 3 41 51 PM" src="https://user-images.githubusercontent.com/54870844/81108763-1a6a8700-8ee7-11ea-9770-2adca123ea43.png">

- Skewness:
    - There was a lot of skewness in the data before performing any data preparation.
    - Skewness should be in the range of -.5 and .5, and all of the results are out of this range, they are all above 1.9 as seen below:
       ```
        Administrative             1.960357
        Administrative_Duration    5.615719
        Informational              4.036464
        Informational_Duration     7.579185
        ProductRelated             4.341516
        ProductRelated_Duration    7.263228
        BounceRates                2.947855
        ExitRates                  2.148789
        PageValues                 6.382964
        dtype: float64
        ```
- Box Plots:
    - Box plots visualizes if there are any outliers in the data and the range of the values.
    - As seen in the EDA file, there are many values outside of plot. 
    - As seen below, in the box plot of all the values, it is clear that ProductRelated_Duration is altering the results due to the large value range, leading us to seperate the data into smaller sectioned box plots to see the results better (see EDA notebook for the seperated box plots)
    
    <img width="600" alt="Screen Shot 2020-05-05 at 3 45 48 PM" src="https://user-images.githubusercontent.com/54870844/81109058-85b45900-8ee7-11ea-8e50-3dcf676c101c.png">

*Categorical Data:*
- Bar Graphs:
    - Bar graphs visualize categorical data through the distribution of bins
    - As seen in the EDA, there is a bar graph for each categorical value in order for us to analyze each variable individually.
    - Key Takeaways:
        - *Months:* 
            - Top 4 largest bins are May, November, March & December
            - Smallest bin is February
            - January and April are not seen in the results.
            - The months data will help further investigate the analysis question asking if the seasons have an impact on a customers purchases.
             <img width="544" alt="Screen Shot 2020-05-05 at 3 50 34 PM" src="https://user-images.githubusercontent.com/54870844/81109583-4e927780-8ee8-11ea-997f-87d44155c1c3.png">
        - *VisitorType:*
             - Visitor Type is made up of 3 bins: Returning_Visitor, New_Visitor & Other
             - Largest bin by far is Returning_Visitor, having over 10,000 counts, followed by New_Visitor having just under 2,000 counts.
             <img width="539" alt="Screen Shot 2020-05-05 at 3 50 49 PM" src="https://user-images.githubusercontent.com/54870844/81109681-74b81780-8ee8-11ea-9da0-90648f49ce42.png">
        - *Weekend:*
             - Weekend has a count of over 8,000 & Weekday has a count of under 4,000, meaning more online shopping is going on during the week than on the weekend.
              <img width="530" alt="Screen Shot 2020-05-05 at 3 50 56 PM" src="https://user-images.githubusercontent.com/54870844/81109721-84cff700-8ee8-11ea-9feb-04377b074a1d.png">
        - *Revenue:*
             - The target value, Revenue, has a much larger count of False over True.
             - False has a count of over 10,000 & True has a count of almost 2,000.
             - The majority of the time, there is no revenue from made from online shopping.
             <img width="534" alt="Screen Shot 2020-05-05 at 3 51 02 PM" src="https://user-images.githubusercontent.com/54870844/81109754-94e7d680-8ee8-11ea-87dc-cb5745cf85b7.png">
         
#### Binning
As stated previously in the Months visualization analysis, the month data will help in determining if season has an impact on purchases. To take this a step further, the months were binned according to the seasons, this will help minimize the data for modeling and focus on seasonal impacts. The following table represents the bins created for the seasons by month:

| Value | Season | 
| ------------ | ---------- | 
| 1 | Winter ('Dec', 'Jan', 'Feb') |
| 2 | Spring ('Mar', 'Apr', 'May') |
| 3 | Summer ('Jun', 'Jul', 'Aug') |
| 4 | Fall ('Sep', 'Oct', 'Nov') |

After creating these bins, the results were visualized with a bar graph to see how the seasons were distribuuted. The results showed that Spring had the most visits throughout the year, followed by Fall, Winter, and then Summer with by far the least. Spring and Fall had much higher counts than the winter and summer.

<img width="530" alt="Screen Shot 2020-05-05 at 3 51 25 PM" src="https://user-images.githubusercontent.com/54870844/81109947-e001e980-8ee8-11ea-937b-69d6f5b548fa.png">

#### Encoding
Through OneHotEncoding, a categorical feature becomes an array with a size that is the number of possible choices for that features. By using one hot encoding, the categorical values can be evaluated more evenly without the values being taken into effect. For example, in region, region 1 and region 8 will not have different results due to their categorical values being 1 and 8. Through encoding, the categorical dataset grew a huge amount, going from 8 features to 65 features. Now, instead of having 1 column for region with values 1-8, there are 8 columns for region, one for each value. For region 1, the values in the column will be 1 where the region is 1 and 0 where the region is anything other than 1; this is how every feature is now represented.
The following table represents the new features from encoding:


|Original Feature|SpecialDay|OperatingSystems|Browser|Region|TrafficType|VisitorType|Weekend|SeasonBins|
|:---------------|----------|---------|-----|-----|-----|-----|----|----|
|__Encoded Features__| SpecialDay_0.0<br/>SpecialDay_0.2<br/> SpecialDay_0.4<br/>SpecialDay_0.6<br/>SpecialDay_0.8<br/>SpecialDay_1.0<br/>|OperatingSystems_1<br/>OperatingSystems_2<br/>OperatingSystems_3<br/>OperatingSystems_4<br/>OperatingSystems_5<br/>OperatingSystems_6<br/>OperatingSystems_7<br/>OperatingSystems_8|Browser_1<br/>Browser_2<br/>Browser_3<br/>Browser_4<br/>Browser_5<br/>Browser_6<br/>Browser_7<br/>Browser_8<br/>Browser_9<br/>Browser_10<br/>Browser_11<br/>Browser_12<br/>Browser_13|Region_1<br/>Region_2<br/>Region_3<br/>Region_4<br/>Region_5<br/>Region_6<br/>Region_7<br/>Region_8<br/>Region_9|TrafficType_1<br/>TrafficType_2<br/>TrafficType_3<br/>TrafficType_4<br/>TrafficType_5<br/>TrafficType_6<br/>TrafficType_7<br/>TrafficType_8<br/>TrafficType_9<br/>TrafficType_10<br/>TrafficType_11<br/>TrafficType_12<br/>TrafficType_13<br/>TrafficType_14<br/>TrafficType_15<br/>TrafficType_16<br/>TrafficType_17<br/>TrafficType_18<br/>TrafficType_19<br/>TrafficType_20|VisitorType_New_Visitor<br/>VisitorType_Other<br/>VisitorType_Returning_Visitor|Weekend_False<br/>Weekend_True|SeasonBins_1<br/>SeasonBins_2<br/>SeasonBins_3<br/>SeasonBins_4|

As seen in this table, it is clear that the number of features in the categorical dataset grew greatly. This will increase the number of features being dealt with in the feature selection step of modeling, but overall, the accuracy of the results will be much stronger.

[Encoding Reference](http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example)

#### Correlation
A correlation matrix is used to visualize the correlation between the continuous features. 
- The highest correlation in the dataset is between ExitRates & BounceRates with .913. Closely followed by ProductRelated & ProductRelated_Duration with .861.
- Another interesting point is that the highest correlations can be seen between a variable and its corresponding duration variable. (Administrative & Administrative_Duration, Informational & Informational_Duration, ProductRelated & ProductRelated_Duration)
- The lowest correlation is between ExitRates & Administrative with -0.316.

Correlation is a vital step in the process, if a feature is too highly coorelated with another, they should not be put back in the same dataframe. After putting the data through the preparation steps, I will finish again by looking at a correlation matrix to determine if any features should be seperated. 

#### CSV Files
The final step in the exploratory data analysis is converting these dataframes into CSV files. These CSV files will be used for the next, data preparation. 

*This is an overview of the steps along with some of the key takeaways from the exploratory data analysis, to see the full analysis with more in depth details along with the code, use this link to the EDA notebook: [Link to EDA](ExploratoryDataAnalysis.ipynb)*

## Data Preparation

Data Preparation consists of the preproccesing of the data, handling outliers, standardization & normalization.
These steps were aranged differently in three pipelines. Each pipeline created different reults, each pipeline result will be used to evaluate the data through the mdoels. 

The three pipelines (linked to their corresponding notebooks) are as follows:

[Pipeline 1](Pipeline1.ipynb) : Normalization -> Z Score -> 3 STD

[Pipeline 2](Pipeline2.ipynb) : IQR -> Min Max -> Normalization

[Pipeline 3.1](Pipeline3.1.ipynb) : IQR -> Normalization -> Z Score

[Pipeline 3.2](Pipeline3.2.ipynb) : Min Max -> Normalization -> 3 STD

After completing the pipelines, the seperated dataframes were joined back together and exported into a csv. These csv's were then used to evaluate the models.

Below are links to the notebooks describing these steps in the pipelines along with their codes:

[Normalization Link](Normalization.ipynb) - [Z Score Link](ZScore.ipynb) - [3 STD Link](3STD.ipynb) - [IQR Link](IQR.ipynb) - [MinMax Link](MinMax.ipynb)

__Alterations:__

While creating the pipelines, it was clear that due to the data being imbalanced, removing the outliers through IQR had a negative effect on the outcomes. 
After removing the outliers through IQR, the data still looks unbalanced.

By looking further into the results after removing outliers with IQR, '.describe()' shows that both the min and max for some of the features are 0. That being said, the only descriptive data left for these features is count of values, meaning the IQR converted everything else to 0. Through looking at histograms, it is clear that the reason for IQR converting all the values to 0 for these 3 varaibles is because in the original data, almost all of the variables are 0. The data being so imbalanced is the reason the IQR was taken out of pipeline's data preparation.

## Data Models

The final dataset CSV files created from pipeline 1, pipeline 2, pipeline 3.1, and pipeline 3.2 were each ran through the models. 
#### Logistic Regression
- Logistic regression is common classification model and is a useful regression method for solving the binary classification problem.
- Logistic regression is a statistical method for predicting binary classes. The outcome (target variable) is dichotomous in nature (only 2 possible classes).

*Logistic Regression was ran through each pipeline, these pipelines and their model codes be found through the following links:*

[Pipeline1-LogisticRegression](1LogisticRegression.ipynb) - [Pipeline2-LogisticRegression](2LogisticRegression.ipynb) - [Pipeline3.1-LogisticRegression](3.1LogisticRegression.ipynb) - [Pipeline3.2-LogisticRegression](3.2LogisticRegression.ipynb)

[Logistic Regression Reference](https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python)

#### Random Forest
- Random Forest is a flexible and easy to use algorithm. 
- A forest is comprised of trees; it is said that the more trees it has, the more robust a forest is. 
- Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting.

*Random Forest was ran through each pipeline, these pipelines and their model codes be found through the following links:*

[Pipeline1-Random Forest](1RandomForest.ipynb) - [Pipeline2-Random Forest](2RandomForest.ipynb) - [Pipeline3.1-Random Forest](3.1RandomForest.ipynb) - [Pipeline3.2-Random Forest](3.2RandomForest.ipynb)

[Random Forest Reference](https://www.datacamp.com/community/tutorials/random-forests-classifier-python)

#### ADA Boost
- ADA Boost combines multiple classifiers to increase the accuracy of classifiers, it is an iterative ensemble method. 
- A strong classifier is built by combining multiple poorly performing classifiers in order to a get high accuracy strong classifier. 
- Adaboost sets up the weights of classifiers and trains the data sample in each iteration so it ensures the accurate predictions of observations.

*ADA Boost was ran through each pipeline, these pipelines and their model codes be found through the following links:*

[Pipeline1-ADA Boost](1ADABoost.ipynb) - [Pipeline2-ADA Boost](2ADABoostt.ipynb) - [Pipeline3.1-ADA Boost](3.1ADABoost.ipynb) - [Pipeline3.2-ADA Boost](3.2ADABoost.ipynb)

[ADA Boost Reference](https://www.datacamp.com/community/tutorials/adaboost-classifier-python)

#### Decision Trees
- Decision Trees can be used for both classification and regression problems. 
- Structured like a flowchart, this model lays out the steps (decisions) taken in determining a prediction or outcome. 
    - The tree splits on attributes, which in turn can become decision nodes(where the tree goes a level deeper) or stopping points(remain as attributes).
    
*Decision Tree was ran through each pipeline, these pipelines and their model codes be found through the following links:*

[Pipeline1-Decision Tree](1DecisionTree.ipynb) - [Pipeline2-Decision Tree](2DecisionTree.ipynb) - [Pipeline3.1-Decision Tree](3.1DecisionTree.ipynb) - [Pipeline3.2-Decision Tree](3.2DecisionTree.ipynb)

[Decision Trees Reference](https://scikit-learn.org/stable/modules/tree.html)

#### XG Boost
- XGBoost (Extreme Gradient Boosting) is a popular and effective machine learning algorithm that can be used for both classification and regression.
- Starts with a beginning [decision] tree, and “boosts” performance with each tree by sequentially penalizing and rewarding the predictions via weights.

*XG Boost was ran through each pipeline, these pipelines and their model codes be found through the following links:*

[Pipeline1-XG Boost](1XGBoost.ipynb) - [Pipeline2-XG Boost](2XGBoost.ipynb) - [Pipeline3.1-XG Boost](3.1XGBoost.ipynb) - [Pipeline3.2-XG Boost](3.2XGBoost.ipynb)

[XG Boost Reference](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)


#### SVM
- SVM is known for its kernel trick to handle nonlinear input spaces and can easily handle multiple continuous and categorical variables. 
- SVM constructs a hyperplane in multidimensional space to separate different classes.
- This was one of the best performing models as the data was well suited for SVM’s strengths.

*SVM was ran through each pipeline, these pipelines and their model codes be found through the following links:*

[Pipeline1-SVM](1SVM.ipynb) - [Pipeline2-SVM](2SVM.ipynb) - [Pipeline3.1-SVM](3.1SVM.ipynb) - [Pipeline3.2-SVM](3.2SVM.ipynb)

[SVM Reference](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)

## Model Evaluation

Each model was evaluated based on two evaluation metrics, AUC and F1 scores, which further helped us identify the best models.
- __F1 score:__ *Prediction Accuracy:* Measured how close the predicted values are to the factual outputs.
- __AUC score:__ *Predictive Power:* The 'Area Under ROC Curve' (hense AUC) predicts the performance of the model.

For both evaluation metrics, the focus is to get the values closest to 1.

## SMOTE

As seen through the analysis, the data is very imbalanced, meaning classes are not represented equally. This imbalanced data can cause the performance of models to be poor. In order to make the data more balanced, the Synthetic Minority Oversampling Technique (SMOTE) is implemented into the models.

The results to these models were first evaluated without SMOTE, then I implemented SMOTE into the training data (*should never be used on the test data*). The results prove that by adding SMOTE, the predictive power of the model increases.

The results of each model before and after SMOTE can be seen in the results: [Link to Results](Results.md)

[SMOTE Reference](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)


## Feature Selection

Feature selection is used to filter out insignificant features from the dataset in order to enhance the performance of a model. One of the final steps taken in the exploratory data analysis was encoding, which resulted in the dataset increasing greatly in size, growing from 18 features to 76 features. With all of these features, it can be assumed that some of them are insignificant to the results. Through feature selection, the features that go into the models are controlled and then it is decided which grouping of features acquires the best overall results. Through feature selection, the machine learning algorithm trains faster, the model is less complex and easier to understand, the model is more accurate, and overfitting is reduced. 

[Feature Selection Reference](https://www.datacamp.com/community/tutorials/feature-selection-python)

After determining the top features for each specific model and pipeline, the next step was finding which grouping of the features presents the best outcome. The feature selection for each model was different, the code and selection techniques by model are explained further below...

#### Logistic Regression
- __Method:__ *RFE*
    - Through recursive feature elimination (RFE), the goal is to select features by recursively considering smaller and smaller sets of features.
    - I put each pipeline through the RFE selector and by changing the number features, I was able to run different sets of features through the models until the best evaluation scores were found.

[RFE Feature Selection Reference](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)

[RFE Logistic Regression Feature Selection Code](RFELogisticRegFeatSelector.ipynb)

#### ADA Boost
- __Method:__ *Scikit Attribute Within Model*
    - Scikit-learn performs feature selection through the feature importance attribute feature.
    - After creatin the ADA model, the feature importance attribute was used to see feature importance scores, and finally visualized the scores through seaborn library.
    - This feature selection code can be seen within the modeling notebooks
    
[ADA Boost Feature Selection Reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=ada#sklearn.ensemble.AdaBoostClassifier)

#### Random Forest
- __Method:__ *Scikit Attribute Within Model*
    - Scikit-learn performs feature selection through the feature importance attribute feature.
    - Similarly to ADA Boost, I first created a random forests model, then used the feature importance attribute to see feature importance scores, and finally visualized the scores through seaborn library.
    - This feature selection code can be seen within the modeling notebooks

#### SVM
- __Method:__ *Select K Best*
    - For certain pipelines, the F CLassif hyperparameter was used since there are negative values in the data.  

[SVM Feature Selection Reference](https://stats.stackexchange.com/questions/341332/how-to-scale-for-selectkbest-for-feature-selection)

#### Decision Tree
- __Method:__ *Self-performing*
    - When optimizing the tree(s), there was a splitting threshold. Entropy was chosen as this measures the information gain from each split. This was important because it fed the most important information into the model instead of noise.

#### XG Boost
- __Method:__ *Self-performing*
    - After the initial run of the model, the features were ranked based on their importance in the model. They were manually entered into the optimized model by declaring these top features as the feature columns.

Feature selection was ran both with and without SMOTE in order to see the changes within the results.

## Top Performing Models

Through evaluating the results of each model, the overall best results per model were determined. These top models and their results are shown in the table below.
<table>
<tr><th>Top Model Results</th></tr>
<tr><td>
    
|Model|F1 Score|AUC|Pipeline|SMOTE|Feature Selection|
|----:|:------:|:-:|:------:|:---:|:-------:|
|__Logistic Regression__|0.6603|0.8208|3.1|Yes|Yes|
|__ADA Boost__|0.6593|0.8352|3.2|Yes|Yes|
|__Random Forest__|0.6636|0.8261|2|Yes|Yes|
|__SVM__|0.6446|0.8553|1|Yes|Yes|
|__Decision Tree__|0.6474|0.8356|1, 2, 3.2|Yes|Yes|
|__XG Boost__|0.6537|0.8212|2|Yes|Yes|

</td></tr></table>

Through these results, it is clear that SMOTE had a positive affect on the results; the best models all came from using SMOTE. All of the final models have an AUC greater than .6400 and a F1 score over .8200, which overall are pretty good results. They also all increased from selecting features.

*The full results for each model can be seen in the results file: [Link to Results](Results.md)*

## The Best Model

As seen in the table in the previous section, the models evaluation scores are all very similar. I first looked into the models with the highest F1 score, which were Random Forest with a score of 0.6636, followed by Logistic Regression with a score of 0.6603. Next, I looked at the highest AUC scores, which were seen in SVM with a score of 0.8553 followed by Decision Tree with a score of 0.8356. With the scores being so close and the none of the top model evaluation scores belonging to one model, I had to look further.

Though Decision Tree had the second highest AUC score, ADA Boost was a stronger overall model fit. In comparision to Decision Tree, ADA Boost had a lower AUC score by only 0.004, and a higher F1 score of 0.6593, as compared to Decision Tree with an F1 score of 0.6474. By narrowing down the top 4 models to Random Forest, Logistic Regression, SVM, & ADA Boost, I looked into more evaluation metrics, accuracy & precision. The following table shows the results:

|Model|Accuracy|Precision|
|-----|--------|---------|
|Random Forest|0.8832|0.5992|
|Logistic Regression|0.8657|0.5383|
|SVM|0.8551|0.5170|
|ADA Boost|0.8756|0.5727|

From these results, the conclusion is made that Random Forest was the best model for the data. It showed consistant results throughout the modeling for each pipeline, and in the best model, it had the highest accuracy and highest precision compared within the top 4 models.

#### __The Best Model :__ Random Forest

*The code and evaluation of this model can be found in the following notebook: [Random Forest Best Model](2RandomForest.ipynb)*

## Adjusting the Hyperparameters

Tuning the hyperparameters for the Random Forest model helps to to increase the F1 and AUC score of this model even further. Through a randomized search,  the model was ran through the hyperparameters to determine which ones would create the best fit. Through this randomized search, the following hyperparameters were determined:
         
         'n_estimators': 48,
         'min_samples_split': 5,
         'max_leaf_nodes': 37,
         'max_features': 0.8999999999999999,
         'max_depth': 18,
         'bootstrap': True
         
Using these hyperparameters in the model resulted in a small decrease of the F1 score to 0.6513 and an increase of the AUC to 0.8369.

*This code can be found in the following notebook: [Random Forest Best Model](2RandomForestBridget.ipynb)*

[Hyperparameter Code Reference](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

## Results

The overarching question of this analysis is, what drives potential customers to make a purchase?
Through feature selection, the top F1 and AUC scores were found. The number of features put into the model was 15. These top 15, strongest features are shown in the bar graph below...

<img width="572" alt="Screen Shot 2020-05-05 at 2 19 51 PM" src="https://user-images.githubusercontent.com/54870844/81101153-b55d6400-8edb-11ea-8846-287ff9fe5850.png">
        
These top features have the strongest impact on if buyers make a purchase or not. 

#### Page Value

As seen in the results, PageValues, by far, is the number 1 driver for a customer potentially making a purchase. Page values is defined as the 'average value for a webpage that a user visited before making a purchase'. Companies should focus on improving mobility between pages to encourage users to browse among different products. 

#### Exit Rate
Exit rate is the second most impactful feature influencing if a customer will make a purchase or not. If customers are leaving a page, it should be looked further into; why are exit rates higher for one page over another? how can these rates improve?

#### Product Related Duration & Product Related
Both Product Related Duration and Product Related are top 5 drivers for predicting purchases. This shows that when a customer is searching to buy a potential product, they are open to looking into related products as well.

#### Administrative Duration & Administrative
Administrative duration takes number 5 in the top five, while administrative related follows closely at spot 7. Both being top 7 features, it is clear that the time spent on administrative pages impacts a buyers purchases. Looking into making these pages more successful will create overall more success.

#### Other Interesting Finds
- _Season:_ Fall is the only season that made it to the top 15 features. As seen in the EDA analysis, the largest bin was the spring season, due to this, it was assumed that the Spring would be the most common time to purchase. After doing this analysis, it proves that much more goes into modeling than just counts, it must be taken into consideration that other features and variables have an influence on one another.
- _Weekend vs Weekday:_ More purchases are made on weekdays than weekends (14th most impactful feature). Being that there are more days in the week, I wondered if this had an impact on the purchasing being higher on weekday, or maybe if being at work on a computer made it easier for people to online shop? There was not enough information in the dataset to come to a conclusion to these questions, but it is definitely something to further investigate.

## Recommendations

__Increase Page Value:__ Optimize the website so that high traffic pages bring value to sales strategy

__Improve Exit Rate %:__ Improve Exit Rates for pages that should be leading to more purchases but aren’t performing as well as expected

__Product Related Pages:__ Whether or not a purchase occurred can be predicted by how many product related pages a user visited

__Administrative Pages:__ Make pages containing user information user friendly

__Focus on Key Seasons:__ The Fall was proven to be an important factor in predicting a purchase

## Page Values Impact

With page value being the top feature by a large margin, it was removed to see how the model responded (visualization shown below). Through comparing these two graphs, it is clear that removing page value has created much closer gaps between the top features importance values. 

<img width="470" alt="Screen Shot 2020-05-05 at 2 19 33 PM" src="https://user-images.githubusercontent.com/54870844/81101406-14bb7400-8edc-11ea-851c-15fee3da0daa.png"> <img width="450" alt="Screen Shot 2020-05-05 at 2 20 46 PM" src="https://user-images.githubusercontent.com/54870844/81101440-2735ad80-8edc-11ea-851b-23795466298e.png">

An important find here is that some of the top features were altered when page value was removed.IWe assumed the features would each just move up one spot when page value was removed, but this did not end up being the case for all of them. The top 1 through 6 features remained as expected, but following feature 7 is now Informational which is followed by SeasonBins_4, these features switched places when page values was removed, making SeasonBins_4 less impactful now. The other change was with Region_3; this feature moved up 2 spots when page values was removed, bumping Browser_2 and Weekend_False down a spot. Finally, the additional feature added to the top 15 is Weekend_True. 

When page values was removed, the F1 and AUC scored dropped a great amount, by adding more features and selecting important features, the highest the F1 and AUC scores came to were 0.3733 and 0.6310 respectively. In order to increase the scores to these values, only the top 10 features were used, when previously the top 15 were used. By removing page values, the influence of the other features was decreased, and the overall model was much less accurate making it clear how important this feature was the the outcome of the model. The positive that came from removing pagevalue, was being able to visualize the top features on a more equally distributed scale.  

*The full analysis and code of the model with Page Values removed can be found in the following notebook: [Random Forest Page Value Removed](Alterations2RandomForest.ipynb)*

## Final Key Takeaway

This analysis proves one key point to be true for online business success; __simplicity__. Customers want an easy to navigate website without any extras making it more difficult to shop. By focusing on related product recommendations, it can keep a customer shopping and looking at different options. The users want admistrative pages to be easy to naviagate and understand, spending time trying to understand these pages is frustrating and companies can lose business due to them. To conclude, by improving the overall user friendliness and simplicity of a webpage, the chances of purchases being made are much higher.

## Taking This Analysis Further

This dataset provided a ton of key points regarding purchases by online shoppers, but there are a few things missing that could have furthered this analysis. Some obvious data to add would be knowing what some of the numbered categorical features really represented; such as region, browser, & operating system. Some other additions are as follows...
- Days of the week: 
    - This can help with narrowing down popular days in order to plan sales and when to release new products
- Type of Product being purchased:
    - By collecting product data, this analysis can increase greatly, companies can use specific product data to analyze how their online sales are affected
- Product Prices:
    - Adding prices can help seperate the data among more expensive companies vs cheaper companies. Does price influence a shopper, or will a shopper buy what they are looking for not matter the price? 
- Sales & Discounts:
    - Are customers more influenced to buy when there is a sale or discount on an item? 
    
Theses are just a few steps that can be taken to take this analysis further. By adding more data, it will help to understand the way customers shop even more and companies can even select specific data to relate it to their sales. The more data there is to work with, the further this analysis can be taken. 