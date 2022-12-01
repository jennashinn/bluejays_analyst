**Jenna Shinn** 
**Blue Jays Analyst Questionnaire**
___
1.  Predict the chance of a pitch being put in play. Please use this model to predict the chance of each pitch in the “deploy.csv” file being put in play and return a csv with your predictions.  
  
2. In one paragraph, please explain your process and reasoning for any decisions you made in Question 1.  
	> After performing basic EDA, I decided to start with a RandomForestClassifier. The only thing I did to the training dataset was remove any null values. With that model having low accuracy (60%) and very low precision and recall, I went through and thoroughly “preprocessed” the dataset- handled outliers, dealt with class imbalance, and scaled the data. After preprocessing the training dataset, model has 75% accuracy, 73% in precision, and 78% in recall. I then took the deploy dataset, removed any null values, then scaled the variables. After that, I applied the RandomForest model and saved the predictions as the “InPlay” column.
  
3. In one or two sentences, please describe to the pitcher how these 4 variables affect the batter’s ability to put the ball in play. You can also include one plot or table to show to the pitcher if you think it would help.
> If the pitcher is looking to avoid putting the ball in play, he should focus on the horizontal break of the ball. HorzBreak is the only variable with a slightly positive correlation with InPlay (ie, higher HorzBreak means more likely to be put into play).


4. In one of two sentences, please describe what you would see as the next steps with your model and/or results if you were in the analyst role and had another week to work on the question posed by the pitcher.

> My next step would be feature engineering and trying to see if I can create a new variable that has a higher correlation with InPlay.

5. Please include any code (R, Python, or other) you used to answer the questions. This code doesn’t need to be production quality or notated.
