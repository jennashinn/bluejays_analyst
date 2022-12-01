#### **Jenna Shinn** 
#### **Blue Jays Analyst Questionnaire**

___
 **1. Predict the chance of a pitch being put in play. Please use this model to predict the chance of each pitch in the “deploy.csv” file being put in play and return a csv with your predictions.**
 
 see deploy_model.csv

**2. In one paragraph, please explain your process and reasoning for any decisions you made in Question 1.**  

After performing basic EDA, I started with a DecisionTreeClassifier and a RandomForestClassifier. The only thing I did to the training dataset was remove any null values. With both models having low accuracy (60%) and very low precision and recall, I went through and thoroughly “preprocessed” the dataset- handled outliers, dealt with class imbalance, and scaled the data. After preprocessing the training dataset, the RandomForest model had 75% accuracy, 73% in precision, and 78% in recall. I saved this model, then took the deploy dataset, removed any null values, and scaled the variables. Once the deploy dataset was prepared, I applied the RandomForest model and saved the predictions as the “InPlay” column.
  
**3. In one or two sentences, please describe to the pitcher how these 4 variables affect the batter’s ability to put the ball in play. You can also include one plot or table to show to the pitcher if you think it would help.**

HorzBreak is the only variable with a positive correlation (thought slight) to InPlay. Since there is a direct correlation, the pitcher should decrease HorzBreak if he is concerned that his fastball is put into play too often, or increase the HorzBreak if he'd like the fastball to put into play more often. 

![](https://lh3.googleusercontent.com/pw/AL9nZEV5m7rIC-3RRvp6APztrl2RhGaF_7xD6ay454wQDsocQvfuSGDjVZwnWcdpfGVZ2BOauEWEHgckXGxU_oTvcob6ULzsBTtQIf9V1aAiBnp-ykwIY7pI6kPeHeh-E4URJpvH5UPumhMGOtlQGQ0xzAqv1ZoPs02bLXnLsq54jnUZHfzFX_5J8X2bpO12kULXKXlmGexxlQ7-voayVsoO7Wi6PQuGCGBy2s8trGJZNAr2xTzbfd-KaaGvyzdbs-owqTVLMjxaNhAasyzJMbYyVRYWt5zdcVJNK0PrJYqpCUopEE4ql55b60fpd4fMQvoo-Qkvl1wsYj0fJmmZ0RylbKoCDefUN7fblNgZHfURfjkZelHviT5_7LDsAXKV2ealacM7N_XWiukOoJeDvhYahOPlRwACb7xWQTEzv67fV6amNdC7Vd39RWS5sTea6cYrVu8bAbzL9CEjmvgA5BfePxJF1Yon-eKWukx-KV8hBDzMgBd4Mfpky-vNV4-GMfZtrWvH4v5WAhth6EB1yVb0zSdGzGJLESUyucqURgacJoS_sFP7-IRB8ZmxEB7dafdidDSTGYxms3B4AL04UqDrIU1-RVg814b_QAaeBlB4P9MIRGff8IOjp1U3P5YDvQYE-D7I8qeSr-EJ_Y1pCiPeZbZClAKOfmngxZwi12nnOehGS5mUvgXx0GOL2ILQLp5Bxl9Rn-cVG57tb8ybxqctUk80w1fvKJbYVZNHaYDZBO9bnzbQFivCXngAg50l0oQ7DOJeCDMX3U2LZ9c3rtUDq-ILzF8Q9BIqud0ClqqN-HIOP-JMaoR46a6j26OTt-sy8hxBdW7wN_jUmbulOhPwGWlGx5iicC2N1XnnUrrMTf_3PirR_hkY9YyxW7PGEhgXNyK_DPf4xmrPZI6T3DPSEbwSWJTVs37wFqUhaWAR7CUxmtxVrurTQUIJRedQfS0v65ye537AJKYg9euXq5XNKpO2OPeLPl1fpqOxrMR7DpbIGzNVLt7kHxSUhY4qHI6tFhOc_TCEQ8NzGGcZ5PRMp7YxV7JrjebdGEg=w1046-h750-no?authuser=0)
![enter image description here](https://lh3.googleusercontent.com/pw/AL9nZEXqalTwKB1sLaGCQqgYDcRTd2mmRJcEDA-VBFMwrDIO6vZAD6l1Q2yUTEnJoyNzRYB0ypx76CVNVwmhijuGOkL3P0tVc446WRGfiBNs3D7LlskOVXgZayqIR_FeLUlTifU64P2F8ytR_0yDptp8kguaJn2cXMiGVQjx8pGZAW4tqt_REODutCjcUe0fe0QYOGC9GZ4GbSBUE1WNrdhgU3bwUTjv-o6FLkQuKpu8op9A3bTem6JOZOGcgsSKJOmC672VOqfQv89MdyI7_uzBp9h2IFKTtPfMehKu3U4Cxvy9JvDZ_L9DPWXpVhXwLEW4Sr6S8m6EbDbaiN_9LdOTDMWjWSp9ImcuChBtoPuWoPH2s-GC_64qGTHlmQUX6wTofX-tEcS7jEVDrVYE3CTUmOIc7xlL40-vYbipK6Lbi2ICA16SdV2A28YKno-3vbFYfqafw24GfXuvgn1Elirj1PdNuNvlHmVCTZ7Sc5pkyUaHrtScpZtjgkA-FTBPoWxOaMDym-hvJQOkPojsI1ipzQnmKOgc6RDtVf9YxLACXOwda05GZ_aYzuGIEnjVig6mGK6IM_A4JmwdjhIoyv-TdOAAgiPNdyaM2B3pj9gcOd4luRdzKjYpcWkR6bZaGMBQ57WriDTrj5L9lbgfC7JT1Eig72mWAtvhXg32IW4FN_pQZJIigCy93zQao8d724Kx5HNKs4ko54JIMlUAPxYwfHtgFeeAiKIGnLLwwN6RKHr5speP_BcyXFkQPHxvJA342WXjdgOmpsWeccoPA7If5Tt4-6zLrloo1j8bxsz506W8TDytF8INpNtrYKzHEeuxbQrq1k5INw4dut4iODIenO6Zco_iEkE4rOZ0nw0_zYNYWH3erCMvYZMdaVQAosfnjr72g7Rh1sEtVpkdjBtpzh_piGaMY3IqM8Ls8FDBZEyrq1p4EvsBihe8fSwFB6R_FhNUKQ3lWEXmgS7as5jMI6eqV3J0NTXbf1hn7Ca9u1ZZ6D6dtHLQrx8YjTIzPvnMuem1haNi4UrSrTLTOlpx24MZzsIXPSQN7V0=w1046-h748-no?authuser=0)

**4. In one of two sentences, please describe what you would see as the next steps with your model and/or results if you were in the analyst role and had another week to work on the question posed by the pitcher.**

My next step would be feature engineering and trying to see if I can create a new variable that has a higher correlation with InPlay. Preferably, something that relates to where the ball crosses the plate since a pitch in the strike zone is more likely to at least get a swing. 

**5. Please include any code (R, Python, or other) you used to answer the questions. This code doesn’t need to be production quality or notated.**
	see AnalystCode.py

