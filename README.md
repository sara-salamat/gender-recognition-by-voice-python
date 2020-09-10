# gender-recognition-by-voice-python

This is the project I did as the final project of 'Pattern Recognition' course. The dataset used in this project was collected from 293 people including 155 women and 138 men.

There is a short description for each part/code:

**Note**

 - Since our data was collected from different people in different environments, it needed to be cleaned. So, we filtered some frequencies in feature extraction part.
 
 ## analysis
 
 Some statiscal analysis was done on our data, like distribution of year of birth, sex and etc. .
 
 ## extract
 
 Some features were extracted from voices using this script. We used [this article](https://github.com/sunilpankaj/Gender-Recognition-by-Voice-using-python/blob/master/Gender%20Recognition%20by%20voice.ipynb) for generating features.
 
 ## gender recognition
 
 We used support vector machines for classification. Our accuracy was 93.6%.
 
 ## clustering
 
 We clustered our data with different number of clusters.
