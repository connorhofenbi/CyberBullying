# CyberBullying

<h3> Project Description </h3>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As a research project with Rutgers University I have been attempting to increase fairness in machine learning using cyber bullying data. Twitter data on cyberbullying was collected, including usernames, social connections, comments, etc. This data was then labeled as cyber bullying or not (one for cyber bullying or zero for not cyber bullying). The current goal of the project is to increase fairness in machine learning accuracies over stratas. For example if more data on introverts was collected than extroverts then the model would predict better for introverts. By increasing fairness I am trying to get the accuracies closer for every strata. The data is split into two stratas by finding the mean interactions between other users and any user below this threshold is classified as "introvert" and any user above is an "extrovert". Classification algorithms are then applied and evaluated, ROC area is used as the metric for how well the classifier performs and is both output and graphed for each algorithm and strata.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The Java project currently uses "Weka" libraries to implement multiple machine learning concepts; these concepts include cross validation splits for creating training and testing sets programatically and industry standard classification methods.

<h3> About the Data </h3>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The twitter data had a total of 25 columns, 20 features were chosen to train off of. Only 20 were chosen because 1 is the output feature and 4 were textual, such as comments and usernames. One of the columns was "Bully or Not" this was a 1 or 0 representation of if the comment was considered to be cyber bullying or not. The data was converted from an excel format to an ARFF format which Weka works well with. When imported into code the data had a 10 fold cross validation split performed to split the data into training and testing sets.

# <h3> Classification </h3>
The project currently classifies using the following algorithms :
  * J48 Decision Trees
  * Random Forest
  * Bagging 
  * Naive Bayes
  * ZeroR (baseline)
  
