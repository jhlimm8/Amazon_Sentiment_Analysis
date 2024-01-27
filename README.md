# Amazon-Sentiment-Analysis
Sentiment Analysis on an Amazon Review Dataset using different classifiers. Three base classifiers are trained, an Optuna hyperparameter-tuned MLP Classifier, a Random Forest Classifier, and a Multinomial Naive Bayes Classifier. A Logistic Regression Model is then used to synthesize the results of the three models to create an ensemble Stacking Classifier.  An example of using one of the models to predict a sentence after training would be:

```
exampletext = 'taylor swift is truly the greatest artist of our generation!!!!'

prep_exampletext = tfidf.transform((([Lemmatizer(Stemmer(Tokenizer(exampletext)))])))

print(prep_exampletext)
print(best_mnbc_model_test.predict((prep_exampletext)))

Output:
  (0, 43994)	0.39724507077922017
  (0, 42228)	0.4496084885731648
  (0, 41765)	0.5007290572366171
  (0, 19071)	0.41019299944016696
  (0, 18116)	0.3681514655156302
  (0, 4915)	0.29244687317950285
[1]
```
In the output above, the model transformed the input into a sparse embedding vector with 6 non-zero terms. The model then predicted [1] i.e. a positive sentiment, as opposed to [0] a negative sentiment.
