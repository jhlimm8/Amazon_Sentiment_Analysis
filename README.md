# Amazon-Sentiment-Analysis-with-Stacking-Classifier
Sentiment Analysis on Amazon Review Dataset using a Stacking Classifier made of an Optuna-tuned MLP Classifier, a Random Forest Classifier, and a Multinomial Naive Bayes Classifier as Base Estimators with a Logistic Regression Classifier as the final estiamator.  An example of using the model to predict a sentence after training would be:

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
In the output above, we see that the model predicted [1] i.e. a positive sentiment, as opposed to [0] a negative sentiment.
