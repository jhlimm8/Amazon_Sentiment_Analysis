# Amazon-Sentiment-Analysis-with-Stacking-Classifier
Sentiment Analysis on Amazon Review Dataset using a Stacking Classifier made of an Optuna-tuned MLP Classifier, a Random Forest Classifier, and a Multinomial Naive Bayes Classifier as Base Estimators with a Logistic Regression Classifier as the final estiamator.  An example of using the model to predict a sentence after training would be:

exampletext = 'taylor swift is truly the greatest artist of our generation!!!!'

prep_exampletext = tfidf.transform((([Lemmatizer(Stemmer(Tokenizer(exampletext)))])))

print(prep_exampletext)
print(best_mnbc_model_test.predict((prep_exampletext)))
