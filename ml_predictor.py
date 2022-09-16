from extract import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def ML_predictor(train, test):
    # Based on bag of words, predict label using multinomial naive bayes classifier.

    pipe = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB())])
    fitted_pipe = pipe.fit(train['text'], train['labels'])

    test_predictions = fitted_pipe.predict(test['text'])

    correct_predicts = 0 # setting a counter to compare the prediction and targets. this can probably be done neater without loops.
    for (guess,target) in zip(test_predictions, test['labels']): 
        if guess == target:
            correct_predicts += 1 

    accuracy = correct_predicts/len(test_predictions)
    print('\n The accuracy of this classifier on the test set was', round(accuracy*100, 2), '% \n')
    return(fitted_pipe)

def User_input(pipe):
    print("Now it's your turn to test the classifier! \n Enter a sentence and see what it is classified as.",
    'If you want to stop, simply write "STOP" and hit enter.')
    

    while True:
        test_input = [input("Enter a sentence (and hit enter)\n\n")]
        if test_input == "STOP":
            break
        else:
            print(pipe.predict(test_input))


dialog_acts = 'data/dialog_acts.dat' # if we change the dataset, we don't need to change the functions by defining it here
train, test = create_dataset(dialog_acts)

pipe = ML_predictor(train,test)
User_input(pipe)

