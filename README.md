### Table of Contents

1. [Project Summary](#summary)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [How to Run](#how)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Summary<a name="motivation"></a>

This project aims to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

To test, I have used a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that, posteriorly, people can send the messages to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3. However, if you need to know which libraries are used, here they are: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re and pickle.

## File Descriptions <a name="files"></a>

On app folder, there is run.py, a flask web app that takes care of data visualizations using plotly and uses the model from the pickle file to classify messages inputted by the user. Inside this folder, there is another folder (templates), with the web pages for the main page (master.html) and for the classification results page (go.html).

On data folder, there is process_data.py that loads the messages and categories datasets, merges the two datasets, cleans the data and stores it in a SQLite database.

On models folder, there is train_classifier.py that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set and exports the final model as a pickle file.

## How to Run<a name="how"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Open another Terminal Window and run the following command.
    `env|grep WORK`
    You will see an output with values for spacedomain and spaceid. Use these values to go to https://SPACEID-3001.SPACEDOMAIN in a web broswer.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Feel free to use the code here as you would like! And if you have any feedback, please share
