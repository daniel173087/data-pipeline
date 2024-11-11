# Disaster Response Pipeline Project

### Summary
This project shows the process of cleaning data and using this data in a pipeline. Data is used from Disaster Response Unit. These things then are added to a flask webpage to show interesting data and classify new text messages.‚

## Code structure
.
├── .gitattributes
├── .gitignore
├── README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── ETL Pipeline Preparation.ipynb
│   ├── ML Pipeline Preparation.ipynb
│   ├── code_from_etl_notebook.py
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── for-mentor
│   ├── process_data_output.png
│   └── train_classifier_output.png
└── models
    ├── code_from_notebook.py
    └── train_classifier.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Code functionality
On my local machine any script runs correctly. I put photos of my Terminal in the folder 'for-mentor' for reference.
I don't know what to change in order to run it on the machine of the mentor.
