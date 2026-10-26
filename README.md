# Flight Delay Prediction

This project predicts the chance that a flight leaving Charlotte Douglas International Airport (KCLT) will be delayed, using aviation weather data. Weather delays can often be spotted hours ahead from the same data pilots use, so the goal is to warn travelers earlier than the airline usually does.

Group 10
Parker Hall, Chris Loor-Luna, Adithya Raman, Sonia Sun

## How it works

We pull weather observations (METAR) and store them in Amazon S3. For training, we match each historical flight from the BTS On-Time Performance database to the last weather reading before its scheduled departure, and label it delayed if it left 15 or more minutes late (the DOT's own cutoff). A random forest is trained on that table, and then it can score current weather and give a delay probability. We also check its accuracy over time and retrain if it drops.

Everything is in one notebook: flight_delay_prediction.ipynb

## How to install

You need Python 3 and these libraries:

- requests
- pandas
- numpy
- matplotlib
- boto3
- joblib
- scikit-learn

Install them with:

```
pip install -r requirements.txt
```

or directly:

```
pip install requests pandas numpy matplotlib boto3 joblib scikit-learn
```

AWS is optional to get started. Without credentials the code saves and reads weather files from the local data/ folder instead of S3. If you want to use S3 and CloudWatch, set up a bucket and an IAM user, run `aws configure`, and then set S3_BUCKET and AWS_REGION at the top of the notebook. Don't put your keys in the notebook.

## Get started

This trains the model on one month of data.

1. Go to the BTS On-Time Performance database (https://www.transtats.bts.gov/DL_SelectFields.asp?gnoyr_VQ=FGJ) and download one month of flights for CLT. Make sure the columns FL_DATE, ORIGIN, CRS_DEP_TIME, DEP_DELAY and CANCELLED are included.
2. Put the CSV in a data/ folder next to the notebook and name it like T_ONTIME_REPORTING-January.csv.
3. Open flight_delay_prediction.ipynb and run it top to bottom.

The notebook loads your flight file, works out the date range, downloads the matching historical weather from the Iowa Environmental Mesonet, builds the features, trains the model, and prints accuracy, precision and recall. The trained model is saved to models/delay_model.joblib.

To get a prediction for right now:

```python
live_weather = collect_weather_data()
latest = live_weather.iloc[-1].to_dict()
print(predict_delay(latest, model))
```

There is also example code in the bin/ folder.

## Main functions

- collect_weather_data(): gets recent METAR observations for KCLT from the Aviation Weather Center API.
- collect_historical_weather_data(): gets older weather for a date range from the IEM archive, and converts the units to match the live data (Celsius and hPa).
- store_data_s3(): uploads weather data to S3 as a timestamped JSON file, or saves it locally if S3 isn't set up.
- load_data_from_s3(): reads all the saved weather files back into one table.
- load_flight_delay_data(): loads the BTS CSVs, keeps only flights leaving CLT, drops cancelled flights, and adds the delayed label.
- preprocess_data(): matches each flight to the closest earlier weather reading (within 2 hours), builds the features, and fills in missing values.
- train_model(): splits the data 80/20, trains the random forest, prints the metrics, saves the model, and plots feature importances.
- predict_delay(): takes one weather reading and returns the probability of a delay.
- monitor_model(): checks accuracy on new data and sends it to CloudWatch if it can.
- retrain_model(): retrains on new data and only keeps the new model if it's better.

## Model

A scikit-learn RandomForestClassifier with 200 trees and max depth 6. The features are temperature, dewpoint, wind speed, visibility, altimeter setting, and flags for low visibility, wind gusts and reported weather conditions like rain or storms. Delayed flights are the minority, so we look at precision and recall and not just accuracy.

## Monitoring and retraining

monitor_model() logs a ModelAccuracy number to CloudWatch. If it falls below 0.70, retrain_model() runs. BTS data comes out about two months late, so this is meant to be checked monthly when new data is released.

## Limitations

- Only works for departures from KCLT right now.
- The weather match uses the closest reading within 2 hours, which could be finer.
- CloudWatch gets the metric but no alarm or alert (like SNS) is set up yet.

## Questions

Contact: [Parker Hall: phall19@charlotte.edu , Chris Loor-Luna: cloorlun@charlotte.edu, Adithya Raman: araman1@charlotte.edu, Sonia Sun: ssun8@charlotte.edu]

## License

MIT License, see the [LICENSE](LICENSE) file.
