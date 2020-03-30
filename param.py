import numpy as np
import pandas as pd
import csv

dataset = pd.read_csv("zeemee_train_binary_qwerty.csv")

from sklearn.model_selection import train_test_split
x = dataset.drop('final_funnel_stage',axis=1)
y_labels = dataset['final_funnel_stage']
assert not np.any(np.isnan(x)) #makes sure there are no missing values in the data

X_train, X_test, y_train, y_test = train_test_split(x, y_labels, test_size=0.2, random_state=101)

import tensorflow as tf

college = tf.feature_column.numeric_column("college")
public_profile_enabled = tf.feature_column.numeric_column("public_profile_enabled")
going = tf.feature_column.numeric_column("going")
interested = tf.feature_column.numeric_column("interested")
start_term = tf.feature_column.numeric_column("start_term")
cohort_year = tf.feature_column.numeric_column("cohort_year")
created_by_csv = tf.feature_column.numeric_column("created_by_csv")
last_login = tf.feature_column.numeric_column("last_login")
schools_followed = tf.feature_column.numeric_column("schools_followed")
high_school = tf.feature_column.numeric_column("high_school")
transfer_status = tf.feature_column.numeric_column("transfer_status")
roommate_match_quiz = tf.feature_column.numeric_column("roommate_match_quiz")
chat_messages_sent = tf.feature_column.numeric_column("chat_messages_sent")
chat_viewed = tf.feature_column.numeric_column("chat_viewed")
videos_liked = tf.feature_column.numeric_column("videos_liked")
videos_viewed = tf.feature_column.numeric_column("videos_viewed")
videos_viewed_unique = tf.feature_column.numeric_column("videos_viewed_unique")
total_official_videos= tf.feature_column.numeric_column("total_official_videos")
engaged = tf.feature_column.numeric_column("engaged")


feat_cols = [college, public_profile_enabled, going, interested, start_term, cohort_year, created_by_csv, last_login, schools_followed, high_school, transfer_status, roommate_match_quiz, chat_messages_sent, chat_viewed, videos_liked, videos_viewed, videos_viewed_unique, total_official_videos, engaged ]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=None,shuffle=True)
model = tf.estimator.DNNClassifier(feature_columns=feat_cols,hidden_units=[5,10], model_dir='stanford_models')
for i in range(30):
  model.train(input_fn=input_func,steps=5000)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
predictions = list(model.predict(input_fn=pred_fn))
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])
from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))


#my_stats = [4,4.8,1600,0.99]

#data = pd.DataFrame({'UW': [my_stats[0]], 'W': [my_stats[1]], 'SAT': [my_stats[2]],
#                     'Rank': [my_stats[3]]})

#pred_fn = tf.estimator.inputs.pandas_input_fn(x=data,num_epochs=1,shuffle=False)

#import numpy as np
#pred_gen = list(model.predict(input_fn=pred_fn))

#likelyhood = pred_gen[0]['logistic']
#likelyhood = round(likelyhood[0]*100,2)
#print('\n')
#print ("You are "+str(likelyhood)+"% likely to get into Stanford")
