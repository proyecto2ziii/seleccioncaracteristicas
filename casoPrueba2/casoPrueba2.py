'''
'srch_id','prop_id','position','prop_location_score1','
prop_location_score2','prop_log_historical_price','
price_usd','promotion_flag','orig_destination_distance','
prop_country_id','prop_starrating','prop_review_score','
prop_brand_bool','count_clicks','avg_bookings_usd','
stdev_bookings_usd','count_bookings','year','month','
weekofyear','time','site_id','visitor_location_country_id','
srch_destination_id','srch_length_of_stay','
srch_booking_window','srch_adults_count','
srch_children_count','srch_room_count','
srch_saturday_night_bool','random_bool'
'''

# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

adult_df = pd.read_csv('EHhold10re.csv')

adult_df.columns = ['srch_id','prop_id','position','prop_location_score1',
					'prop_location_score2','prop_log_historical_price',
					'price_usd','promotion_flag','orig_destination_distance',
					'prop_country_id','prop_starrating','prop_review_score',
					'prop_brand_bool','count_clicks','avg_bookings_usd',
					'stdev_bookings_usd','count_bookings','year','month',
					'weekofyear','time','site_id','visitor_location_country_id',
					'srch_destination_id','srch_length_of_stay','srch_booking_window',
					'srch_adults_count','srch_children_count','srch_room_count',
					'srch_saturday_night_bool','random_bool']

adult_df_rev = adult_df
print (adult_df)
#print(adult_df_rev)


num_features = ['srch_id','prop_id','position','prop_location_score1',
				'prop_location_score2','prop_log_historical_price',
				'price_usd','promotion_flag','orig_destination_distance',
				'prop_country_id','prop_starrating','prop_review_score',
				'prop_brand_bool','count_clicks','avg_bookings_usd',
				'stdev_bookings_usd','count_bookings','year','month',
				'weekofyear','time','site_id','visitor_location_country_id',
				'srch_destination_id','srch_length_of_stay','srch_booking_window',
				'srch_adults_count','srch_children_count','srch_room_count',
				'srch_saturday_night_bool','random_bool']

scaled_features = {}

features = adult_df_rev.values[:,:30]
target = adult_df_rev.values[:,30]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.33, random_state = 10)

#target_train = list(target_train)

print(target_train)
'''
clf = GaussianNB()
clf.fit(features_train, target_train)

target_pred = clf.predict(features_test)

print(target_pred)
print(accuracy_score(target_test, target_pred, normalize = True))
'''