#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:12:16 2021

@author: kieranshaw
"""

import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import six

# =============================================================================
# ########################### OVERVIEW & ANALYSIS ############################
# =============================================================================

# GOAL: "Design an algorithm to grade quality of contact on the attached data set of batted balls." 

# STEPS: 
#    1. Data Importation & Exploration
#        a. Basic Statistical Analysis, Correlations, Inspection, Plotting, etc. 
#    2. Data Cleaning 
#        a. Null Values, Nonsensical Inputs, Categorical Encoding (might not be necessary for final model), Dealing with Outliers (z-score), etc. 
#    3. Model Building 
#        a. Model Building (kmeans, density) - Includes discussion of methodology and design decisions 
#        b. Model Selection & Validation - Scoring of Clustering Models, Evaluating Cluster Labels for Contact "Grade", & Labeling Clusters w/ Grade

# Initial Thoughts: After looking at the dataset and the variables - while taking into account the goal - I will be implementing unsupervised 
# clustering (kmeans, density, hierachical). An initial question that I have right now - will the score improve or decrease when I include 
# the non-numerical variables (inning, pitcher-handedness, play_result). In current baseball thought, the variables which are most important 
# for evaluating quality of contact are launch angle, exit velocity, and distance. But, bearing could prove important here.
# Taking into account the result of the play (homerun, single, etc) is not a controllable. Thus, when evaluating quality of contact, 
#it makes sense to use variables/inputs which depends strictly on the act of hitting the ball (exit velocity, launch angle, direction, bearing). 
# In addition, intuitively, I do not think that pitcher handedness, hitter handedness, pitch type, outs, balls, or strikes are going to be 
# important features in the model. I want to grade the quality of contact, I am not attempting to grade the quality of contact situationally 
# (count, runners on, or even pitch type). Doing so would eliminate the standard method of analysis throughout the dataset, and it would most 
# likely result in unintelligable results. Although I have no doubt that the clustering algorithms can handle the inputs, I beleive that they 
# are not necessary. We can examine this through some correlations. Finally, I think it is important to note that in addition to gathering a 
# "score" for the clustering  - which evalutes how succesfully the data was clustered - I am going to be labeling the clusters with a contact grade. 
# I will do this in two ways. First: some basic graphical representations of each observations (each hit) important characteristics/variables; 
# this will show how each cluster differ in terms of results that we care about. This will be creating some average performance scores for each 
# variable in the cluster to draw some distinctions. Second, I will take the original independent variables and run regressions against the 
# clusters while interacting the variables against one another.
    


# =============================================================================
# ########################## IMPORTING & EXPLORATION ##########################
# =============================================================================

all_bip = pd.read_csv('//Users/kieranshaw/Desktop/Giants Analyst/all_bip_raw.csv', index_col='RowNum')

# Doesn't look like there is much cleaning to do with nulls, etc. Will most likely have bulk of cleaning with categorical data. 
# This looks like a Trackman dataset that has been cleaned to include only batted balls - implies that data dictionary can be pulled from Trackman 
# Defining Tricky Columns: 
#   pa_of_inning = Indication of the batting order of the half inning where the first batter of the inning is assigned a 1, the second batter a 2, etc. 
#   tagged_pitch_type = Pitch classification (fastball, curveball, etc.) selected manually by system operator. NOTE: this is manual and not AutoPitchType
#   play_result = undefined happens ball not put in play (most likely foul out?) -----> check this out
all_bip.play_result.value_counts()
undefined = all_bip[all_bip.play_result == 'Undefined']
####       NOTE: deal with "Undefined" in data cleaning - will most likely drop all occurances where play_result = undefined. Also, saw some Nan values. 
#   exit_speed = The speed of the ball, measured in miles per hour, as it comes off the bat at the moment of contact
#   angle = How steeply up or down the ball leaves the bat, reported as an angle. A positive number means the ball is initially traveling upward, 
#       while a negative number means the ball is initially traveling downward. NOTE: this is from contact point, which differs by hitter. 
#   direction = Left-right (horizontal) direction in which the ball leaves the bat, reported as an angle. A negative number represents a ball 
#       initially traveling toward the third base side of second base while a positive number represents a ball initially traveling toward the 
#       first base side.
#   distance = The estimated “carry flat” distance, measured in feet, meaning the distance that the ball travels before it lands, 
#       or would have landed if it were not caught or obstructed.
#   bearing = Indicates where on the field the ball lands or would have landed, had it not been caught or obstructed. It is reported in degrees 
#       relative to home plate. A bearing of 0 degrees means the ball landed on a straight line from home through second base. A positive number 
#       means the ball landed on the first base side, while a negative number means the ball landed on the third base side.

### Lets do some basic descriptive statistics on some of the numerical variables. 

all_bip.balls.mean()
# 1.11
all_bip.strikes.mean()
# 1.09
all_bip.exit_speed.mean()
# 87.74
np.std(all_bip.exit_speed)
# 14.510003110767483

all_bip.angle.mean()
# 12.074525405370132
# all balls are initially travelling "upward" at around 12 degrees. 
all_bip.angle.median()
# 12.36
all_bip.angle.std()
#26.958108142550415 --- pretty high standard deviation here... 

all_bip.direction.mean()
# -1.0882117290636795
# the average of all balls put in play on this dataset inititally travel towards the third base side of second base - oversample of RHH? 
# median = -1.49 and mode = -0.74 and -0.23 

all_bip.distance.mean()
# 167.03402381320774 --- this really does not tell us much, ie: the average here describes the average of all batted balls, which doesn't tell us much 
# because outliers do not have a large effect. 

outfield_batted_balls = all_bip[all_bip.distance > 150] 
outfield_batted_balls.distance.mean()
# 287.82 --- this tells us a little bit more. Given that a batted ball flies more than 150 feet, the average of these occurances is 287 feet. 
all_bip.distance.std()
# 137.2226495305137 --- a high std. dev. makes sense here, and is to be expected. 
outfield_batted_balls.distance.std()
# 74.37945543044357 -- a lower one here makes sense, espeically as a lower bound has been set. 

all_bip.bearing.mean()
# -0.5936631131453084

#### Slicing by Exit Speed #####
exit_speed_large = all_bip[all_bip['exit_speed'] > 90]

## Describing Play Outcomes for Exit Velocity > 90 ##
exit_speed_large['play_result'].value_counts()
        # Out               40529
        # Single            17015
        # Double             7342
        # HomeRun            6609
        # FieldersChoice     1930
        # Sacrifice           901
        # Triple              708
        # Error               670
        # Undefined            42

exit_speed_large['hit_type'].value_counts()
        # GroundBall    22849
        # LineDrive     20479
        # FlyBall       17982
        # Popup           455
        # Undefined        10

exit_speed_large['angle'].describe()
        # count    61775.000000
        # mean        13.458075
        # std         18.911488
        # min        -82.160000
        # 25%          0.300000
        # 50%         13.410000
        # 75%         26.570000
        # max         86.790000


#######  DISCOVERING DIRECTION VS. BEARING  ########
# This discussion is best begun with a clarification of terms: BEARING deals with balls when they land or would have landed. This means that 
# the ball has completed its flight in the air and has either been caught of landed. DIRECTION deals with the initial flight direction of the ball
# at the moment of contact. Thus, there is a significant difference between direction and bearing. Bearing registers the slice of a ball. What I mean
# by slice here is that the bearing (at the end of ball flight) does not equal the direction (moment of contact). Thus, the ball has deviated from
# its initial direction as a result of spin, airflow, physics, etc. This is a relatively simple concept for a person with baseball knoweldge, 
# but it could potentially prove important for grading contact. If not that, it is cool to recognize in the data!  

# I am not going to dive into percentiles mainly because I am not expecting outliers, I am familiar with the range of these variables, and 
# I do not necessarily care -- at this point - if a bunch of my data is bunched around a specific number. For example, grading quality of contact
# will deal with a bunch of exit speeds at the average, etc. 

#### CHECKING SKEW AND KUROTSIS ##### 
# Note: I do not expect to need to scale or transform the data. I want the data in its origional form so that I can grade quality. 

all_bip.describe()
# WOAH - a 538 foot home run? Need to check that out for later. 

######## CORRELATION ####### 

# Number of balls in count vs. exit speed
all_bip['balls'].corr(all_bip['exit_speed'])
# = 0.07033769760185259

# Number of strikes vs. launch angle 
all_bip['strikes'].corr(all_bip['angle'])
# -0.02522528938804365  ---- a negative correlation, what we would expect but not that strong. 

# Exit_speed vs. distance
all_bip['exit_speed'].corr(all_bip['distance'])
# 0.3866936368344466 ---- looks a litttle bit better, more of something we would expect to see. the harder you hit it the further it is likelier to go. 

# What about this correlation with balls that travel further than 150 ft? 
outfield_batted_balls['exit_speed'].corr(outfield_batted_balls['distance'])
# .563877731964695 -- this makes even more sense. 
plt.scatter(outfield_batted_balls['exit_speed'], outfield_batted_balls['distance'])
plt.ylabel('distance')
plt.xlabel('exit speed')
plt.show()
# This is an incredibly instructive graphic - and super cool. Albeit, this is expected. 

all_bip['angle'].corr(all_bip['distance'])
# 0.6733447171030079 --- this says a lot right here and basically reinforces the launch angle revolution. Well, I want to be careful here. 
# This says that there is a relationship between the launch angle and the distance, and the goal of the player/org is to determine the optimal --
# which is what the launch angle revoltion should be about. Yes, increasing the launch angle produces more fly balls (and home runs), but 
# we should be teaching for optimal launch angle. This is fairly basic statistics work. 
# Lets look at this correlation. 
plt.scatter(all_bip['angle'], all_bip['distance'])
plt.ylabel('distance')
plt.xlabel('launch angle')
plt.show()

# EXAMPLE: Say we want to find the optimal launch angle for balls travelling at 95 or greater mph that are home runs. 

high_mph = all_bip[(all_bip['exit_speed'] >= 95) & (all_bip['play_result'] == 'HomeRun')]

# Lets say that the optimal launch angle is probably between the 25 and 75 percentile -- as a rough estimate. 
high_mph.angle.describe()
# Look at that - we have an optimal launch angle between 24.29 and 31.46 for balls hit 95 or greater that end up home runs. 

# We could do more work here if we wanted to -- we could change the angle to bins so that we could find the highest frequency of launch angle 
# home runs when the exit velocity is 95 or greater. 

# NOTE: it is pretty obvious, after running some basic correlations that what I anticipated is true regarding the categorical data.
# balls, strikes, pitcher handedness, etc, does not impact launch angle, etc. I could do more of this correlation when I clean up the 
# categorical data to get a better picture. This is important becuase if I can deem these variables unimportant, they should not be 
# in the clustering. 

# NOTE: there seem to be a few strange outliers here - some extremely large home run numbers... clean this up in data cleaning. 

# =============================================================================
# ############################## DATA CLEANING ###############################
# =============================================================================
## STEPS: 
    #   1. Deal with Nan values
    #   2. Deal with "underfined" and any other questionable areas.
    #########   n\a. Categorical --> Numerical (Dummy, encode with values, etc.) 
    #   3. Deal with outliers (like max distance, etc)  
    

########## NAN VALUES ###########

# First, I need to do some additional inspection for "nan" - where does it appear, how often, and can I figure out why? 
# Next, I need to decide if I want to drop these occurances (do-able if there is not a large occurace), or if I want to figure out a way 
# to change them to numerical values (unlikely) using imputed means, etc. 

all_bip.isnull().sum()
# There are 104 null values in exit_speed, angle, direction, distance, and bearing. makes sense to drop these rows as I do not want null values for any variables. 
all_bip.dropna(axis=0, inplace=True)


########## "UNDEFINED" VALUES ###########
# Lets first write a for loop to see how many "undefined" exist for each column with categorical data. 

categorical = all_bip.select_dtypes(include=['object'])
for col in categorical.columns: 
    print(col + ': ' + str((categorical[col] == 'Undefined').sum()))

        # batter_side: 0
        # pitcher_throws: 0
        # tagged_pitch_type: 465
        # hit_type: 30
        # play_result: 69

# Before dropping these I think its smart to note that I want to cleanest data set possible. So, I do not want any values with "undefined"
# as that may make it difficult to do cluster analysis. At this point, I am operting on the assumpting that dropping 465 rows of data 
# will not significantly impact cluster analysis because we have over 122,000 rows. 

# Lets use boolean indexing to drop all rows with "undefined" in tagged_pitch_type, hit_type, and play_result
all_bip = all_bip[(all_bip['tagged_pitch_type'] != 'Undefined') & (all_bip['hit_type'] != 'Undefined') & (all_bip['play_result'] != 'Undefined')]

# Lets check the work here by utilizing the same for loop as before, but with a new dataframe name so as not to confused. 

categorical_check = all_bip.select_dtypes(include=['object'])
for col in categorical_check.columns: 
    print(col + ': ' + str((categorical_check[col] == 'Undefined').sum()))

        # batter_side: 0
        # pitcher_throws: 0
        # tagged_pitch_type: 0
        # hit_type: 0
        # play_result: 0
        
        
########## DEALING WITH CATEGORICAL ###########
########## I am commenting this out after much more coding - not necessary to make these changes because I am not using the categorical in 
# cluters analysis. I will leave it here for viewing, but I am no longer using it. 
            # For cluster analysis - which is an unsupervised learning technique, I need to make sure that I do not have categorical data like I have here. 
            # Usually, a boolean, 0 or 1 type of categorical data is acceptable (like pitcher or hitter handedness). 
            # But, in this dataset there is also play result (more than 2 outcomes), pitch type (more than 2 outcomes), and hit type (more than 2 outcomes). 
            # Basically, I will need to create columns which mirror the cateogircal data but in a numerical way. This is fairly simple to do,
            # it is just extremely important to keep track of what each categorical variable corresponds to. 
            
            #  BATTER SIDE - I am going to change the columns in place, this means that instead of right and left there will be 1's and 0's (instead of new column)
            #all_bip['batter_side'] = np.where(all_bip['batter_side'] == 'Right', 1, 0)
            
            # PITCHER THROWS 
            #all_bip['pitcher_throws'] = np.where(all_bip['pitcher_throws'] == 'Right', 1, 0)
            
            # TAGGED PITCH TYPE 
            # Identify all the different possible outcomes for "tagged pitch type" first, then illustrate which pitch type will equal a value, then encode. 
            #all_bip['tagged_pitch_type'].value_counts()
                    # Fastball       53601 == 1
                    # Slider         20871 == 2 
                    # ChangeUp       15382 == 3 
                    # Sinker         10858 == 4
                    # Curveball       9372 == 5
                    # Cutter          8330 == 6
                    # Other           1972 == 7
                    # Splitter        1824 == 8 
                    # Knuckleball       29 == 9 
            
            # I could write a for loop here which would arguably be easier to read, BUT np.where is a pandas method function and that is MUCH faster. 
            # Granted, the nested statement can be confusing to read, but it works much faster and is similar to an excel if command. 
            #all_bip['tagged_pitch_type'] = np.where(all_bip['tagged_pitch_type'] == 'Fastball', 1, np.where(all_bip['tagged_pitch_type'] == 'Slider', 2, np.where(all_bip['tagged_pitch_type'] == 'ChangeUp', 3, np.where(all_bip['tagged_pitch_type'] == 'Sinker', 4, np.where(all_bip['tagged_pitch_type'] == 'Curveball', 5, np.where(all_bip['tagged_pitch_type'] == 'Cutter', 6, np.where(all_bip['tagged_pitch_type'] == 'Other', 7, np.where(all_bip['tagged_pitch_type'] == 'Splitter', 8, 9))))))))
            
            # HIT TYPE 
            # Follow the same sort of patter here - look at frequency of poissible outcomes and then encode. 
            #all_bip['hit_type'].value_counts()
                    # GroundBall    52160 == 1
                    # LineDrive     30715 == 2
                    # FlyBall       29003 == 3
                    # Popup          8866 == 4
                    # Bunt           1495 == 5
            
            #all_bip['hit_type'] = np.where(all_bip['hit_type'] == 'GroundBall', 1, np.where(all_bip['hit_type'] == 'LineDrive', 2, np.where(all_bip['hit_type'] == 'FlyBall', 3, np.where(all_bip['hit_type']  == 'Popup', 4, 5))))
            
            # PLAY RESULT 
            # Follow the same sort of patter here - look at frequency of poissible outcomes and then encode. 
            #all_bip['play_result'].value_counts()
                    # Out               74069 == 1
                    # Single            25276 == 2
                    # Double             8270 == 3
                    # HomeRun            6581 == 4
                    # FieldersChoice     4115 == 5
                    # Sacrifice          1854 == 6
                    # Error              1308 == 7
                    # Triple              766 == 8
                    
            #all_bip['play_result'] = np.where(all_bip['play_result'] == 'Out', 1, np.where(all_bip['play_result'] == 'Single', 2, np.where(all_bip['play_result'] == 'Double', 3, np.where(all_bip['play_result'] == 'HomeRun', 4, np.where(all_bip['play_result'] == 'FieldersChoice', 5, np.where(all_bip['play_result'] == 'Sacrifice', 6, np.where(all_bip['play_result'] == 'Error', 7, 8)))))))


####### POTENTIAL OUTLIERS #######

# Beacause I am planning on using a clustering algorithm, I want to ensure that the clustering is not skewed by outliers. This can be an issue, 
# espeically if using k-means or any clustering technique which uses mean and standard deviation. In fact, outliers generally have a large impact
# on all clustering data. So, I am going to use a combination of common sense (baseball common sense that is) and basic statistics to deal with outliers. 
# There are clustering techniques which are not as suscpetible to outliers, but I also believe that the goal of this contact grading algorithm 
# is not to grade extreme outliers, but to grade the vast majority of contact. 

# So, what are some potential outliers which could pop up? Well, since this dataset is produced by physical humans - there should not be any absurd 
# outliers. However, a technical error could be a cause. We are most likely to see outliers - or data points which don't make baseball sense - in 
# the quantitative variables. 

# Just examining some of the boxplots - looks like we do have some outliers here. 
sns.boxplot(x=all_bip['angle'])

sns.boxplot(x=all_bip['exit_speed'])

sns.boxplot(x=all_bip['distance'])

# I think the best method here will be to use z-score with +/- 3 standard deviations from the mean. This means we will retain most of our data for each
# variable, but get rid of outliers which can skew the k-means/cluster analysis.

# Before I execute the code, note that I start with 122,239 rows of data. 
all_bip = all_bip[(np.abs(stats.zscore(all_bip[['distance','angle','exit_speed','bearing','direction']])) < 3).all(axis=1)]
# Now, I have 119,736 rows of data. Obviously, we got rid of a signfiicant amount of outliers, and they are all from outliers in the quantitative variables. 

# Check the plots again - visually see where the ends are. 
sns.boxplot(x=all_bip['angle'])

sns.boxplot(x=all_bip['exit_speed'])

sns.boxplot(x=all_bip['distance'])

# Looks like we still have some outliers in the distance and exit velocity plots. I am going to specifically choose these columns and lower the z-score threshold. 
# I do not think this will signifciantly reduce the number of rows/observations - there is a lot of bunched data here so shouldn't do too much. 

# Distance Variable --- choose 2.7, sucessfully gets rid of the 538 foot home run. 
all_bip = all_bip[(np.abs(stats.zscore(all_bip['distance'])) < 2.6)]

# Exit Velocity Variable --- choose 2.8, gets rid of the absurd 130 mph value. This trims the lower bound a little, but we shouldn't be that 
# concerened here because any contact with that low a exit speed is most likely poor contact (or bunt). 
all_bip = all_bip[(np.abs(stats.zscore(all_bip['exit_speed'])) < 2.8)]

######## Cleaned Data Set has 118,855 observations, which is plenty. We have a cleaned dataset which can now be used for ML analysis. 


# =============================================================================
# ############################## MODEL BUILDING ###############################
# =============================================================================
# IMPORT STATEMENTS 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder

#### GOAL: Build a model which grades the quality of a hitters contact. 

# Why am I choosing clustering and what will it allow me to say about quality of contact? 
#   1. The objective of clustering is to interpret any structure in the data which the human eye can not see by grouping items with similar characteristics. 
#       In this case, there are multiple variables which can be of importance to contact quality. The human eye, nor a feasible forumla or function, could 
#       be written by hand by me to cluster these data points. 
#   2. I will be able to say how many clusters I want and I will be able to label each of these clusters -- this will allow me to use the variables 
#       to "grade" quality of contact. 

##### STEPS: 
#       1. K-Means Clustering 
#       2. Hierarchical Clustering/Agglomerative Clustering
#       3. Gaussian Mixture Mode

            
# NOTE:  I will be slowely removing variables from the dataset to see how the clustering responds. I am anticipating having to think about
#           if handedness of hitter is important or handedness of pitcher is important, etc. Might have to create multiple datasets which split by 
#           hitter handedness. I will be using SKLearn because I am most comofortable with the ML concepts, code execution, and website for troubleshooting. 

# Clean up the INDEX 
all_bip.reset_index(drop=True, inplace=True)

# As I explained before, I am simply going to use the numerical variables. The initial correlations we saw between some of the categorical 
# and numerical variables were not indicative of any relationship. And, because I care about building and evaluating a model for grading
# contact quality, variables such as strikes, balls, pitch type, handedness, play_result are not something which the hitter can control. 

##### This is the origional slicing of the dataset which was attempting to use all of the numericalv variables. 
# X_test = all_bip.iloc[:,10:14] 
# After running the clustering and analyzing the results, it was obvious that there were too many variables. Ie: there was too much noise. 
# So, I am revising to include only launch angle, exit speed, and distance. 
# The silhouette scores (how well the clustering grouped variables) is higher with just these three variables. 
###### Scores with 5 Variables ####
# 1. K-Means Score:  0.36989724778767213
# 2. Agglomerative Score: 0.33984108828733645
# 3. Gaussian Score: 0.3605514635452921

X_test = all_bip[['exit_speed','angle','distance']]


############ 1 - K-MEANS CLUSTERING  ###########

# For kmeans, I am going to use PCA (Principal Component Analysis) to reduce the dimensionality of the dataset. 
# This will increase the interpretability but at the same time minimizing information loss. 
preprocessing_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pre-processing', PCA(n_components=2))
                ])
                    
cluster_pipeline = Pipeline([
                ('kmeans', KMeans(
                                n_clusters = 5,
                                init="k-means++",
                                n_init=50,
                                max_iter=500)
                )])

pipe = Pipeline([
                ("preprocessor", preprocessing_pipeline),
                ("clusterer", cluster_pipeline)
                ])

pipe.fit(X_test)

preprocessed_data = pipe["preprocessor"].transform(X_test)
predicted_labels = pipe["clusterer"]["kmeans"].labels_

####### SCORING KMEANS ######
silhouette_score(preprocessed_data, predicted_labels)
#### - score with five clusters = 0.3980308666113088
# while not a bad score, this is to be expected. This means that the clusters are overlapping, but it is not less than 0 (which would mean wrong 
# clusters). 
# The value of the Silhouette score varies from -1 to 1. If the score is 1, the cluster is dense and well-separated than other clusters. 
# A value near 0 represents overlapping clusters with samples very close to the decision boundary of the neighboring clusters. 
# A negative score [-1, 0] indicates that the samples might have got assigned to the wrong clusters.
# Essentially, while we are not at 0, we are also not at 1. So our clusters do the job, but there is some overlapping on the boundaries. 

#### Saving Back Into all_bip Dataframe for Analysis #####
kmeans_cluster = pd.DataFrame(pipe["preprocessor"].transform(X_test),columns=["component_1", "component_2"])
kmeans_cluster["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
all_bip['k_means_predicted_cluster'] = kmeans_cluster['predicted_cluster']

##### PLOT TO INSPECT #####
plt.style.use("fivethirtyeight") 
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(x=kmeans_cluster["component_1"], y=kmeans_cluster["component_2"], s=50, data=kmeans_cluster, hue=kmeans_cluster["predicted_cluster"].tolist(), palette="Set2")
scat.set_title("5 KMeans Clusters - Contact Quality")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()
   
############ 2 - AGGLOMERATIVE CLUSTERING (HIERARCHICAL)  ###########
# I am going to use a hierarchical clustering method and see what the score/plot looks like compared to k-means. 
# It performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, 
# and clusters are successively merged together. Oftentimes works better on datasets depending on shape of variables. 

# Because Agglomerative Clustering is not suited to large datasets like kmeans or gaussian is, we need to dial it back on the number of samples. 
# Going with a sample size of 10,000

X_test_agglomerative = all_bip[['exit_speed','angle','distance']]
X_test_agglomerative = X_test_agglomerative.iloc[:10000,:]


preprocessing_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pre-processing', PCA(n_components=2))
                ])
                    
cluster_pipeline = Pipeline([
                ('Agglomerative', AgglomerativeClustering(
                                n_clusters = 5)
                )])

pipe = Pipeline([
                ("preprocessor", preprocessing_pipeline),
                ("clusterer", cluster_pipeline)
                ])

pipe.fit(X_test_agglomerative)

preprocessed_data = pipe["preprocessor"].transform(X_test_agglomerative)
predicted_labels = pipe["clusterer"]["Agglomerative"].labels_

####### SCORING AGGLOMERATIVE ########
silhouette_score(preprocessed_data, predicted_labels)
# score with 5 clusters ---> 0.372817061254939

# Saving back to all_bip Dataframe for analysis ####
agglomerative_cluster = pd.DataFrame(pipe["preprocessor"].transform(X_test_agglomerative),columns=["component_1", "component_2"])
agglomerative_cluster["predicted_cluster"] = pipe["clusterer"]["Agglomerative"].labels_
all_bip['agglomerative_predicted_cluster'] = agglomerative_cluster['predicted_cluster']

# because we were only able to get up to 10,000 rows, we have about 100,000 rows with nan values. So, going to fill those with 0's 

all_bip.fillna(value=0, inplace=True)


#### Plotting to Visually Inspect ####
plt.style.use("fivethirtyeight") 
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(x=agglomerative_cluster["component_1"], y=agglomerative_cluster["component_2"], s=50, data=agglomerative_cluster, hue=agglomerative_cluster["predicted_cluster"].tolist(), palette="Set2")
scat.set_title("5 Hierarchical Clusters (10,000 Observations) - Contact Quality")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()


######## GAUSSIAN MIXTURES ########

X_test_gaussian = all_bip[['exit_speed','angle','distance']]

preprocessing_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pre-processing', PCA(n_components=2))
                ])
                    
cluster_pipeline = Pipeline([
                ('Gaussian', GaussianMixture(
                                n_components = 5)
                )])

pipe = Pipeline([
                ("preprocessor", preprocessing_pipeline),
                ("clusterer", cluster_pipeline)
                ])

pipe.fit(X_test_gaussian)

preprocessed_data = pipe["preprocessor"].transform(X_test_gaussian)
predicted_labels = pipe.predict(X_test_gaussian)

##### SCORING GAUSSIAN #####
silhouette_score(preprocessed_data, predicted_labels)
# score with five cluters = 0.24448213315789538
# yikes, this is obviously not as good of a score as the other two clustering methods. Most likely, the previous clustering methods 
# do a better job of positioning the data. 


# Saving Back to DataFrame 
gaussian_cluster = pd.DataFrame(pipe["preprocessor"].transform(X_test_gaussian),columns=["component_1", "component_2"])
gaussian_cluster["predicted_cluster"] = predicted_labels
all_bip['gaussian_predicted_cluster'] = gaussian_cluster['predicted_cluster']

# Plotting Gaussian 
plt.style.use("fivethirtyeight") 
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(x=gaussian_cluster["component_1"], y=gaussian_cluster["component_2"], s=50, data=gaussian_cluster, hue=gaussian_cluster["predicted_cluster"].tolist(), palette="Set2")
scat.set_title("5 Gaussian Clusters - Contact Quality")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()

# =============================================================================
# ####################### Model Selection/Validation #########################
# =============================================================================

# The goal of this section is to select a specific method/algorithm/clustering technique. 

# K-Means Clustering Score = 0.4
# Agglomerative Clustering Score = 0.37
# Gaussian Clustering Score = 0.25

# Wow, they are all decently close. This means that each clustering method pretty effectively identified where cluster centroids should be, 
# found similar shape of those centroids, and then consequently scored similar - arguably, execpt for Gaussian. 
# Based off of these scores, I will choose the one most people are familiar with and also scored 
# slightly higher than the other two ----- K MEANS 

# But, how do we know how effectively they were clustered. In other words, now that I have clustered these and found their labels and plotted 
# the clusters, what does each cluster represent in terms of a "grade." ie: we must finish the GOAL - to evaluate/grade the quality of contact. 
# I am nearly there, but this last part will enable me to say what each cluster (1 through 5) means in terms of a contact quality "grade." 

###### EVALUATION TECHNIQUE ########

# For this evaluation technique, I will connect the K Means clusters (labeled 0 - 4 ) to a grade (A, B, C, D, F). Moreover, 
# this is not a random "grade." This grade of contact quality is deteremined by the unsupervised learning algorithm kmeans, and I am going to 
# use a few different tools to analyze it (visual inspection and statistical inspection)


################# VISUALIZATION #######################

# For each cluster, I will create a new dataframe (with all the other variables) so that the it becomes easier to work with
cluster_0 = all_bip[all_bip['k_means_predicted_cluster'] == 0]
cluster_1 = all_bip[all_bip['k_means_predicted_cluster'] == 1]
cluster_2 = all_bip[all_bip['k_means_predicted_cluster'] == 2]
cluster_3 = all_bip[all_bip['k_means_predicted_cluster'] == 3]
cluster_4 = all_bip[all_bip['k_means_predicted_cluster'] == 4]

# At first glance, we can even see how similar the clustering is. Each of the clustering types generally have the same cluster number. 

# What I am doing in creating this dataframe is to tabluate the mean results of each of the 5 numerical variables. 
# I also want to look at value counts/frequency of categorical variables like like play_result and hit_type for each cluster.
# this is reult oriented, but I think it can help us understand the clusters as well.

clustering_numerical_results = pd.DataFrame({
'cluster': ['0','1','2','3','4'],
'distance': [cluster_0['distance'].mean(), cluster_1['distance'].mean(), cluster_2['distance'].mean(), cluster_3['distance'].mean(), cluster_4['distance'].mean()],
'exit_speed': [cluster_0['exit_speed'].mean(), cluster_1['exit_speed'].mean(), cluster_2['exit_speed'].mean(), cluster_3['exit_speed'].mean(), cluster_4['exit_speed'].mean()],
'angle': [cluster_0['angle'].mean(), cluster_1['angle'].mean(), cluster_2['angle'].mean(), cluster_3['angle'].mean(), cluster_4['angle'].mean()],
'direction': [cluster_0['direction'].mean(), cluster_1['direction'].mean(), cluster_2['direction'].mean(), cluster_3['direction'].mean(), cluster_4['direction'].mean()],
'bearing': [cluster_0['bearing'].mean(), cluster_1['bearing'].mean(), cluster_2['bearing'].mean(), cluster_3['bearing'].mean(), cluster_4['bearing'].mean()],
})

clustering_numerical_results = clustering_numerical_results.round()


# Turning clustering_numerical_results into Table #

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

render_mpl_table(clustering_numerical_results, header_columns=0, col_width=2.0)

######## Clustering Results Statistical Median ##########

clustering_numerical_results_median = pd.DataFrame({
'cluster': ['0','1','2','3','4'],
'distance': [cluster_0['distance'].median(), cluster_1['distance'].median(), cluster_2['distance'].median(), cluster_3['distance'].median(), cluster_4['distance'].median()],
'exit_speed': [cluster_0['exit_speed'].median(), cluster_1['exit_speed'].median(), cluster_2['exit_speed'].median(), cluster_3['exit_speed'].median(), cluster_4['exit_speed'].median()],
'angle': [cluster_0['angle'].median(), cluster_1['angle'].median(), cluster_2['angle'].median(), cluster_3['angle'].median(), cluster_4['angle'].median()],
'direction': [cluster_0['direction'].median(), cluster_1['direction'].median(), cluster_2['direction'].median(), cluster_3['direction'].median(), cluster_4['direction'].median()],
'bearing': [cluster_0['bearing'].median(), cluster_1['bearing'].median(), cluster_2['bearing'].median(), cluster_3['bearing'].median(), cluster_4['bearing'].median()],
})

clustering_numerical_results_median = clustering_numerical_results_median.round()


# Turning clustering_numerical_results_median into Table #

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

render_mpl_table(clustering_numerical_results_median, header_columns=0, col_width=2.0)

# Understanding Categorical Using Value Counts and Manually Input #
# Starting with Hit_Type ###
cluster_0['hit_type'].value_counts()
cluster_1['hit_type'].value_counts()
cluster_2['hit_type'].value_counts()
cluster_3['hit_type'].value_counts()
cluster_4['hit_type'].value_counts()

clustering_hit_type = pd.DataFrame({
'cluster': ['0','1','2','3', '4'],
'Bunt': ['0','0','16','0','0'],
'GroundBall': ['25541','886','24480','10','9'],
'PopUp':['0','7070','1','23','1179'],
'LineDrive': ['4644','3864','191','13395','8547'],
'FlyBall': ['2','3817','0','12191','12989']})


# Turn cluster_hit_type DataFrame into Table # 
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

render_mpl_table(clustering_hit_type, header_columns=0, col_width=2.0)

# Plot each cluster hit type as Bar Chart # 



# Next is play_result #
cluster_0['play_result'].value_counts()
cluster_1['play_result'].value_counts()
cluster_2['play_result'].value_counts()
cluster_3['play_result'].value_counts()
cluster_4['play_result'].value_counts()

clustering_play_result = pd.DataFrame({
'cluster': ['0','1','2','3','4'],
'Out': ['160140','10947','19013','9337','16952'],
'Single': ['10595','3997','2631','3884','3568'],
'Double': ['1385','456','233','4966','1225'],
'Triple': ['81','24','11','515','135'],
'HomeRun': ['0','1','0','6422','158'],
'Error': ['507','72','584','49','59'],
'Sacrifice': ['0','52','2','441','618'],
'FieldersChoice': ['1605','88','2214','5','9']})



# Clustering Play Results to Table ####
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

render_mpl_table(clustering_play_result, header_columns=1, col_width=2.0)


# =============================================================================
# ######################## LABELLING CLUSTERS AS GRADE ########################
# =============================================================================


#### AFTER a lot of manual visualization and statistical measures, I am very confident in labeling each cluster for contact quality. 
# Essentially, the statistical measures which I used were able to clearly illuminate which cluster corresponds to which grade (A. B. C. D. F)
# As part of the dataset, there were bunts (and while I could have filtered them out) I decided to keep them as I felt that it would instruct better
# clusters. Moreover, if there was more of a separation of data points (not all grouped), the clustering algorithum might to a better job of separating. 
# In addition, I used some data visualization tools - that I include in the summary of results - to explore the clusters before finally labelling them 

####### SO, CONTACT QUALITY GRADE IS AS FOLLOWS: 
        # cluster 0 = C
        # cluster 1 = B
        # cluster 2 = F
        # cluster 3 = D
        # cluster 4 = A

# A grade cluster is the 4 cluster - simple statistical analysis and visualization shows this - longest average distance, highest average exit speed and best launch angle. 
# B grade cluster is the 1 cluster - second longest average distance, pretty solid exit velocity, but the launch angle is more ideal here. 
# C grade cluster is the 0 cluster - Now, this is where baseball experience and knowledge comes in. If we are grading contact quality, we shold be 
    # be looking beyond distance, as sometimes a player can just miss a ball (thus their launch angle is not optimal), but they still hit it very hard. 
    # For this reason, I have graded the 0 cluster as a C contact quality because the exit velocity is really high, but the launch angle is not optimal.
    # Thus, the might have just missed the ball, BUT it was either a linedrive or an extremely hard groundball, both of which put extreme pressure on defenses. 
# D grade cluster is the 3 cluster - although the distance is higher here than cluster C, the exit velocity and launch angle are all less optimal 
    # than the C cluster - for this reason, I have graded it as the D cluster. This is not great contact, most likely popouts and shallow flyouts. 
# F grade cluster is the 2 cluster - this is probably where the bunts and little dribblers/popups are located. 

# I will create a new column and then place the Grade in #####

all_bip['Contact_Quality_Grade'] = np.where(all_bip['k_means_predicted_cluster'] == 0, 'C', np.where(all_bip['k_means_predicted_cluster'] == 1, 'B', np.where(all_bip['k_means_predicted_cluster'] == 2, 'F', np.where(all_bip['k_means_predicted_cluster'] == 3, 'D', 'A'))))



##### LAST THING, DESCRIBE EACH OF THE IMPORTANT VARIABLES with 25 and 75 percentiles to get an understanding of how 
##### these can be used for future prediction ##### 
##### manually make table #####

### CLUSTER 0 ###
cluster_0['angle'].describe()
# 25% = -9.71
# 75% = 4.94

cluster_0['distance'].describe()
# 25% = 13.73
# 75% = 104.34

cluster_0['exit_speed'].describe()
# 25% = 92.98
# 75% = 102.37

### CLUSTER 1 ###
cluster_1['angle'].describe()
# 25% = 21.99
# 75% = 44.06

cluster_1['distance'].describe()
# 25% = 248.99
# 75% = 315.15

cluster_1['exit_speed'].describe()
# 25% = 84.54
# 75% = 91.80

### CLUSTER 2 ###
cluster_2['angle'].describe()
# 25% = -25.74
# 75% = -6.05

cluster_2['distance'].describe()
# 25% = 5.76
# 75% = 26.11

cluster_2['exit_speed'].describe()
# 25% = 70.57
# 75% = 82.77

### CLUSTER 3 ###
cluster_3['angle'].describe()
# 25% = 24.12
# 75% = 60.41

cluster_3['distance'].describe()
# 25% = 138.76
# 75% = 222.01

cluster_3['exit_speed'].describe()
# 25% = 69.58
# 75% = 76.45

### CLUSTER 4 ###
cluster_4['angle'].describe()
# 25% = 16.31
# 75% = 30.26

cluster_4['distance'].describe()
# 25% = 303.75
# 75% = 385.01

cluster_4['exit_speed'].describe()
# 25% = 97.88
# 75% = 104.65

### Saving to Folder for Data Visualization ###
#all_bip.to_csv('/Users/kieranshaw/Desktop/Giants Analyst/all_bip.csv')





