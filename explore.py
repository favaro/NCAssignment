
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import feature_selection
import scipy.stats as ss
import seaborn as sea
import statsmodels.formula.api as sm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn import tree
from sklearn.cross_validation import train_test_split

from config import categorical_features, numerical_features 
from config import FPlength_dict, Pill_dict, CycleVar_dict, NCbefore_dict, ExitStatus_dict
from config import load_data, encode_all, get_algo_performance

describe_and_plot = 0 



#****************************************************************

# Read in and pre-process the input data
data = load_data()

# ***************************************************************

# Explore and visualize the data 

if describe_and_plot:

	data.describe()

	# Print out the statistics for categorical features
	for c in categorical_features:
		data[c].value_counts()
		fig = plt.figure()
		data[c].value_counts().plot(kind='bar')
		plt.savefig('visualization/general/' + c + '.pdf')

	for c in numerical_features:
		fig = plt.figure()
		plt.hist(data[c])
		plt.savefig('visualization/general/' + c + '.pdf')

# ***************************************************************

# ANALYSIS FOR Time it takes to get pregnant 

print '*********** Performing analysis for TIME TO PREGNANCY ***********'

# Select users who got pregnant while actively using the app and having intercourse
data_preg = data[ data['ExitStatus'] == 'Pregnant']
data_preg = data_preg[ data_preg['SexLogFreq'] > 0 ]

target = 'DaysTrying' 

# Convert FPlength to numerical using ordering assumption
for name,value in FPlength_dict.iteritems():
	data_preg['FPlength'][ data_preg['FPlength'] == name ] = value

categorical_features_tpreg = categorical_features[:]
categorical_features_tpreg.remove('FPlength')
numerical_features_tpreg = numerical_features[:]
numerical_features_tpreg.append('FPlength')

# Correlate categorical feature to numerical target
for f in categorical_features_tpreg:

	if describe_and_plot:
		data_preg.boxplot(target, by=f, notch=True)
		plt.savefig('visualization/toTimeToPregnancy/boxplot_' + f + '.pdf')

	means   = data_preg.groupby(f).mean()[target]
	medians = data_preg.groupby(f).median()[target]
	sems    = data_preg.groupby(f).agg(ss.sem)[target]
	if describe_and_plot:
		print ' summary for feature ', f, ' mean, median, sem = ', means, medians, sems

	vvalues = list(data_preg[f].unique())

	data_0 = data_preg[ data_preg[f] == vvalues[0] ][target]
	data_1 = data_preg[ data_preg[f] == vvalues[1] ][target]
	if describe_and_plot:
		if f == 'Pill':
			data_2 = data_preg[ data_preg[f] == vvalues[2] ][target]
			#print 'ChiSquare = ', ss.friedmanchisquare( data_0, data_1, data_2)
			print f, ' ANOVA = ', ss.f_oneway( data_0, data_1, data_2)
			print f, ' Kruskal test  = ', ss.kruskal( data_0, data_1, data_2)
		else:
			print f, ' ANOVA = ', ss.f_oneway( data_0, data_1)
			print f, ' Kruskal test  = ', ss.kruskal( data_0, data_1)

# Correlate numerical features: Pearson/Spearman correlation coefficient
for f in numerical_features_tpreg:

	pearson  = ss.pearsonr( data_preg[f], data_preg[target] )
	spearman = ss.spearmanr( data_preg[f], data_preg[target] )
	if describe_and_plot:
		print f
		print '  Pearson coeff = ', pearson
		print '  Spearman coeff = ', spearman

		fig = plt.figure()
		plt.scatter( data_preg[f], data_preg[target], marker='.' )
		plt.savefig('visualization/toTimeToPregnancy/corr_' + f + '.pdf')

		fig = plt.figure()
		seaplot = sea.regplot(x=data_preg[f], y=data_preg[target], x_bins=15, fit_reg=None).get_figure()
		plt.ylim(0,250)
		seaplot.savefig('visualization/toTimeToPregnancy/profile_' + f + '.pdf')


# ***************************************************************

# ANALYSIS FOR Dropout Rate

print '********* Performing analysis for DROPOUT RATE ***********'

data_out = data[ data['ExitStatus'] != 'Pregnant']
target_out = 'ExitStatus'

numerical_features_out = numerical_features[:]

# Visualize dependencies with box plots (numerical features only)
for f in numerical_features_out:
	plt.figure()
	data_out.boxplot(f, by=target_out, notch=True)
	plt.savefig('visualization/toDropoutRate/boxplot_' + f + '.pdf')

data_out.loc[:,'ExitStatus'] = data_out.loc[:,'ExitStatus'].replace({'Dropout':1, 'Right':0})

# Visualize dependencies with scatter plots (numerical features only)
for f in numerical_features_out:
	plt.figure()
	plt.scatter( data_out[f], data_out[target_out], marker='.' )
	plt.savefig('visualization/toDropoutRate/corr_' + f + '.pdf')


# Convert FPlength to numerical using ordering assumption
for name,value in FPlength_dict.iteritems():
	data_out['FPlength'][ data_out['FPlength'] == name ] = value

# Convert other categorical features to 0-1 binary
to_encode_out = ['Pill', 'NCbefore', 'CycleVar', 'Country'] 
data_out = encode_all(to_encode_out, data_out)

data_out = data_out.drop(['SexLogFreq'],axis=1)

# Setup features and labels
yfit = data_out[target_out]
xfit = data_out.drop([target_out], axis=1)

# Setup training & testing sets (should be randomized)
ntrain = 8000
ytrain = yfit[:ntrain]
xtrain = xfit.iloc[:ntrain,:]
ytest = yfit[ntrain:]
xtest = xfit.iloc[ntrain:,:]

# Construct an ensemble classifier and fit it to the data
forest = RandomForestClassifier(max_features='auto', max_depth=13, n_estimators=1000)
forest.fit(xtrain.values, ytrain.values)
# Predict
ytrain_predicted = forest.predict(xtrain)
ytest_predicted = forest.predict(xtest)


# Estimate algorithm performance to make sure the feature relevance makes sense
perftrain_out = get_algo_performance( forest, ytrain_predicted, ytrain )
perftest_out  = get_algo_performance( forest, ytest_predicted, ytest )

frel_forest = forest.feature_importances_
frel_forest_df = pd.DataFrame(data=frel_forest, columns = ['FeatureImportance'], index=xtrain.columns)

print '*** Results of Dropout Rate analysis ***'
print 'Random Forest Classifier '
print 'training accuracy ', forest.score(xtrain, ytrain)
print 'testing accuracy  ', forest.score(xtest, ytest)
print 'training purity, efficiency (dropout class) ', perftrain_out
print 'testing purity, efficiency (dropout class)  ', perftest_out
print 'feature relevance '
print frel_forest_df.sort_values(by='FeatureImportance',ascending=False)[0:10]


# *************************************************************************************************************

# ANALYSIS FOR Fertility

# I define fertility as the potential to conceive, and not to carry a pregnancy to full term
# Fertility in this analysis is the capability of the couple to conceive
#
# fertile: 
#	ExitStatus = Pregnant
# 	any Days/CyclesTrying
# 	any SexLogFreq
#
# non-fertile:
#	ExitStatus = Right (still trying)
#	CyclesTrying > 6
# 	SexLogFreq 

# * * * *

print '*********** Performing analysis for FERTILITY  ***********'

data_f = data.copy()
# remove NCbefore which I consider irrelevant
data_f = data_f.drop(['NCbefore'], axis=1)

target_f = 'Fertile'
features_to_fit_f = ['Age', 'NumBMI', 'FPlength', 'Weight', 'AnovCycles', 'Pill', 'CycleVar', 'Country','Fertile']

numerical_features_f   = ['Age','NumBMI','Weight','AnovCycles']
categorical_features_f = categorical_features[:]
categorical_features_f.remove('FPlength')

# Build dataset of potentially infertile users
data_fneg = data_f[ data_f['SexLogFreq'] > 0 ] 
data_fneg = data_fneg[ data_fneg['invSexLogFreq'] < 29 ] 
data_fneg = data_fneg[ data_fneg['CyclesTrying'] > 6 ]
data_fneg = data_fneg[ data_fneg['ExitStatus'] == 'Right']
data_fneg.loc[:,'Fertile'] = 0
# Build dataset of fertile users
data_fpos = data_f[ data_f['ExitStatus'] == 'Pregnant' ]
data_fpos.loc[:,target_f] = 1
# Merge the two datasets
data_fall = pd.concat([data_fneg, data_fpos])
data_fall = data_fall.loc[:,features_to_fit_f]

# Plot the effect of numerical features to the target
for f in numerical_features_f:
	plt.figure()
	data_fall.boxplot(f, by=target_f, notch=True)
	plt.savefig('visualization/toFertility/boxplot_' + f + '.pdf')

for name,value in FPlength_dict.iteritems():
	data_fall['FPlength'][ data_fall['FPlength'] == name ] = value

to_encode_f = ['Pill', 'CycleVar', 'Country'] 
data_fall   = encode_all(to_encode_f, data_fall)

xfit_f = data_fall.drop([target_f], axis=1)
yfit_f = data_fall.loc[:,target_f]

train, test = train_test_split( data_fall, test_size = 0.2)
xtrain_f = train.drop([target_f], axis=1)
ytrain_f = train.loc[:,target_f]
xtest_f  = test.drop([target_f], axis=1)
ytest_f  = test.loc[:,target_f]

forest_f = RandomForestClassifier( max_features='auto', max_depth=12, n_estimators=1000, class_weight={0:0.95,1:0.05})
forest_f.fit(xtrain_f, ytrain_f)

ytrain_predicted_f = forest_f.predict(xtrain_f)
ytest_predicted_f  = forest_f.predict(xtest_f)

# Calculate algorithm performance
perftrain_f = get_algo_performance( forest_f, ytrain_predicted_f, ytrain_f )
perftest_f  = get_algo_performance( forest_f, ytest_predicted_f, ytest_f )

frel_forest_f = forest_f.feature_importances_
frel_forest_f_df = pd.DataFrame(data=frel_forest_f, columns = ['FeatureImportance'], index=xtrain_f.columns)

print '*** Results of Fertility analysis ***'
print 'Random Forest Classifier '
print 'training accuracy ', forest_f.score(xtrain_f, ytrain_f)
print 'testing accuracy  ', forest_f.score(xtest_f, ytest_f)
print 'training purity, efficiency (dropout class) ', perftrain_f
print 'testing purity, efficiency (dropout class)  ', perftest_f
print 'feature relevance '
print frel_forest_f_df.sort_values(by='FeatureImportance',ascending=False)[0:10]

