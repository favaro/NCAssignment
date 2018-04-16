
import matplotlib.pyplot as plt
import pandas as pd


categorical_features = [#'Country',
						'Pill',
						'NCbefore',
						'FPlength',
						'CycleVar']

numerical_features = ['Age',
					  'NumBMI',
					  'Weight',
					  'TempLogFreq',
					  'SexLogFreq',
					  'AnovCycles',
					  'DaysTrying',
					  'CyclesTrying']
					


# To convert FPlenght to numerical
FPlength_dict = {'long'  : 3,
				 'normal': 2,
				 'short' : 1}

# To convert Pill to numerical
Pill_dict = {'TRUE' : 3,
			 'NULL' : 2,
			 'FALSE': 1}

# To convert CycleVar to numerical
CycleVar_dict = {'regular'   : 1,
			 	 'irregular' : 0}

# To convert NCbefore to numerical
NCbefore_dict = {'TRUE'   : 1,
			 	 'FALSE'  : 0}

# To convert ExitStatus to numerical (for dropout rate analysis only)
ExitStatus_dict = {'Dropout': 1,
					'Right': 0}



#***************************************************************************************************

# Function to load and treat the input data
def load_data( ):

	data = pd.read_csv('anafile_challenge_170522.csv')
	data.columns = data.columns.map(lambda x: x.replace(' ',''))

	# Preprocess
	data['NCbefore']   = data['NCbefore'].str.strip()
	data['Pill']       = data['Pill'].str.strip()
	data['FPlength']   = data['FPlength'].str.strip()
	data['CycleVar']   = data['CycleVar'].str.strip()
	data['ExitStatus'] = data['ExitStatus'].str.strip()

	data['invSexLogFreq'] = data['SexLogFreq'].map(lambda x: 1/x if x!=0 else 0)
	numerical_features.append('invSexLogFreq')

	return data


# Funtion to apply encoding to turn categorical data columns into numerical (one column)
def encode(feature_name, data_in):

	data_out = data_in.copy()
	binaryfeatures = pd.get_dummies(data_out.loc[:,feature_name])
	binaryfeatures.columns = binaryfeatures.columns.map(lambda fname: feature_name + fname.replace(" ", "_"))   
	data_out = data_out.drop(feature_name, axis=1).join(binaryfeatures)
	
	return data


# Funtion to apply encoding to turn categorical data columns into numerical (all columns)
def encode_all(feature_names, data):

	for name in feature_names:
		binaryfeatures = pd.get_dummies(data.loc[:,name])
		binaryfeatures.columns = binaryfeatures.columns.map(lambda fname: name + fname.replace(" ", "_"))
		data = data.drop(name, axis=1).join(binaryfeatures)
	
	return data


# Function to normalize data for regression algos
def normalize(feature_names, data):

	for name in feature_names:
		col = data[name].values.astype(float) 
		x_scaled = preprocessing.MinMaxScaler().fit_transform(col.reshape(-1,1))
		data = data.drop(name, axis=1)
		data.name = pd.DataFrame(x_scaled)

	return data


#***************************************************************************************************

def get_algo_performance( model, predicted, true ):

	truepos    = ( (predicted==1) & (true==1) )
	purity     = float(truepos.sum())/float(predicted.sum())
	efficiency = float(truepos.sum())/float(true.sum())

	return purity, efficiency
