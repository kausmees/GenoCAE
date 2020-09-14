import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numba import jit


def flip_genos(genodata):
	'''
	Flip genotype labels between representation of REF allele count and ALT allele count.


	:param genodata: genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata: array, shape (n_markers x samples)
	'''
	print("Flipping genotype labels.")

	genodata[genodata == 0.0] = 3.0
	genodata[genodata == 2.0] = 0.0
	genodata[genodata == 3.0] = 2.0

	return genodata

@jit(nopython=True)
def get_mean_and_std_smartPCAstyle(genodata):
	'''
	Get the means and std-estimations to normalize smartPCAstyle


	:param genodata: data to define normalization based on. genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata: array, shape (n_markers x samples)
	:return:
	'''
	means = []
	stds = []
	for r in range(genodata.shape[0]):
		# one row = one SNPs genotypes
		genotypes = genodata[r, :]

		rowvalid = 0.0
		rowsum = 0.0

		for n in range(len(genotypes)):
			if genotypes[n] < 9.0:
				rowvalid  += 1.0
				rowsum += genotypes[n]

		rowmean = rowsum / rowvalid

		p = (rowsum + 1.0) / ((2.0 * rowvalid) + 2.0)
		y  = p * (1.0-p)

		if (y > 0.0):
			ynew = np.sqrt(y)
		else:
			ynew = 1.0

		means.append(rowmean)
		stds.append(ynew)

	return means, stds


def normalize_genos_smartPCAstyle(genodata_train, genodata_test, flip=False, missing_val = 0.0, get_scaler = False):
	'''
	Normalize genotypes same way as smartpca code
	(https://github.com/chrchang/eigensoft/blob/e7a66ede12a3e6567e491f14f2980119d84d6162/src/eigensrc/smartpca.c)

	Normalization over rows (over SNPs).
	Based on train data, applied to train data and test data.

	Missing values ignored in normalization, replaced by missing_val in output.


	:param genodata_train: data to define normalization based on. genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata_train: array, shape (n_markers x samples)
	:param genodata_test: data to normalize based on the fitted normalization. genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata_test: array, shape (n_markers x n_samples)
	:param flip: Flip genotype labels, so they represent REF allele count instead of ALT allele count.
	:return: Normalized genodata_train, transposed so its (n_samples x n_markers).
  			 Normalized genodata_test, transposed so its (n_samples x n_markers).
	'''


	if flip:
		genodata_train = flip_genos(genodata_train)
		genodata_test = flip_genos(genodata_test)

	means, stds = get_mean_and_std_smartPCAstyle(genodata_train)

	scaler = StandardScaler()
	# scaler.fit(genodata_train[0:2].T)
	scaler.fit(genodata_train.T)
	scaler.mean_ = means
	scaler.scale_ = stds

	# have to replace with NaN so standardscaler ignores
	genodata_train[genodata_train == 9.0] = np.nan
	genodata_test[genodata_test == 9.0] = np.nan


	genodata_train = scaler.transform(genodata_train.T)

	if len(genodata_test) > 0:
		genodata_test = scaler.transform(genodata_test.T)

	genodata_train[np.isnan(genodata_train)] = missing_val
	genodata_test[np.isnan(genodata_test)] = missing_val

	if get_scaler:
		return genodata_train, genodata_test, (means, stds)
	else:
		return genodata_train, genodata_test



def normalize_genos_genotypewise01(genodata_train, genodata_test, flip=False, missing_val=9, get_scaler = False):
	'''
	Normalize genotypes into interval [0,1] by translating 0,1,2 -> 0.0, 0.5, 1.0

	OBS this is done independent of the observed genotypes. A marker with only genotypes 0,1 observed will this only
	have 0 and 0.5 in the normalized data, meaning the different variables don't neccessatily have values in the same
	intervals.


	Missing values will have value missing_val in output.

	:param genodata_train: genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata_train: array, shape (n_markers x samples)
	:param genodata_test: genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata_test: array, shape (n_markers x n_samples)
	:param flip: Flip genotype labels, so they represent REF allele count instead of ALT allele count.
	:return: Normalized genodata_train, transposed so its (n_samples x n_markers).
  			 Normalized genodata_test, transposed so its (n_samples x n_markers).
	'''
	if flip:
		print("Flipping genotype labels.")

		genodata_train[genodata_train == 1.0] = 0.5
		genodata_train[genodata_train == 0.0] = 1.0
		genodata_train[genodata_train == 2.0] = 0.0
		genodata_train[genodata_train == 9.0] = missing_val

	else:
		genodata_train[genodata_train == 1.0] = 0.5
		genodata_train[genodata_train == 2.0] = 1.0
		genodata_train[genodata_train == 9.0] = missing_val


	genotypes_normed_train = genodata_train.T


	if flip:
		genodata_test[genodata_test == 1.0] = 0.5
		genodata_test[genodata_test == 0.0] = 1.0
		genodata_test[genodata_test == 2.0] = 0.0
		genodata_test[genodata_test == 9.0] = missing_val

	else:
		genodata_test[genodata_test == 1.0] = 0.5
		genodata_test[genodata_test == 2.0] = 1.0
		genodata_test[genodata_test == 9.0] = missing_val

	genotypes_normed_test = genodata_test.T

	if get_scaler:
		return genotypes_normed_train, genotypes_normed_test, None
	else:
		return genotypes_normed_train, genotypes_normed_test


def normalize_genos_standard(genodata_train, genodata_test, flip=False, missing_val=None, get_scaler = False):
	'''
	Normalize genotypes by subtracting mean and dividing with standard deviation.

	Missing values ignored in normalization, replaced by missing_val in output.


	:param genodata_train: data to fir the scaler to. genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata_train: array, shape (n_markers x samples)
	:param genodata_test: genotypes represented as 0,1,2, missing values encoded as 9
	:type genodata_test: array, shape (n_markers x n_samples)
	:param flip: Flip genotype labels, so they represent REF allele count instead of ALT allele count.
	:param missing_val: not used.
	:type missing_val: not used.
	:return: Normalized genodata_train, transposed so its (n_samples x n_markers).
  			 Normalized genodata_test, transposed so its (n_samples x n_markers).
	'''

	if flip:
		genodata_train = flip_genos(genodata_train)
		genodata_test = flip_genos(genodata_test)


	# have to replace with NaN so standardscaler ignores
	genodata_train[genodata_train == 9.0] = np.nan
	genodata_test[genodata_test == 9.0] = np.nan


	scaler = StandardScaler()
	scaler.fit(genodata_train.T)


	genodata_train = scaler.transform(genodata_train.T)

	if len(genodata_test) > 0:
		genodata_test = scaler.transform(genodata_test.T)

	genodata_train[np.isnan(genodata_train)] = missing_val
	genodata_test[np.isnan(genodata_test)] = missing_val

	if get_scaler:
		return genodata_train, genodata_test, (scaler.mean_, scaler.scale_)
	else:
		return genodata_train, genodata_test
