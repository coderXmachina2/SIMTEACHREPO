import pandas as pd
import numpy as np

def clean_df( data_F , verbose_shape=False):
	#replace nans
	data_F['Gender'].fillna('Male', inplace=True  )
	data_F['Married'].fillna('Yes', inplace=True  )
	data_F['Education'].fillna('Graduate', inplace=True  )
	data_F['LoanAmount'].fillna(np.mean(data_F['LoanAmount']), inplace=True  )
	data_F['Loan_Amount_Term'].fillna(360, inplace=True  )
	data_F['Credit_History'].fillna(1, inplace=True  ) #0, 1 #
	data_F['Dependents'].fillna(0, inplace=True)
	data_F['Self_Employed'].fillna(0, inplace=True)
	data_F['Property_Area'].fillna('Urban', inplace=True)

	#Binarize/ Replace Categorical with Numerical
	data_F['Married'].replace(['Yes', 'No'],
	                        [0, 1], inplace=True)
	data_F['Gender'].replace(['Male', 'Female'],
        	                [0, 1], inplace=True)
	data_F['Education'].replace(['Graduate', 'Not Graduate'],
        	                [0, 1], inplace=True)
	data_F['Self_Employed'].replace(['Yes', 'No'],
        	                [1, 0], inplace=True)
	#Integerise
	data_F['Dependents'].replace(['0', '1', '2', '3+'],
        	                 np.linspace(0,3,4),
                	         inplace=True)
	data_F['Loan_Amount_Term'].replace(np.sort(data_F['Loan_Amount_Term'].unique()),
        	                 np.linspace(0,9,10), inplace=True)
	data_F['Property_Area'].replace(['Rural', 'Semiurban', 'Urban'],
        	                 [0,1,2], inplace=True)

	data_F['HouseIncome'] = data_F['ApplicantIncome']+ data_F['CoapplicantIncome']

	if(verbose_shape):
		print(data_F.shape[0], "Rows x ", data_F.shape[1], "Columns"), data_F.describe()
        
    #return( data_F )