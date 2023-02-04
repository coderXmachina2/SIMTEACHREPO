import pandas as pd
import numpy as np
import random
from scipy.stats import norm, kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.stats import normaltest, anderson, kstest, norm, shapiro
import pylab
import statsmodels.api as sm
from scipy.stats import norm

def stat_test_normal(data_in):
    stat, p = shapiro(np.array(data_in[:4500]))
    print("Shapiro Stat:", stat, "p-value", p)

    stat, p = normaltest(np.array(data_in[:]))
    print("Nortmaltest Stat:", stat, "p-value", p)

    ks_statistic, p_value = kstest(np.array(data_in[:7000]), 'norm')
    print("KS test:", ks_statistic, p_value)
    
    result = anderson(data_in[:])

    print('AD test stat=%.3f' % (result.statistic))
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print('Probably Gaussian at the %.1f%% level' % (sl))
        else:
            print('Probably not Gaussian at the %.1f%% level' % (sl))

#
def plot_pdf(data_in, dtitle, n_bins =50):
    # Generate some data for this demonstration.
    data = data_in
    mu, std = norm.fit(data)

    # Plot the histogram.
    fig, ax1 = plt.subplots(figsize=(12, 12))
    
    plt.hist(data, 
             bins=n_bins, 
             density=True, 
             alpha=0.6)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results for %s : mu = %.2f,  std = %.2f" % (dtitle,mu, std)
    plt.grid()
    plt.title(title)

    plt.show()
    
def qq_plot(data_in):
    sm.qqplot(data_in, line='45')
    pylab.grid()
    pylab.title('QQ plot')
    pylab.show()
    
def clean_movie( df1 , verbose_shape=False):
    approved=['PG-13', 'R', 'PG', 'TV-14','TV-14','18TC','TV-MA', '18+', 'G', '12+', 'TV-PG', '16+', '13+',
           'MA-17', 'TV-Y7', 'TV-G', 'TV-Y', '13 +']

    corrT=['https://www.imdb.com/name/nm7517010/?ref_=adv_li_st_1',
          '4,766.', 'https://www.imdb.com/name/nm11581010/?ref_=adv_li_st_1', '409 min', '7,055.',
          'https://www.imdb.com/name/nm14231318/?ref_=adv_li_st_1', 
          'https://www.imdb.com/name/nm9643298/?ref_=adv_li_st_1']

    df1['certificate'].replace(df1['certificate'].unique()[25:],
            [random.choice(approved) for x in approved][:10], inplace=True)
    df1['certificate'].fillna('Not Rated', inplace=True)
    df1['certificate'].replace('Approved',
            random.choice(approved), inplace=True)
    
    #df1['Dependents'].replace(df1['certificate'].unique(),
    #                        np.linspace(0,22,23),
    #                        inplace=True)
    
    #############################################################################################################
    
    timekill = ['https://www.imdb.com/name/nm7517010/?ref_=adv_li_st_1', '5,538 min',
                'https://www.imdb.com/name/nm11581010/?ref_=adv_li_st_1',
                'https://www.imdb.com/name/nm14231318/?ref_=adv_li_st_1',
                'https://www.imdb.com/name/nm9643298/?ref_=adv_li_st_1',
                '7,793.' 'https://www.imdb.com/name/nm9643298/?ref_=adv_li_st_1',
                '3 min', '4,766.', '6,297.', '7,055.', '7,793.'] 
    df1['Time'].fillna('60 min', inplace=True)
    df1['Time'].replace( timekill ,
            '60 min', inplace=True)
    
    df1['inlineblock'].fillna('6.2', inplace=True)
    df1['inlineblock'].replace(df1['inlineblock'].unique()[90:],
            '6.2', inplace=True)
    df1['inlineblock'].replace('https://www.imdb.com/name/nm10791067/?ref_=adv_li_st_3',
            '6.5', inplace=True)
    df1['inlineblock'].replace('Ashley Thornhill',
            '6.5', inplace=True)

    df1['Score'].fillna(random.choice(df1['Score'].unique()[:50]), inplace=True)
    df1['Score'].replace(df1['Score'].unique()[69:],
            [random.choice(df1['Score'].unique()[:50]), random.choice(df1['Score'].unique()[:50]), 
             random.choice(df1['Score'].unique()[:50]), random.choice(df1['Score'].unique()[:50]),
             random.choice(df1['Score'].unique()[:50])], inplace=True)
    
    df1['reviews'].fillna('2,500', inplace=True  )
    df1['reviews'].replace('$0.00M', '880', inplace=True  )
    
    return(df1)

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