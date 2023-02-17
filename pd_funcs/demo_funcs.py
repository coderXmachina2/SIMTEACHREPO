import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats
import math
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy as cp

def data_loader(filename):  
	with open('./data/'+filename, 'rb') as fid:
		a = pickle.load(fid)

	return(a)

def demo_plot(input_data, 
		title,
		verbose_data = False,
		verbose_plot_mag_line=False,
		mag_line=False):

	data_names = []

	xs = []
	x_errs = []
	ys = []
	y_errs = []

	x_bar = []
	xerr_bar = []
	y_bar = []
	yerr_bar = []

	#Intermediary processing operation. What has been packaged needs to be unpackaged
	for dats in input_data:		
		x_bar.append( np.mean(np.array(dats[list(dats.keys())[0]][0]))   )
		xerr_bar.append( np.mean(np.array(dats[list(dats.keys())[0]][3])  )      )
		y_bar.append( np.mean(np.array(dats[list(dats.keys())[0]][2]) )    )
		yerr_bar.append(  np.mean(  np.array(dats[list(dats.keys())[0]][3])  )     )

		xs.append(np.array(dats[list(dats.keys())[0]][0]))
		x_errs.append(np.array(dats[list(dats.keys())[0]][3]))
		ys.append(np.array(dats[list(dats.keys())[0]][2]))
		y_errs.append(np.array(dats[list(dats.keys())[0]][3]))

		data_names.append(list(dats.keys())[0])

	if(verbose_data):
		print("xbar:", x_bar)
		print("xerr_bar:", xerr_bar )
		print("ybar:", y_bar )
		print("yerr_bar:", yerr_bar )

		print("xs:", xs)
		print("xerrs:", x_errs )
		print("ys:", ys )
		print("yerrs:", y_errs )
		print( data_names )

	plt.scatter(x_bar, y_bar )
	plt.errorbar(x_bar, y_bar, 
			xerr = xerr_bar, yerr=yerr_bar, 
			ls='none', capsize=4)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(title)
	plt.axvline(x=0, color = 'black')
	plt.axhline(y=0, color = 'black')    
	if(mag_line):
		for f in range(0, len(y_bar)):
			plt.plot([0,x_bar[f]], [0, y_bar[f]], 
				 color = 'grey', lw=1.75, 
				 alpha=0.4, linestyle = '--')
	plt.grid()
	plt.show()

def demo_comp(input_data,
				mod_vals, 
				verbose_data = False):

	mod_prod = cp(input_data) #Make a copy here
	data_names = []

	#Intermediary processing operation. What has been packaged needs to be unpackaged
	for dats in mod_prod:	
		if(verbose_data):
			print("Old:", np.array(dats[list(dats.keys())[0]][0]), "New:", np.array(dats[list(dats.keys())[0]][0]) - mod_vals[0])
			print("Old:", np.array(dats[list(dats.keys())[0]][2]), "New:", np.array(dats[list(dats.keys())[0]][2]) - mod_vals[1])

		dats[list(dats.keys())[0]][0] = np.array(dats[list(dats.keys())[0]][0]) - mod_vals[0]
		dats[list(dats.keys())[0]][2] = np.array(dats[list(dats.keys())[0]][2]) - mod_vals[1]

		data_names.append(list(dats.keys())[0])

	if(verbose_data):
		print("Modified data:", mod_prod)

	return(mod_prod)

def demo_plot2d(input_data, 
		title,
		verbose_data = False,
		verbose_plot_mag_line=False,
		mag_line=False):

	data_names = []

	xs = []
	x_errs = []
	ys = []
	y_errs = []

	x_bar = []
	xerr_bar = []
	y_bar = []
	yerr_bar = []

	#Intermediary processing operation. What has been packaged needs to be unpackaged
	for dats in input_data[0]:	
		x_bar.append( np.mean(np.array(dats[list(dats.keys())[0]][0]))   )
		xerr_bar.append( np.mean(np.array(dats[list(dats.keys())[0]][3])  )      )
		y_bar.append( np.mean(np.array(dats[list(dats.keys())[0]][2]) )    )
		yerr_bar.append(  np.mean(  np.array(dats[list(dats.keys())[0]][3])  )     )

		xs.append(np.array(dats[list(dats.keys())[0]][0]))
		x_errs.append(np.array(dats[list(dats.keys())[0]][3]))
		ys.append(np.array(dats[list(dats.keys())[0]][2]))
		y_errs.append(np.array(dats[list(dats.keys())[0]][3]))

		data_names.append(list(dats.keys())[0])

	if(verbose_data):
		print("xbar:", x_bar)
		print("xerr_bar:", xerr_bar )
		print("ybar:", y_bar )
		print("yerr_bar:", yerr_bar )

		print("xs:", xs)
		print("xerrs:", x_errs )
		print("ys:", ys )
		print("yerrs:", y_errs )
		print( data_names )

	plt.scatter(x_bar, y_bar)
	plt.errorbar(x_bar, y_bar, 
			xerr = xerr_bar, yerr=yerr_bar, 
			ls='none', capsize=4)

	plt.title(title)
	if(mag_line):
		for f in range(0, len(y_bar)):
			plt.plot([0,x_bar[f]], [0, y_bar[f]], 
				 color = 'grey', lw=1.75, 
				 alpha=0.4, linestyle = '--')
	plt.grid()
	plt.show()

def append_struct(input_data_a, input_data_b):
    return [np.concatenate((input_data_a[0],input_data_b[0])), np.concatenate((input_data_a[1],input_data_b[1]))]

def delete_struct(input_data_a, index):
    return [np.delete(input_data_a[0], index), np.delete(input_data_a[1], index)]
    
def function_of_time(input_data, verbose=False):
    print("Plot magnitude as a function of time")
    
    mags = []
    yerrs = []
    
    for k in range(0, len(input_data[0])):
        if(verbose):
            print(list(input_data[0][k].keys())[0] )
            print(  input_data[0][k][ list(input_data[0][k].keys())[0]  ][0]  )

            print(  input_data[0][k][ list(input_data[0][k].keys())[0]  ][2]  )        

            print(  np.linalg.norm([np.mean(input_data[0][k][ list(input_data[0][k].keys())[0]][0] ),
                                    np.mean(input_data[0][k][ list(input_data[0][k].keys())[0]][2])])  )    
        mags.append(np.linalg.norm([np.mean(input_data[0][k][ list(input_data[0][k].keys())[0]][0] ),
                                np.mean(input_data[0][k][ list(input_data[0][k].keys())[0]][2])])  )
        yerrs.append(10*np.linalg.norm([np.mean(input_data[0][k][ list(input_data[0][k].keys())[0]][1] ),
                                np.mean(input_data[0][k][ list(input_data[0][k].keys())[0]][3])])            )
        
        #print(yerrs)
        #print(input_data[1]   )
    plt.scatter([int(x.strftime('%Y%m%d')) for x in np.array(input_data)[1]  ] ,mags, marker = '.' )
    plt.errorbar([int(x.strftime('%Y%m%d')) for x in np.array(input_data)[1]  ] ,mags,
                 xerr= [0]*len(input_data[1]), yerr=yerrs, 
                 capsize=4, ls='none')
    plt.grid()
    plt.show()
        

    
def demo_comp2d(input_data, mod_vals, 
		verbose_data = False):
	
	mod_prod = cp(input_data) #Make a copy here
	data_names = []

	#Intermediary processing operation. What has been packaged needs to be unpackaged
	for dats in mod_prod[0]:	
		if(verbose_data):
			print("Old:", np.array(dats[list(dats.keys())[0]][0]), "New:", np.array(dats[list(dats.keys())[0]][0]) - mod_vals[0])
			print("Old:", np.array(dats[list(dats.keys())[0]][2]), "New:", np.array(dats[list(dats.keys())[0]][2]) - mod_vals[1])

		dats[list(dats.keys())[0]][0] = np.array(dats[list(dats.keys())[0]][0]) - mod_vals[0]
		dats[list(dats.keys())[0]][1] = np.array(dats[list(dats.keys())[0]][1])
		dats[list(dats.keys())[0]][2] = np.array(dats[list(dats.keys())[0]][2]) - mod_vals[1]
		dats[list(dats.keys())[0]][3] = np.array(dats[list(dats.keys())[0]][3])

		data_names.append(list(dats.keys())[0])

	if(verbose_data):
		print("Modified data:", mod_prod)

	return(mod_prod)

def confidence_level_categorical(input_data, obs, confl, verbose=False):
    # Specify sample occurrences (x), sample size (n) and confidence level
    x = input_data.count(obs)
    n = len(input_data)
    cl = confl #This is set by professor Fisher

    #Calculate 
    pes = x/n #point estimate
    a = (1-cl) #alpha - level of significance
    critical_z = stats.norm.ppf(1-a/2) #critical z-value
    standard_error = math.sqrt((pes*(1-pes)/n)) #standard error, 
    margin_of_error = critical_z * standard_error #margin of error

    # Calculate the lower and upper bound of the confidence interval
    lower_bound = pes - margin_of_error
    upper_bound = pes + margin_of_error

    # Print the results
    if(verbose):
        print("The confidence interval of observing observing strawberry from the distribution is", lower_bound,"and",upper_bound,
             "given a condfidence level of:", cl)
    return (upper_bound-  lower_bound)


def confidence_level_numerical(input_data, confl, verbose=False):
    # Specify sample mean (x_bar), sample standard deviation (s), sample size (n) and confidence level
    x_bar = np.mean(input_data)
    s = np.std(input_data)
    n = len(input_data)
    confidence_level = confl

    # Calculate
    #Alpha
    #Degrees of freedom (df)the critical t-value, and the margin of error
    alpha = (1-confidence_level)
    dof = n - 1
    standard_error = s/math.sqrt(n)
    critical_t = stats.t.ppf(1-alpha/2, dof)
    margin_of_error = critical_t * standard_error

    # Calculate the lower and upper bound of the confidence interval
    lower_bound = x_bar - margin_of_error
    upper_bound = x_bar + margin_of_error

    # Print the results
    if(verbose):
        print("The confidence interval of the population mean", np.round(x_bar, 4)," is", 
              np.round(lower_bound,4),"and", np.round(upper_bound,4),
             "given a condfidence level of:", confidence_level)
    
    #print("The 95% confidence interval for the mean Houshold income is",lower_bound,"and",upper_bound)