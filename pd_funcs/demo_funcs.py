import pandas as pd
import numpy as np
import pickle

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

	plt.title(title)
	if(mag_line):
		for f in range(0, len(y_bar)):
			plt.plot([0,x_bar[f]], [0, y_bar[f]], 
				 color = 'grey', lw=1.75, 
				 alpha=0.4, linestyle = '--')
	plt.grid()
	plt.show()

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
		print("Dats", dats)	
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

def demo_comp(input_data, mod_vals, 
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