import pandas as pd
import matplotlib.pyplot as plt 
df = pd.read_csv("data.csv")



# ************************************* SINGLE PLOT *************************************
fig,ax = plt.subplots() # creates a pyplot interface 

# adding data to axes 
ax.plot(df['x'],df['y1'],color='red',marker='o',linestyle='dashed') 
ax.plot(df['x'],df['y2'],color='green',marker='*',linestyle='--')  # add another plot to the same interface (used for comparison)
ax.plot(df['x'],df['y3'],color='blue',marker='s',linestyle='-')
ax.set_xlabel('x axis') 
ax.set_ylabel('y axis')
ax.set_title("This is a demo plot")
# displaying the plot
plt.show() 


# ************************************** MULTIPLE PLOTS **************************************

# Creating multiple plots in the same interface
fig,ax = plt.subplots(2,2) # gives 4 subplots in 2 by 2 dimension 
ax[0,0].plot(df['x'],df['y1'],color='red',marker='o',linestyle='dashed')
ax[0,1].plot(df['x'],df['y2'],color='green',marker='*',linestyle='--')
ax[1,0].plot(df['x'],df['y3'],color='blue',marker='s',linestyle='-')
ax[1,1].plot(df['x'],df['y4'],color='purple',marker='^',linestyle='v')

# fig, ax = plt.subplots(2,1, sharex=True) # shares same x axis for all subplots 
# fig, ax = plt.subplots(2,1, sharey=True) # shares same y axis for all subplots 


# **************************************** TIME SERIES PLOT ************************************ 
fig, ax = plt.subplots() 
ax.plot(df.index, df['co2']) 
ax.set_xlabel('time') 
ax.set_ylabel('co2 level') 
ax.set_title("Time Series Plot") 
plt.show()

# Zooming in on a decade of data
sixties = df["1960-01-01":"1969-12-31"] 

fig, ax = plt.subplots()
ax.plot(sixties.index, sixties['co2'])
ax.set_xlabel('time')
ax.set_ylabel('co2 level')
ax.set_title("Time Series Plot")
plt.show()

# Plot two time series together 

fig, ax = plt.subplots()
ax.plot(df.index, df['co2_Kolkata'], label='Kolkata')
ax.plot(df.index, df['co2_Mumbai'], label='Mumbai')
ax.set_xlabel('time')
ax.set_ylabel('co2 level')
ax.set_title("Time Series Plot")
plt.show() 

# Using twin axes 

fig, ax = plt.subplots()
ax1 = ax.twinx()
ax1.plot(df.index, df['co2'], color='red', label='co2')
ax1.plot(df.index, df['rel_temperature'], color='green', label='rel_temperature')
ax.set_xlabel('time')
ax.set_ylabel('co2 level')
ax.set_title("Time Series Plot")
plt.show()

# ************************************* BAR PLOT ****************************************** 
fig, ax = plt.subplots() 
ax.bar(df.index, df['co2']) 
# ax.set_xticklabels(df.index,rotation=90)
ax.set_xlabel('time') 
ax.set_ylabel('co2 level') 
ax.set_title("Bar Plot") 
plt.show() 

# *************************************** HISTOGRAM PLOT ************************************* 
fig, ax = plt.subplots() 
ax.hist("Co2", df['co2'].mean()) 
ax.hist("O2", df['o2'].mean())
ax.set_ylabel('Gas Mean Value') 
ax.set_title("Histogram Plot") 
plt.show()

# more customized histogram
fig, ax = plt.subplots() 
ax.hist(df["mens_height"], label="Mens", bins=10) 
ax.hist(df["womens_height"], label="Womens", bins=10) 
ax.set_xlabel('Height') 
ax.set_ylabel('Count') 
ax.legend()
plt.show() 


# ************************************* BOX PLOT ****************************************** 
fig, ax = plt.subplots() 
ax.boxplot([df['co2'], df['O2']]) 
ax.set_xticklabels(['co2', 'O2'])
ax.set_ylabel('gas level') 
ax.set_title("Box Plot") 
plt.show()

# *************************************** SCATTER PLOT *************************************
fig, ax = plt.subplots() 
ax.scatter(df['co2'], df['rel_temperature']) 
ax.set_xlabel('co2') 
ax.set_ylabel('rel_temperature')
ax.set_title("Scatter Plot - Co2 wrt rel_temp") 
plt.show() 

# customize scatter plot 
eighties = df["1980-01-01":"1999-12-31"]
nineties = df["1990-01-01":"2009-12-31"] 

fig, ax = plt.subplots() 
ax.scatter(eighties['co2'], eighties['rel_temperature'], color='red', label='eighties') 
ax.scatter(nineties['co2'], nineties['rel_temperature'], color='green', label='nineties') 
ax.legend(loc='upper right')
ax.set_xlabel('co2') 
ax.set_ylabel('rel_temperature')
ax.set_title("Scatter Plot - Co2 wrt rel_temp") 
plt.show() 

# Choosing a style

plt.style.use('ggplot') 
# Put it on top of your imports
# other style options are: dark_background, bmh, seaborn-colorblind, fivethirtyeight, ggplot, grayscale, etc.

# Use default style 
plt.style.use('default') 


# Saving the figure to a file 
fig.savefig('myplot.jpg') # Make sure this line appears just before plt.show()
fig.savefig('myplot.png', dpi=300) # dpi = dots per inch, controls the resolution of the figure 
fig.savefig('myplot.pdf') # save as pdf
fig.savefig('myplot.svg') # save as svg 


# Setting the figure size
plt.figure(figsize=(10, 6)) # width, height 
fig.set_size_inches(10, 6) # width, height 

# End Note; Pandas + Matplotlib = Seaborn , Great!