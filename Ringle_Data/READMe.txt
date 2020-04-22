Ringle_DataFrame.py READEme file.

This program reads in the Ringle data from an excel sheet and puts it into a
pandas dataframe. The dataframe makes the data analysis incredibly easy as the program will show.


------Program Walkthrough------
Lines 22-32
The beginning lines use warp to retrieve the mass of the element Yttrium. Lines 31
and 32 read in the data from an excel sheet I made. Column_list will be the column names
of the data frame which is specified as an input in pd.read_excel(..., names = column_list).

Lines 35 - 132
These lines are pretty self-explanatory. They are just function for calculating the different
statistics I want to look at. It was easier to specify each function individually so that after
running the code I can look at each chunk of statistics alone rather than as a table of data.
The final line creates a copy of the data that will be used in generating the plots which has some
bit of code that is not so straight forward. Note that the prepartion() function creates new
columns in the data frame. This is easy to do by naming the dataframe column

dataframe['new column'] = function or whatever for column

The new columns created here are the kinetic energy for the x, y, and z motions.
Also, the z(m) columns is corrected to adjust for the beam motion.


Lines 134-156 Creating energy distribution plots
Here, I wanted to create histograms of the kinetic energy for each axis of motion.
First, the figure and plots are created. The plots are then added on the figure
with lines 143 and 144. Pandas has a built-in functionality where you can specify the
plot you want to add a histogram too. To create the histogram, the columns of the dataframe
have an attribute .hist() that will create a histogram of the data. The rest is typical
figure manicuring.
Line 138&139 is a neat trick that creates an invisible plot covering the whole plot area.This
allows me to center the x-axis label so that it looks nicer.

Lines 159-200 Creating temperature distribution plots
Lines 161-164 creates a three new columns: one for the deviation in the x-velocity, one for this deviation squared
and one called scaled_x-temps which is each particles temperature divided by the number of particles (100,000).

Lines 164-172 then creates the histogram.

Lines 173-200
This part of the code scrapes the tail end of the data and creates the percentage histogram.
First, the bins are created in line 176.
Lines 177 is the method for scraping data. This line creates a new data frame that matches the condition.
The condition is data['scaled_x-temps'] < .1. then by calling

data[data['scaled_x-temps'] <.1]

you will create a new dataframe where all the data meeting that condition is kept.
Lines 178 and 179 then creates a new data array of just the scaled_x-temps and redefines the sample
number N.

Lines 181-190
These lines create the weights that will be the percentage of the data. Line 183 is the conditional
statement that selects the data within each bin range. Following this, the weights are
then created and appended and printed out for to make sure it is looking correct.

Finally lines 192-200 create the histogram.


------Running the program----
The plots can be commented out out as to not be cumbersome. Once the program is run, data analysis
or debugging can be done in the interactive terminal. The program will create the prepared dataframe
called data. From here, the different functions can be called on this dataframe which will return the desired
statistics. Note that the columns created in the plotting routines will not be in the dataframe and will have to be either uncommented and
run again or created in the interactive terminal. 
