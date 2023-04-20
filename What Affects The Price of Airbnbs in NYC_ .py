#!/usr/bin/env python
# coding: utf-8

# # Project 1

# **Introduction**
# 
# Airbnb is one of the most popular options around the world when it comes to travelers trying to find places to stay during their voyage. It is a system where property owners can rent out their space to other people or travelers. This is done through the Airbnb app which provides you with many different options such as location, number of rooms, private or shared, and many more! It has become an extremely popular choice due to its fast, convenient booking and often, it offers a much lower price compared to hotels, giving people much more options when traveling.

# Since property owners worldwide are willing to rent out their properties to others, what factors about the Airbnbs in a specific location make it cheaper or more expensive compared to Airbnbs in other locations? For example, if we look at the housing markets in NYC (New York City), we can see that houses in some neighbourhoods are more expensive than others. Some factors such as distance from high-density job sectors, average neighbourhood income, or the number of amenities surrounding the home can come into play when deciding the prices. Therefore, there must be factors that decide or affect how the prices of Airbnbs are set. In this paper, I will dive into the Airbnbs in NYC and see what factors of the city affect their prices. Everyone may have different preferences when it comes to choosing their Airbnb, but we can just focus on the characteristics of the city and observe the trends.

# The data we will analyze comes from Kaggle. The given CSV file is information on all Airbnbs located in NYC during 2019 that were web-scraped off of the Airbnb website. Each row of the DataFrame consists of each Airbnb's id, name, host id, neighborhood, coordinates, price, and many more. We can use some of these columns and import other data about NYC to do a proper price analysis of these Airbnbs.

# In[1]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import matplotlib.colors as mplc
import matplotlib.patches as patches
import geopandas as gpd
get_ipython().run_line_magic('matplotlib', 'inline')
# activate plot theme


# **Data Cleaning/Loading**
# 
# I loaded the original data from Kaggle and had to gather information from other sources to find values for my x-variables. I loaded data from different sources to find the crime rate, median monthly rent rate, and the median income of neighbourhood groups of New York City. Since there were about 50,000 entries in "AB_NYC_2019.csv", I had to create a new DataFrame that contained only 5 rows with the neighbourhood's average Airbnb prices instead of individual prices for data cleaning purposes. 

# In[2]:


#create data frame of the uploaded csv and set index to neighbourhood_group first and order them by alphabetical order.
df = pd.read_csv("AB_NYC_2019.csv").sort_values(by=['neighbourhood_group'])
crime_rate = pd.read_csv('NYC Violent Crime Rate 2019.csv')
median_income = pd.read_csv('Median Income in NYC by Borough.csv')
df.head()
#with open('table.tex', "w") as f:
    #f.write(df.head(10).to_latex(index = False))

#print(df.head(14).to_latex())


# In[3]:


df = df.drop(['host_id', 'host_name', "name", 'last_review', 'reviews_per_month'], axis = 1)


# In[4]:


df2 = df[['neighbourhood_group', 'price']].groupby(by='neighbourhood_group').mean().reset_index().rename(columns={'price':'average_neighbourhood_price'})


# The data retrieved from Criminal Justice NY showed crime rate based off of counties instead of neighbourhood groups. In addition, Manhattan, Brooklyn, and Staten Island have different county names. Therefore, I dropped all the irrelevant counties, and had to replace Kings, New York and Richmond to its respective neighbourhood group names. After cleaning all the data, I was able to merge the Data Frames together. 

# In[5]:


#Read the 2019 NYC violent crime rate data
crime_rate = pd.read_csv("NYC Violent Crime Rate 2019.csv")
#drop all rows except the relevant rows that only contain crime rates from NYC counties
crime_rate = crime_rate.query("County == 'Bronx' | County == 'Kings' | County == 'Queens'| County == 'New York' | County == 'Richmond'")
crime_rate.reset_index()
#Kings county represents Brooklyn, New York county represents Manhattan, Richmond county represents Staten Island


# In[6]:


income = pd.read_csv("Median Income in NYC by Borough.csv")
income


# In[7]:


df2.isnull().any(axis=0)


# In[8]:


merge1 = pd.merge(df2, income, on = 'neighbourhood_group')


# In[9]:


#change crime rate data frame's 'county' column to match with the other data frames
#by changing each county name to each respective neighbourhood group names. 

crime_rate["neighbourhood_group"] = crime_rate["County"].str.replace("Kings", "Brooklyn").str.replace("Richmond", "Staten Island").str.replace("New York", "Manhattan")


# In[10]:


crime_rate = crime_rate.drop("County", axis = 1)


# In[11]:


merge2 = pd.merge(merge1, crime_rate, on = "neighbourhood_group")
merge2


# In[12]:


rent = pd.read_csv("nyc median rent 2019.csv")
rent


# In[13]:


merge3 = pd.merge(rent, merge2, on = "neighbourhood_group")
merge3["Median Income ($)"] = merge3["Median Income ($)"].str.replace(",", "")
merge3["Median Income ($)"] = pd.to_numeric(merge3["Median Income ($)"])
merge3["Rent Price 2019 ($)"] = merge3["Rent Price 2019 ($)"].str.replace(",", "")
merge3["Count"] = merge3["Count"].str.replace(",", "")
merge3["2019 Population"] = merge3["2019 Population"].str.replace(",", "")
merge3["Count"] = pd.to_numeric(merge3["Count"])
merge3["Rent Price 2019 ($)"] = pd.to_numeric(merge3["Rent Price 2019 ($)"])
merge3["2019 Population"] = pd.to_numeric(merge3["2019 Population"])
merge3


# **Summary Statistics**
# 
# I created summary statistics for average neighbourhood Airbnb price, NYC's crime rate and median rent. The summary statistics for average neighbourhood price shows that at a glance, Bronx has the lowest average price and Manhattan has the highest average price. This seems obvious, because Manhattan is known to be one of the biggest tourist attraction in the world and Bronx being known as a ghetto area. When looking at the median rent summary statistics, we can see that Bronx has the lowest median rent and Manhattan has the highest median rent, which is the same result as the average neighbourhood Airbnb prices. We can start this off as a basis of our observations. Looking at the crime rate, we can see that predictably, Bronx has the highest crime rate. However, although Manhattan asks for the highest rent and has the highest median income, the crime rate is relatively high. With this, we might get an idea that it may be difficult predicting the price of Airbnbs based off of the crime rates.

# In[14]:


print(merge3.describe())


# For a better view, we can also group by neighbourhood groups and look at the summary statistics for each group. Now for each variable, we can easily compare neighbourhood groups at a glance.

# In[15]:


merge3 = merge3.drop(["2019 Population"], axis = 1)


# **Plots/Figures**
# 
# First, we can seperate some dependent variables, and analyze their frequencies. This way, we can easily see how the data is distributed and know if these variables can be effective on finding our answer. For this, I will use the raw data, before the cleaning phase with all the individual Airbnbs, and NYC Counties, because the final dataframe that I have made by merging and cleaning all the data together does not have many inputs to observe frequencies. 

# In[16]:


value_counts = df['neighbourhood_group'].value_counts()
plt.bar(value_counts.index, value_counts.values, color = "orange")
plt.title('Frequncies of Airbnbs in Each Neighbourhood Group')
plt.xlabel('Neighbourhood Groups')
plt.ylabel('Frequency')

# display plot
plt.show()


# The bar graph above shows the frequencies of Airbnbs in each neighbourhood group. Right away, you can see that most of the Airbnbs in NYC are located in Manhattan and Brooklyn. This tells us that the market for Airbnb is much higher in these areas, meaning higher demand. With basic economics understanding, we know that higher demand leads to higher price. Without looking at the cleaned data that shows the average price for each neighbourhood group, readers can get a small hint of how prices may be distributed based on this graph. Now we can do the same for other variables.

# The next graph shows the frenquency of violent crimes in 2019 in each neighbourhood group. We can see that Bronx and Brooklyn have the highest amount of crime, while Staten Island has significantly less crime than any other neighbourhood groups. This makes us question whether higher crime rates discourage people from coming which lowers the price of the Airbnbs or tells us it is just a side effect of having high population density and tourist attractions? There may be many reasons why so much crime happen in the area, and its effect on housing, hotel, or Airbnb prices. It looks like we need much more context for a better analysis. 

# In[17]:


Count = merge3["Count"]
neighbours = merge3["neighbourhood_group"]
plt.bar(neighbours, Count)
plt.xlabel("Neighbourhood Groups")
plt.ylabel("Violent Crime Frequency")
plt.title("Crime in Neighbourhoods")

plt.show()


# # Project 2

# **The Message**
# 
# We want to see what characteristics of NYC affects the prices of the Airbnbs. For convenience, we grouped all the Airbnbs into neighbourhood groups. Then, we looked at the explanatory variables for each neighbourhood group. However, since there are only 5 neighbourhood groups, we have 5 observations. Therefore, we can sometimes expand it to neighbourhoods or just individual Airbnbs. Lets see the trends.

# **Plots by Subgroups**

# In[18]:


#this shows the average Airbnb price for each neighbourhood_group in NYC. 
fig, ax = plt.subplots(figsize = (5, 5))
merge3.plot(kind = "bar",ax = ax, x = "neighbourhood_group", y = "average_neighbourhood_price", legend = False)
ax.set_title("Airbnbs in Each Neighbourhood Group")
ax.set_ylabel("Average Airbnb price")


# Now, we exclude each x-variable and see its relation with the y-variable(average price). First I isolated the crime rates of each neighbourhood group and compared it to the average prices. With this graph we can see that it is hard to predict the average price with crime rates, because although Manhattan has a much higher crime rate than Staten Island and Queens, its average price is much higher. However, we can still see that Bronx, which has by far the highest crime rate, has the lowest average price. That observation is reasonable, because people tend to avoid dangerous locations, so the demand of Airbnbs in those locations would be low. 

# In[19]:


fig, ax = plt.subplots(figsize = (15, 10))
merge3.sort_values(by=['Rate'], ascending=True).plot(kind = "bar", ax=ax, x = "Rate", y = "average_neighbourhood_price", legend = False)
ax.set_title("Crime Rates of Each Neighbourhood Group")
ax.set_ylabel('Average Price ($ in 2019 value)')


# This line graph represents the relationship between rent price in NYC in 2019 and the average price of Airbnbs in each neighbourhood group. We can see a generally positive relationship between the two variables. On average, as rent price of the location increases, the average cost of Airbnb increases. 

# In[20]:


fig, ax = plt.subplots()
merge3.sort_values(by=['Rent Price 2019 ($)'], ascending=True).plot(ax = ax, x = "Rent Price 2019 ($)", y = "average_neighbourhood_price", label='NYC Airbnbs')
ax.set_title("Airbnb Price Trend Based on Rent Prices")
ax.set_ylabel('Price ($ in 2019 value)')


# This scatterplot shows the relationship, between median income of the neighbourhood and the average price. This trend looks very similar to the relationship between rent price and average Airbnb price. This can also seem reasonable, as when people gain more income, they tend to live in more costly homes. 

# In[21]:


fig, ax = plt.subplots()
merge3.sort_values(by=['Median Income ($)'], ascending=True).plot(ax = ax, x = "Median Income ($)", y = "average_neighbourhood_price", color = "red", label='NYC Airbnbs')
ax.set_title("Airbnb Price Trend Based on Median Income")
plt.ylabel("Price ($ in 2019 value)")


# 

# **Geographical Analysis**
# 
# Now, we will use GeoPandas to make geographical analysis of the Airbnbs located in NYC. We can get a glimpse of their locations and the neighbourhood groups that they are located in. First, I will create a map using the shapefile that represents the outline of NYC. Then I will create another GeoPandas DataFrame of the locations of the Airbnbs and try to plot them together.

# In[22]:


NYC = gpd.read_file('geo_export_d5c8a532-cc3c-4127-a54f-16fe9b40d513.shp')
fig, ax = plt.subplots(figsize = (15, 10))
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude))
#plot two togehter
NYC.plot(ax = ax, column = "boro_name", legend = True, alpha = 0.5, edgecolor='black')
gdf.plot(ax=ax, markersize = 0.1, color = "blue")
plt.axis('off')
plt.plot()
ax.set_title("Airbnb plots in NYC Map")


# This map gives us a great geographical representation of the Airbnbs in NYC. Right away, we can see that the Airbnb density in Manhattan is extremely high, filled with countless red marks, while Staten Island's Airbnb density is comparatively very low. By this map, we can clearly tell that Manhattan has the most tourist attractions, making the process of booking a room there look very desirable. With high demand, meets high supply. 

# Now, we can do a price analysis of the Airbnbs. We will increase price ranges by \\$100 each time and colour code them. Notice that some prices of Airbnbs can go up to \\$10,000. Therefore, we must clean some data again to prevent so much discrepency. The huge discrepency would make most of the Airbnbs in the map have the exact same colour to the point where we cannot differentiate them. Therefore, I created a ceiling of \\$700 for a better visualization.

# In[23]:


df = df.drop(df[df.price > 700].index)
fig, ax = plt.subplots(figsize = (15, 10))
#plot two togehter
NYC.plot(ax = ax, column = "boro_name", legend = True, alpha = 0.5, edgecolor='black')
gdf2 = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude))
gdf2.plot(ax=ax, column="price", legend = True, markersize = 0.5, alpha = 0.5)
ax.annotate('Price',xy=(0.76, 0.06),  xycoords='figure fraction')
plt.axis('off')
plt.plot()
ax.set_title("Price Ranges of Airbnbs in NYC")


# Now, we have a complete map that shows price ranges of all the plausible Airbnbs we want to observe that are between \\$0 to \\$700. We can see that most dots on the map are purple, meaning most Airbnbs have prices that lie within the range of \\$0 to about \\$150. However, we can see a lot of blue marks, mainly in Manhattan, and north west part of Brooklyn, which offer prices from \\$200 to \\$300. When we zoom in closer towards Manhattan, we can see some Airbnbs that have a light green or yellow mark, which offer hefty prices (\\$600 ~ \$700).

# # Project 3

# **Potential Data to Scrape**
# 
# Some of the most useful data to use when making a price analysis of a product is looking at what its competitors have to offer. In our economy, companies that are in similar industries always compete each other, trying to offer the best price and quality to their customers to gain higher market share. Therefore, companies try to adjust their prices relative to their competitor's price all the time. Some of Airbnb's greatest competitions are other travel booking websites such as Vrbo, Booking.com, Tripadvisor, and Agoda. We can potentially webscrape the listings of properties for rentals, look at the prices they offer to observe a trend between price of Airbnbs in NYC and price of listing from other websites in NYC. We can also try looking at the distance between the Airbnbs and the popular amenities that NYC offer such as the airport, Empire State Building, Times Square, sports arenas, and Central Park. 

# **Potential Challenges**
# 
# Web scraping the listings off of different websites to compare prices can be logistically very difficult. The data of the Airbnbs provided by Kaggle was from the year 2019. Currently, we are in the year 2023, and it would not be the greatest idea to compare the current listings to the 2019 Airbnbs. Firstly, we cannot travel back in time to look at the properties listed for rent on other websites back in 2019. Secondly, on average, inflation rate has been increasing by 1.5 percentage points every year since 2019. Therefore, the listed properties in current time are likely to be much more expensive. Although this data collection can be very useful for our price analysis, we cannot collect them. Even if we did, the price comparison would be extremely biased.

# **Scraping Data Off from the Website**
# 
# Due to the challenges of the potential data to scrape, we will dive deeper into household income in NYC. Using the HTML web scraping method, we can look at the median household income of all neighbourhoods. 

# In[24]:


import requests
from bs4 import BeautifulSoup

url = "https://statisticalatlas.com/place/New-York/New-York/Household-Income"
response = requests.get(url)
soup = BeautifulSoup(response.content)
table = soup.find_all("div", "figure-container")[11]  #the 16th table on the website is the 11th element of soup.find_all
all_values = table.find_all('text', {'fill-opacity': "0.400"}) #get income value from the table
all_values2 = table.find_all('text', {"font-family":'sans-serif'}) #get neighbourhood for corresponding income value
ix = 0 # Initialise index to zero
income_df = pd.DataFrame(columns = ['neighbourhood', "median income ($1000's)"]) # Create an empty dataframe

for row, row2 in zip(all_values, all_values2):  #loop through each row of two all_values' due to its element structure.
    neighbour = row2.text
    value = row.text  #get the text value of each row.
    income_df.loc[ix] = [neighbour, value]
    ix += 1
income_df.head(15)


# In[25]:


income_df["median income ($1000's)"] = income_df["median income ($1000's)"].astype('str') 
income_df["median income ($1000's)"] = income_df["median income ($1000's)"].str.replace("$", "")
income_df["median income ($1000's)"] = income_df["median income ($1000's)"].str.replace("k", "")
income_df["median income ($1000's)"] = pd.to_numeric(income_df["median income ($1000's)"])


# In[26]:


fig, ax = plt.subplots(figsize=(12, 8))
income_df.plot(kind = "barh", y = "median income ($1000's)", x = 'neighbourhood', ax = ax, width = 0.5)
plt.yticks(size = 7)
ax.set_title('NYC Neighbourhood Median Income')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[27]:


scrape_merge = pd.merge(df, income_df, on = "neighbourhood")
scrape_merge.head(10)


# With this dataset that has the scraped data merged with the original data, we can look deeper into the income-price relationship by looking at each 39 unique neighbourhoods in NYC.
# We will create a scatterplot to represent the relationship with a regression line.

# In[28]:


fig, ax = plt.subplots()
scrape_merge.plot(kind = "scatter", x = "median income ($1000's)", y = "price", ax = ax, s = 1)
X = scrape_merge["median income ($1000's)"]
Y = scrape_merge["price"]
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),
         color='black')
ax.set_title("Airbnb Price Based on Nieghbourhood's Income ")


# This scatter plot consists of about 20,000 observations. It may not look useful to create a proper relationship as each plots are distributed by specific neighbourhoods with specific median incomes, so there are no points everywhere throughout all the x-values. However, we can see the distribution of prices of the Airbnbs of each neighbourhood. It is clear that for all neighbourhoods with less than \\$80k median income, almost all of their Airbnbs are in the $0-$300 range. However, the neighbourhoods with over \\$100k median income have quite a number of Airbnbs that are much more expensive. We can have a cleaner observation if we average out the price of Airbnbs of each neighbourhood. Alhough it reduces the total observation count, it is easy to see the general relationship at a glance.

# In[29]:


avg_price = scrape_merge[['neighbourhood', 'price', "median income ($1000's)"]].groupby(by='neighbourhood').mean().reset_index().rename(columns={'price':'average_neighbourhood_price'})
fig, ax = plt.subplots()
avg_price.plot(kind = "scatter", x = "median income ($1000's)", y = "average_neighbourhood_price", ax = ax, s = 5)
X = avg_price["median income ($1000's)"]
Y = avg_price["average_neighbourhood_price"]
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),
         color='black')
ax.set_title("Average Price of Airbnb's Based on Their Neighbourhood's Income")


# **Merging With New Dataset**
# 
# I decided that a dataset that can compliment my existing dataset is the population density per square mile of each neighbourhood in NYC. Using urban economics theory, we can tell that generally, neighbourhoods with higher population density implies greater desire to live in that area as it offers better homes, education, job positions, prosperity, public transportation, and other amenities. We also know that with higher demand for people to live in that area, prices of properties in that neighbourhood tends to increase. Therefore, we can make an observation of how population density in certain neighbourhoods correlate to the price of Airbnbs. We will use web scraping method again to import the dataset.

# In[30]:


url_2 = "https://statisticalatlas.com/place/New-York/New-York/Population"
response2 = requests.get(url_2)
soup3 = BeautifulSoup(response2.content)
table3 = soup3.find_all("div", "figure-container")[1]
pop_values = table3.find_all('text', {'fill-opacity': "0.400"})
neigh_values = table3.find_all('text', {"font-family":'sans-serif'})
index = 0

pop_df = pd.DataFrame(columns = ['neighbourhood', "pop_density in sqr m (by 1000's)"])
for row1, row2 in zip(pop_values, neigh_values):
    pop = row1.text
    neighbourhood = row2.text
    pop_df.loc[index] = [neighbourhood, pop]
    index = index + 1
    
pop_df["pop_density in sqr m (by 1000's)"] = pop_df["pop_density in sqr m (by 1000's)"].astype('str') 
pop_df["pop_density in sqr m (by 1000's)"] = pop_df["pop_density in sqr m (by 1000's)"].str.replace("k", "")
pop_df["pop_density in sqr m (by 1000's)"] = pd.to_numeric(pop_df["pop_density in sqr m (by 1000's)"])

pop_df.head(10)


# In[31]:


pop_df.dtypes


# In[32]:


fig, ax = plt.subplots(figsize=(12, 8))
pop_df.plot(kind = "barh", y = "pop_density in sqr m (by 1000's)", x = 'neighbourhood', ax = ax, width = 0.5)
plt.yticks(size = 7)
ax.set_title('NYC Neighbourhood Population Density')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[33]:


scrape_merge2 = pd.merge(scrape_merge, pop_df, on = "neighbourhood")
scrape_merge2.head(10)


# In[34]:


fig, ax = plt.subplots()
scrape_merge2.plot(kind = "scatter", x = "pop_density in sqr m (by 1000's)", y = "median income ($1000's)", ax = ax, s = 3)
X = scrape_merge2["pop_density in sqr m (by 1000's)"]
Y = scrape_merge2["median income ($1000's)"]
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),
         color='black')
ax.set_title("Relationship Between Income and Population Density")


# In[35]:


fig, ax = plt.subplots()
scrape_merge2.plot(kind = "scatter", x = "pop_density in sqr m (by 1000's)", y = "price", ax = ax, s = 1, color = "orange")
X = scrape_merge2["pop_density in sqr m (by 1000's)"]
Y = scrape_merge2["price"]
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),
         color='black')
ax.set_title("Price Observations based on Neighbourhood's Pop Density")


# In[36]:


avg_dens = scrape_merge2[['neighbourhood', 'price', "pop_density in sqr m (by 1000's)"]].groupby(by='neighbourhood').mean().reset_index().rename(columns={'price':'average_neighbourhood_price'})
fig, ax = plt.subplots()
avg_dens.plot(kind = "scatter", x = "pop_density in sqr m (by 1000's)", y = "average_neighbourhood_price", ax = ax, s = 5)
X = avg_dens["pop_density in sqr m (by 1000's)"]
Y = avg_dens["average_neighbourhood_price"]
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),
         color='black')
ax.set_title("Average Price of Airbnbs Based on Their Neighbourhood's Pop Density")


# In[37]:


emp_url = "https://statisticalatlas.com/place/New-York/New-York/Employment-Status"
response_emp = requests.get(emp_url)
soup5 = BeautifulSoup(response_emp.content)
table5 = soup5.find_all("div", 'figure-container')[15]
emp_values = table5.find_all('text', {'fill-opacity': "0.400"})
neigh_values = table5.find_all('text', {"font-family":'sans-serif'})
index = 0

emp_df = pd.DataFrame(columns = ['neighbourhood', "employment rate (%)"])
for row1, row2 in zip(emp_values, neigh_values):
    emp = row1.text
    neighbourhood = row2.text
    emp_df.loc[index] = [neighbourhood, emp]
    index = index + 1
emp_df["employment rate (%)"] = emp_df["employment rate (%)"].astype('str') 
emp_df["employment rate (%)"] = emp_df["employment rate (%)"].str.replace("%", "")
emp_df["employment rate (%)"] = pd.to_numeric(emp_df["employment rate (%)"])

emp_df.head(8)


# In[38]:


scrape_merge3 = pd.merge(scrape_merge2, emp_df, on = "neighbourhood")


# In[39]:


avg_emp = scrape_merge3[['neighbourhood', 'price', "employment rate (%)"]].groupby(by='neighbourhood').mean().reset_index().rename(columns={'price':'average_neighbourhood_price'})
fig, ax = plt.subplots()
avg_emp.plot(kind = "scatter", x = "employment rate (%)", y = "average_neighbourhood_price", ax = ax, s = 5)
X = avg_emp["employment rate (%)"]
Y = avg_emp["average_neighbourhood_price"]
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)),
         color='black')
ax.set_title("Average Airbnb Price Based on Neighbourhood's Employment Rate")


# In[43]:


Final = scrape_merge3.drop(['room_type',"calculated_host_listings_count", "availability_365", "minimum_nights", "number_of_reviews"],axis=1).copy()
Final.head()


# **OLS Regression**
# 
# Recapping back to the beginning of the paper, we chose crime, income, and rent prices of NYC to be our explanatory (X) variables and the price of Airbnbs to be our dependent (Y) variable. We will see if these explanatory variables create a linear or non-linear relationship with the Y variable.
# 
# The crime rate's relationship with the Airbnb prices does not show any consistency. When looking at the crime rate of each neighbourhood group in NYC, we can see that Bronx has by far the highest violent crime rate in the city and it offers the lowest average price. However, it is also easy to see that Manhattan and Brooklyn also have higher crime rates than Queens and Staten Island, but offers much higher average price than them. With this contradiction, we cannot see any negative or a positive linear relationship. Therefore, we cannot conclude that crime rate and Airbnb price have a linear relationship.
# 
# When it comes to income, it was easy to predict that people with higher income generally live in areas that have properties that are generally more expensive than others. By the charts we created, we can see that there is a strong positive linear relationship between median income and Airbnb prices, as properties are more expensive in high-income areas. Economically this makes sense, because as more people with high income gather up in an area, the demand for properties in that location tends to increase.
# 
# It was also obvious that rent prices have a linear relationship with Airbnb prices. As average rent price increase in an area, it is trivial to think that property owners would also offer to lend their properties at a higher price as shown in the graphs. Therefore, average rent prices of neighbourhoods also has a positive linear relationship with Airbnb prices. 
# 
# Lastly, as we merged our dataset with a new dataset containing the popular density of all neighbourhoods in NYC, we could see that higher popular density implied that the demand of homes in the area increases, therefore increasing the average price of the properties, which leads to increase in average Airbnb prices. Through the graph and our economic analysis, we can conclude that population density and Airbnb price has a positive linear relationship.

# The X's that I will choose are median income, population density, and employment rate. Using statistical modeling, we will run four seperate regressions.

# In[42]:


get_ipython().system('pip install linearmodels')
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS
get_ipython().system(' pip install stargazer')


# In[43]:


scrape_merge3['constant']= 1
X1 = ['constant', "median income ($1000's)", 'employment rate (%)']
X2 = ['constant', "pop_density in sqr m (by 1000's)", 'employment rate (%)']
X3= ['constant', "median income ($1000's)", "pop_density in sqr m (by 1000's)"]
X4= ['constant', "median income ($1000's)", "pop_density in sqr m (by 1000's)", 'employment rate (%)']

reg1 = sm.OLS(endog=scrape_merge3['price'], exog=scrape_merge3[X1], missing = 'drop').fit()
reg2 = sm.OLS(endog=scrape_merge3['price'], exog=scrape_merge3[X2], missing = 'drop').fit()
reg3 = sm.OLS(endog=scrape_merge3['price'], exog=scrape_merge3[X3], missing = 'drop').fit()
reg4 = sm.OLS(endog=scrape_merge3['price'], exog=scrape_merge3[X4], missing = 'drop').fit()


# In[44]:


from statsmodels.iolib.summary2 import summary_col
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}", 
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[reg1,reg2,reg3,reg4],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 1',
                                         'Model 2',
                                         'Model 3',
                                         'Model 4'],
                            info_dict=info_dict,
                            regressor_order=['constant',
                                             "median income ($1000's)",
                                             "pop_density in sqr m (by 1000's)",
                                             'employment rate (%)'])

#results_table.add_title('Table 2 - OLS Regressions')

print(results_table)


# In[45]:


from stargazer.stargazer import Stargazer
from IPython.core.display import HTML

stargazer = Stargazer([reg1, reg2, reg3, reg4])

HTML(stargazer.render_html())


# These four exaplanatory variables were chosen as they were the best candidates that would help explain the increase or decrease in the price of Airbnbs in NYC. Other variables did not serve as a great explanatory variables, because they either did not have any hint of a linear relationship or felt redundant as other variables are able to explain the prices much better. For example, median rent price does not need to be included as a part of the slope of the line as it has a very similar effect to the median income of the neighbourhood. Anyone with basic economics knowledge knows that neighbourhoods with higher average income implies that it offers properties that are more expensive. Therefore, median rent price is redundant. As we saw previously, crime rates in NYC almost has no relationship with the price of Airbnbs, so we do not include crime rate as one of our regression variables. 

# By observing the Stargazer regression table, we can make interpretations of each regression. The interpretation of the row of "constant" can be dismissed as they do not hold much meaning at all, because it is just a column of 1's. By looking at the first regression conducted, it is estimated that every 1\% increase in employment rate results in about \\$2.42 decrease in the price of Airbnbs in NYC and every and every \\$1000 increase in median income results in about \$1.54 in the price of Airbnbs. 
# 
# The second regression tells us that it is estimated that every 1\% increase in employment rate results in about \\$0.89 increase in the prices of Airbnbs. Meanwhile, on average, every 1000 people per square mile increase in the neighbourhood results in about 0.492 increase in the prices.
# 
# The third regression shows that when median income and population density are together in the regression, it is estimated that \$1000 increase in income results to about \\$1.16 increase in Airbnb price and every 1000 people per square mile increase results to about \\$0.389 increase.
# 
# Finally, when all the wanted explanatory variables are put together, we estimate that every 1\% increase in employment rate results in about \\$1.73 decrease in Airbnb price, every \\$1000 income increase results in about \\$1.42 increase in prices, and every 1000 people per square mile increase in the city results in about \\$0.26 increase.

# With this result we can learn that income in the neighbourhoods is the biggest driving factor when it comes to affecting the prices of Airbnbs in New York City. No matter which variables it is paired up with, it has a consistent slope that is well over 1. Unexpectedly, we see that employment rate's regression value fluctuates depending on which variables it is paired with as it has a negative and a positive value. This might suggest that employment rate of neighbourhoods is not a great variable when it comes to explaining the price of Airbnbs. Lastly, population density always has a positive slope but did not yield as much value as expected.

# **Machine Learning**

# In[46]:


import seaborn as sns
from sklearn import (
    linear_model, metrics, pipeline, model_selection
)
from sklearn import tree


# In[48]:


y = scrape_merge3['price']
X = scrape_merge3.drop(['id','neighbourhood_group','neighbourhood','price','latitude','longitude', 'room_type','number_of_reviews',"minimum_nights", "calculated_host_listings_count", "availability_365","geometry"],axis=1).copy()
sqft_tree = tree.DecisionTreeRegressor(max_depth=3).fit(X,y)
y_pred_tree = sqft_tree.predict(X)
#drop all columns that are not included in the desired regression.

# find the error of prediction (MSE)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred_tree))


# In[56]:


sqrf_fig = plt.figure(figsize=(25,20))
sqrf_fig = tree.plot_tree(sqft_tree, feature_names=X.columns, filled=True)


# The regression tree divides the data into whether the neighbourhood that the Airbnb is located in has lower than or equal to $53,100 on the first node. Then it divides into subcategories based on the population density and employment rate. The first node shows that the median income of a neighbourhood is the biggest factor when it comes to deciding the prices of Airbnbs.

# **Conclusion**
# 
# So, what characteristics of New York City affect the price of its Airbnbs? After analyzing all the Airbnbs, their locations, and their characteristics, we find that Brooklyn and Manhattan are the hottest spots, telling us that the demand for Airbnbs in those areas are the highest which leads to higher prices. We find that there are some variables that can help us predict the price of Airbnbs, and they were median neighbourhood income, population density, and employment rate. By analyzing each variable by neighbourhoods and dividing all the Airbnbs correspondingly, helped us see trends and relationships they have with prices. We find that median income has the strongest relationship with prices and employment rate has the weakest. I thought population density can be a big deciding factor for prices, but it unexpectedly had a relatively weaker relationship. Some variables such as crime rate. Other variables such as crime rate and median rent price did not serve as a great explanatory variable, because they either did not have any hint of a linear relationship or felt redundant as other variables are able to explain the prices much better. Therefore, it is important for tourists to observe the characteristics of regions of the city to decide where to stay.

# **Sources**
# 1. https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm
# 
# 2. https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?resource=download
# 
# 3. https://data.cccnewyork.org/data/map/66/median-incomes#66/39/2/107/62/a/a
# 
# 4. https://www.criminaljustice.ny.gov/crimnet/ojsa/stats.htm
# 
# 5. https://ny.curbed.com/2019/12/16/20994162/nyc-decade-in-review-rental-market-queens-bronx-corona
# 
# 
