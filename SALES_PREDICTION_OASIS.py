#!/usr/bin/env python
# coding: utf-8

# In[44]:


# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[45]:


# Loading dataset

data = pd.read_csv("Advertising.csv")


# ### EDA

# In[46]:


# Get the dimensions of the DataFrame

rows, columns = data.shape

# Create a text annotation to display the dimensions
plt.figure(figsize=(6, 4))
plt.text(0.5, 0.5, f'Rows: {rows}\nColumns: {columns}', fontsize=12, ha='center', va='center')
plt.axis('off')
plt.title('DataFrame Dimensions')
plt.show()


# In[47]:


# Display the first 10 rows of the dataset

data.head(10)


# In[48]:


# Drop the column
data.drop('Unnamed: 0', axis=1, inplace=True)


# In[49]:


# Basic dataset information

data.info()


# In[50]:


# Check for missing values

data.isnull().sum


# In[51]:


# Descriptive statistics of the dataset

data.describe()


# In[52]:


# Distribution of Sales

plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], bins=30, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# In[53]:


# Pairplot to visualize relationships between numerical features

sns.pairplot(data, vars=['TV', 'Radio', 'Newspaper', 'Sales'])
plt.title('Pairplot of Numerical Features ')
plt.show()


# In[54]:


# Correlation heatmap

plt.figure(figsize=(8, 6))
correlation_matrix = data[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[55]:


# Create subplots for each histogram

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot histograms for 'TV,' 'Radio,' and 'Newspaper' columns
data["TV"].plot.hist(ax=axes[0], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('TV Advertising Budget')
axes[0].set_xlabel('Spending')
axes[0].set_ylabel('Frequency')

data["Radio"].plot.hist(ax=axes[1], bins=10, color='lightcoral', edgecolor='black')
axes[1].set_title('Radio Advertising Budget')
axes[1].set_xlabel('Spending')
axes[1].set_ylabel('Frequency')

data["Newspaper"].plot.hist(ax=axes[2], bins=10, color='lightgreen', edgecolor='black')
axes[2].set_title('Newspaper Advertising Budget')
axes[2].set_xlabel('Spending')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# ### MODELING

# In[56]:


# Split the data into training and testing sets

from sklearn.model_selection import train_test_split
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#x_train, x_test, y_train, y_test = train_test_split(data[["TV"]], data[["Sales"]], test_size=0.3, random_state=42)
#x_test.shape
y_test.shape


# In[58]:


# Create a linear regression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# ### Model Evaluation

# In[60]:


# Make predictions on the test data

predictions = model.predict(X_test)
predictions


# In[61]:


# Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(y_test,predictions, c='blue', marker='o', alpha=0.7, label='Actual vs. Predicted Sales')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

