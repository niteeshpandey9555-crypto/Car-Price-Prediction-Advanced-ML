# Exploratory Data Analysis – Used Car Price Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/car_data.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()
plt.hist(df["Selling_Price"], bins=30)
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Count")
plt.title("Selling Price Distribution")
plt.show()
df.groupby("Year")["Selling_Price"].mean().plot()
plt.title("Average Selling Price by Year")
plt.xlabel("Year")
plt.ylabel("Price (Lakhs)")
plt.show()
### Observations
- Newer cars have higher selling prices
- Selling price decreases as vehicle age increases
