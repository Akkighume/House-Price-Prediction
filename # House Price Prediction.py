import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fetch the Boston housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Create a pandas dataframe
df = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

# Add the target column
df['MEDV'] = target

# Data exploration and preprocessing
print(df.head())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

# Visualizations
sns.boxplot(data=df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Create KDE plots for different variables
sns.kdeplot(data=df['NOX'], shade=True, label='NOX')
sns.kdeplot(data=df['RM'], shade=True, label='RM')
sns.kdeplot(data=df['DIS'], shade=True, label='DIS')
sns.kdeplot(data=df['PTRATIO'], shade=True, label='PTRATIO')
sns.kdeplot(data=df['LSTAT'], shade=True, label='LSTAT')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('KDE Plot of Different Variables')

# Add legend
plt.legend()

# Show KDE plot
plt.show()

# Normalization of data
normalized_df = df.copy()
scaler = MinMaxScaler()
normalized_df[df.columns[:-1]] = scaler.fit_transform(normalized_df[df.columns[:-1]])

# Modeling
X = normalized_df[['NOX', 'RM', 'DIS', 'PTRATIO', 'LSTAT']]
y = normalized_df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Evaluation
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Plot regression line and scatter plot
plt.plot(X_test['NOX'], regressor.coef_[0]*X_test['NOX'] + regressor.intercept_, color='red')
plt.scatter(X_test['NOX'], y_test, color='blue')
plt.xlabel('Nitric Oxide Content')
plt.ylabel('Median Value')
plt.show()

# Create pairplot
sns.pairplot(df, x_vars=['NOX', 'RM', 'DIS', 'PTRATIO', 'LSTAT'], y_vars='MEDV', height=5, aspect=0.7, kind='reg')
plt.show()
