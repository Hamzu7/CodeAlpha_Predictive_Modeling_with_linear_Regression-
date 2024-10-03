import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    'Rooms': [1, 2, 3, 4, 5, 6, 7, 8],
    'Area': [400, 500, 600, 700, 800, 900, 1000, 1100],
    'Price': [100, 150, 200, 250, 300, 350, 400, 450]
}


df = pd.DataFrame(data)


X = df[['Rooms', 'Area']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


predictions_df = pd.DataFrame(X_test, columns=['Rooms', 'Area'])
predictions_df['Predicted Price'] = y_pred

print(predictions_df)
