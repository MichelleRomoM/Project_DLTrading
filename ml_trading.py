import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.svm import SVC
import xgboost
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Import datasets
train_data = pd.read_csv("files/aapl_5m_train.csv")
validation_data = pd.read_csv("files/aapl_5m_validation.csv")

# Data preparation for logistic regression model
df = train_data.loc[:,['Datetime','Close']].set_index("Datetime")

df1 = df.shift(periods=-1)
df2 = df1.shift(periods=-1)
df3 = df2.shift(periods=-1)
df4 = df3.shift(periods=-1)

a = pd.DataFrame({})
a['X1'] = df
a['X2'] = df1
a['X3'] = df2
a['X4'] = df3
a['X5'] = df4

a["Y"] = a["X5"].shift(-5) > a["X5"]
a = a.dropna()

# Variables
x = a.drop('Y', axis = 1)
y = a['Y']
y_int = y.astype(int)

# Create binary numbers(convert 1s to 1s & 2s to 0s)
array_binario = np.where(y_int == 1, 1, 0)

# Split the data into a training set (70%) and a test set (30%)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(x, array_binario, test_size=0.3, random_state=42)

# Train the logistic model
log_model = LogisticRegression()
log_model.fit(X_train_lr, y_train_lr)

# Data preparation for svc model
rets = train_data.loc[:,['Close']].pct_change().dropna()

x = pd.DataFrame()
x['X0'] = rets.Close
x['X1'] = rets.Close.shift(-1)
x['X2'] = rets.Close.shift(-2)
x['X3'] = rets.Close.shift(-3)
x['X4'] = rets.Close.shift(-4)
x['X5'] = rets.Close.shift(-5)
x = x.dropna()

y = (train_data.Close.shift(-5) * 1.005 < train_data.Close.shift(-15))[:-6]

assert len(x) == len(y)

# Split the data into a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize the feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train svc model
svc_model = SVC(random_state =42)
svc_model.fit(X_train, y_train)

# Train xgboost with the data of the svc model
boosted_model = xgboost.XGBClassifier()
boosted_model.fit(X_train, y_train)

# Prediction and evaluation of the models
log_pred = log_model.predict(X_test_lr)
svc_pred = svc_model.predict(X_test)
xgb_pred = boosted_model.predict(X_test)

log_accuracy = accuracy_score(y_test_lr, log_pred)
svc_accuracy = accuracy_score(y_test, svc_pred)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")
print(f"SVC Accuracy: {svc_accuracy:.2f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

# Generate trading signals based on model predictions
def generate_signals(predictions):
    signals = ["Buy" if pred == 1 else "Sell" for pred in predictions]
    return signals

lr_signals = generate_signals(log_pred)
svc_signals = generate_signals(svc_pred)
xgb_signals = generate_signals(xgb_pred)

#print("Logistic Regression Signals:", lr_signals)
#print("SVC Signals:", svc_signals)
#print("XGBoost Signals:", xgb_signals)

# Combinations for strategies
n = list(range(1, 2**7))
combinations = list(map(lambda x: [int(bit) for bit in f"{x:07b}"], n))

# Backtesting
#
#
#comission = .0025
stop_loss = .025
take_profit = .025
cash = 1000000
positions=[]
operations=[]

# Prueba backtesting Gptchat
class Backtester:
    def __init__(self, signals, prices, initial_cash):
        self.signals = signals
        self.prices = prices
        self.cash = initial_cash
        self.portfolio_value = [initial_cash]
        self.positions = 0

    def backtest(self):
        for i in range(len(self.signals)):
            if self.signals[i] == "Buy":
                # Execute a buy order
                price = self.prices[i]
                max_buyable = self.cash // price
                self.positions += max_buyable
                self.cash -= max_buyable * price
            elif self.signals[i] == "Sell":
                # Execute a sell order
                price = self.prices[i]
                self.cash += self.positions * price
                self.positions = 0
            self.portfolio_value.append(self.cash + self.positions * self.prices[i])

# Create a Backtester instance
backtester = Backtester(svc_signals, train_data["Close"], initial_cash=10000)

# Backtest the strategy
backtester.backtest()

# Get the portfolio value over time
portfolio_value = backtester.portfolio_value



# Optimize parameters
# Define the parameter grid to search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  
    }
# Define a custom scorer (negative accuracy) for grid search
scorer = make_scorer(accuracy_score)

# Create a grid search object
grid_search = GridSearchCV(log_model, param_grid, cv=5, scoring=scorer)

# Perform grid search
grid_search.fit(X_train_lr, y_train_lr)

# Get the best hyperparameters
best_C = grid_search.best_params_['C']

# Train a Logistic Regression model with the best C value
optimized_logistic_model = LogisticRegression(C=best_C)
optimized_logistic_model.fit(X_train_lr, y_train_lr)

# Evaluate the model
logistic_predictions = optimized_logistic_model.predict(X_test_lr)
logistic_accuracy = accuracy_score(y_test_lr, logistic_predictions)

print(f"Optimized Logistic Regression Accuracy: {logistic_accuracy:.2f}")

def opt_params_svc(x_):
    C, gamma = x_  # unpack the parameters
    svc_model = SVC(C=C, gamma=gamma)
    svc_model.fit(X_train, y_train)
    svc_pred = svc_model.predict(X_test)
    acc = accuracy_score(y_test, svc_pred)
    
    return -acc 

def opt_params_xgb(x_:np.array) -> float:
    gamma, reg_alpha = x_ #unpack parameters
    n_estimators = 10
    
    # X Y
    boosted_model = xgboost.XGBClassifier(n_estimators=n_estimators,
                                  gamma=gamma,
                                  reg_alpha=reg_alpha,
                                  reg_lambda=reg_alpha)
    boosted_model.fit(X_train, y_train)
    xgb_pred = boosted_model.predict(X_test)
    #sharpe_ratio = calculate_sharpe_ratio(y_validation, xgb_pred)
    acc = accuracy_score(y_test, xgb_pred)
    
    return -acc

initial_C = 0.1
initial_gamma = 1.0
initial_reg_alpha = 0.01

# Optimization
svc_optimal = minimize(opt_params_svc, x0=[1.0, 0.1], bounds=[(0.01, 100.0), (0.01, 10.0)])
xgb_optimal = minimize(opt_params_xgb, x0=[0.1, 0.1])


svc_C_optimal, svc_gamma_optimal = svc_optimal.x
xgb_gamma_optimal, xgb_reg_alpha_optimal = xgb_optimal.x

# Train models with optimal parameters
svc_model = SVC(C=svc_C_optimal, gamma=svc_gamma_optimal)
xgb_model = xgboost.XGBClassifier(gamma=xgb_gamma_optimal, reg_alpha=xgb_reg_alpha_optimal)


svc_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Evaluate the models and generate signals with the optimal parameters
svc_pred = svc_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

#backtesting and startegy optimization
# Define initial cash for backtesting
initial_cash = 10000

# Create Backtester instances for each model
backtester_lr = Backtester(lr_signals, train_data["Close"], initial_cash)
backtester_svc = Backtester(svc_signals, train_data["Close"], initial_cash)
backtester_xgb = Backtester(xgb_signals, train_data["Close"], initial_cash)

# Backtest the strategies
backtester_lr.backtest()
backtester_svc.backtest()
backtester_xgb.backtest()

# Get the portfolio values over time
portfolio_value_lr = backtester_lr.portfolio_value
portfolio_value_svc = backtester_svc.portfolio_value
portfolio_value_xgb = backtester_xgb.portfolio_value

# Define a function to calculate profit
def calculate_profit(portfolio_values):
    return portfolio_values[-1] - portfolio_values[0]

# Calculate profits for each strategy
profit_lr = calculate_profit(portfolio_value_lr)
profit_svc = calculate_profit(portfolio_value_svc)
profit_xgb = calculate_profit(portfolio_value_xgb)

print(f"Logistic Regression Profit: {profit_lr:.2f}")
print(f"SVC Profit: {profit_svc:.2f}")
print(f"XGBoost Profit: {profit_xgb:.2f}")

# Use the optimal models with the validation data
svc_pred_validation = svc_model.predict(X_validation)
xgb_pred_validation = xgb_model.predict(X_validation)

# Generate signals for the validation dataset
svc_signals_validation = generate_signals(svc_pred_validation)
xgb_signals_validation = generate_signals(xgb_pred_validation)

# Create a Backtester instance for the validation dataset
backtester_svc_validation = Backtester(svc_signals_validation, validation_data["Close"], initial_cash)
backtester_xgb_validation = Backtester(xgb_signals_validation, validation_data["Close"], initial_cash)

# Backtest the strategies on the validation dataset
backtester_svc_validation.backtest()
backtester_xgb_validation.backtest()

# Get the portfolio values over time for validation
portfolio_value_svc_validation = backtester_svc_validation.portfolio_value
portfolio_value_xgb_validation = backtester_xgb_validation.portfolio_value

# Calculate profits for validation
profit_svc_validation = calculate_profit(portfolio_value_svc_validation)
profit_xgb_validation = calculate_profit(portfolio_value_xgb_validation)

print(f"SVC Profit (Validation): {profit_svc_validation:.2f}")
print(f"XGBoost Profit (Validation): {profit_xgb_validation:.2f}")

# Buy and Hold Strategy on Validation Dataset
initial_price = validation_data.iloc[0]["Close"]
final_price = validation_data.iloc[-1]["Close"]
profit_buy_hold = (final_price - initial_price) * (initial_cash / initial_price)

print(f"Buy and Hold Profit (Validation): {profit_buy_hold:.2f}")