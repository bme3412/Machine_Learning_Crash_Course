import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool

# Generate synthetic data for player stats
num_players = 1000
player_heights = np.random.normal(loc=200, scale=10, size=num_players)  # Heights in cm
player_weights = np.random.normal(loc=90, scale=10, size=num_players)  # Weights in kg
player_ages = np.random.randint(18, 40, size=num_players)  # Ages
player_positions = np.random.choice(['Guard', 'Forward', 'Center'], size=num_players)
player_minutes_played = np.random.randint(0, 2500, size=num_players)  # Minutes played in a season
player_points_scored = player_minutes_played * np.random.normal(loc=0.4, scale=0.1, size=num_players)

# Convert positions to numeric labels for logistic regression
position_labels = {'Guard': 0, 'Forward': 1, 'Center': 2}
player_position_labels = np.array([position_labels[pos] for pos in player_positions])

# Linear Regression: Predict points scored based on minutes played
lin_reg = LinearRegression()
lin_reg.fit(player_minutes_played.reshape(-1, 1), player_points_scored)
predicted_points = lin_reg.predict(player_minutes_played.reshape(-1, 1))
print("Linear Regression - Predicted Points:")
print(predicted_points[:5])

# Polynomial Regression: Predict points scored based on minutes played (quadratic)
poly_features = PolynomialFeatures(degree=2)
player_minutes_poly = poly_features.fit_transform(player_minutes_played.reshape(-1, 1))
poly_reg = LinearRegression()
poly_reg.fit(player_minutes_poly, player_points_scored)
predicted_points_poly = poly_reg.predict(player_minutes_poly)
print("\nPolynomial Regression - Predicted Points:")
print(predicted_points_poly[:5])

# Logistic Regression: Predict player position based on height and weight
log_reg = LogisticRegression(multi_class='multinomial')
log_reg.fit(np.column_stack((player_heights, player_weights)), player_position_labels)
predicted_positions = log_reg.predict(np.column_stack((player_heights, player_weights)))
print("\nLogistic Regression - Predicted Positions:")
print(predicted_positions[:5])

plt.style.use('fivethirtyeight')

# Plotting Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(player_minutes_played, player_points_scored, color='blue', label='Actual')
plt.plot(player_minutes_played, predicted_points, color='red', label='Predicted')
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.title('Linear Regression: Points Scored vs. Minutes Played')
plt.legend()
plt.show()

# Plotting Polynomial Regression
plt.figure(figsize=(8, 6))
plt.scatter(player_minutes_played, player_points_scored, color='blue', label='Actual')
plt.plot(player_minutes_played, predicted_points_poly, color='red', label='Predicted')
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.title('Polynomial Regression: Points Scored vs. Minutes Played')
plt.legend()
plt.show()

# Plotting Logistic Regression
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
labels = ['Guard', 'Forward', 'Center']
for i in range(3):
    mask = player_position_labels == i
    plt.scatter(player_heights[mask], player_weights[mask], color=colors[i], label=labels[i])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Logistic Regression: Player Positions')
plt.legend()
plt.show()


#### Ploting with bokeh
# Prepare data for Bokeh plotting
data = {
    'minutes_played': player_minutes_played,
    'points_scored': player_points_scored,
    'predicted_points': predicted_points,
    'predicted_points_poly': predicted_points_poly,
    'height': player_heights,
    'weight': player_weights,
    'position': player_positions
}

# Plotting Linear Regression with Bokeh
p_linear = figure(title='Linear Regression: Points Scored vs. Minutes Played',
                  x_axis_label='Minutes Played', y_axis_label='Points Scored')
p_linear.scatter('minutes_played', 'points_scored', source=data, color='blue', legend_label='Actual')
p_linear.line('minutes_played', 'predicted_points', source=data, color='red', legend_label='Predicted')
p_linear.add_tools(HoverTool(tooltips=[('Minutes Played', '@minutes_played'), ('Points Scored', '@points_scored')]))

# Plotting Polynomial Regression with Bokeh
p_poly = figure(title='Polynomial Regression: Points Scored vs. Minutes Played',
                x_axis_label='Minutes Played', y_axis_label='Points Scored')
p_poly.scatter('minutes_played', 'points_scored', source=data, color='blue', legend_label='Actual')
p_poly.line('minutes_played', 'predicted_points_poly', source=data, color='red', legend_label='Predicted')
p_poly.add_tools(HoverTool(tooltips=[('Minutes Played', '@minutes_played'), ('Points Scored', '@points_scored')]))

# Plotting Logistic Regression with Bokeh
p_logistic = figure(title='Logistic Regression: Player Positions',
                    x_axis_label='Height (cm)', y_axis_label='Weight (kg)')
colors = ['red', 'green', 'blue']
labels = ['Guard', 'Forward', 'Center']
for i in range(3):
    mask = player_position_labels == i
    p_logistic.scatter(player_heights[mask], player_weights[mask], color=colors[i], legend_label=labels[i])
p_logistic.add_tools(HoverTool(tooltips=[('Height', '@height'), ('Weight', '@weight'), ('Position', '@position')]))

# Output and display the plots
output_file('basketball_regression_plots.html')
show(p_linear)
show(p_poly)
show(p_logistic)