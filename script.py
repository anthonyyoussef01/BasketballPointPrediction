import pandas as pd
from IPython.display import display

url = "https://raw.githubusercontent.com/Benjamin-Wolff/BasketballPointPrediction/main/julius_randle_career_stats_by_game.csv"

data = pd.read_csv(url, error_bad_lines=False)

data_cpy = data.copy()
data_cpy.columns

data_cpy["Points"] = (data_cpy["ThreePointersMade"] * 3) + ((data_cpy["FieldGoalsMade"] - data_cpy["ThreePointersMade"]) * 2) + data_cpy["FreeThrowsMade"]
df = data_cpy.tail()[["Day", "ThreePointersMade", "FieldGoalsMade", "FreeThrowsMade", "Points"]]

df = data_cpy.columns

display(df)