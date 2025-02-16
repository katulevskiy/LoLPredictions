# AI Agent for League of Legends Match Prediction Using Bayesian Network

## Abstract
This AI agent is designed to predict match outcomes in League of Legends based on in-game statistics. Using a dataset containing high-ranked matches with detailed in-game metrics—such as minion kills, gold earned, experience gained, jungle control, and objective captures—the agent analyzes early-game conditions and predicts whether a team will win or lose. The performance measure of the agent is the accuracy of match predicting outcomes, whereas the environment is the stats of League of Legends Platinum Ranked Games (starting with minute 10 with strides of 2 minutes). Actuators are displaying match prediction output and Sensors are the Riot League of Legends Game API. The agent operates as a goal-based AI, focusing solely on maximizing the accuracy of its match outcome predictions. Given the strong conditional dependencies among in-game variables (e.g., minion kills correlating with gold and experience leads) along with the binary nature of predictions makes probabilistic agents such as Bayesian networks a good fit.

## AI Agent Type
The AI agent used in this model is a goal-based agent, meaning it operates by assessing game state information and attempting to maximize its ability to predict the winning team. The prediction is probabilistic, leveraging a Bayesian Network structure to infer likely outcomes based on early-game conditions.

## AI Agent Setup and Probabilistic Modeling
The Bayesian network is structured to capture key dependencies between game metrics, such as gold difference, experience difference, kills, deaths, and objective control. The structure is learned using a Hill Climbing Search algorithm with a Bayesian Information Criterion (BIC) Score to find the best-fitting network structure.

## Datasets

Please provide links to your proposed datasets below. You can submit up to 3, but only need 1. The top one will ranked 1st and the bottom ranked last:

1. https://www.kaggle.com/datasets/bobbyscience/league-of-legends-soloq-ranked-games

This dataset contains stats of approx. 25000 ranked games (SOLO QUEUE) from a Platinium ELO. Each game is unique. The gameId can help you to fetch more attributes from the Riot API. Each game has features from different time frames from 10min to the end of the game. For example, game1 10min, game1 12min, game1 14min etc. In total there are +240000 game frames. There are 55 features collected for the BLUE team. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights. The column hasWon is the target value if you're doing classification to predict the game outcome. Otherwise you can use the gameDuration attribute if you wanna predict the game duration. Attributes starting with is* are boolean categorial values (0 or 1).

2. https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min/data

This dataset contains the first 10min. stats of approx. 10k ranked games (SOLO QUEUE) from a high ELO (DIAMOND I to MASTER). Players have roughly the same level. Each game is unique. The gameId can help you to fetch more attributes from the Riot API. There are 19 features per team (38 in total) collected after 10min in-game. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights. The column blueWins is the target value (the value we are trying to predict). A value of 1 means the blue team has won. 0 otherwise.

3. https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025

This dataset haven 94 attributes capturing comprehensive match and player data. 
Key columns: game_id, game_start_utc, game_duration, queue_id, participant_id, kills, deaths, assists, final_damageDealt, final_goldEarned, and more.

## Training the First Model
The first version of the model was trained using:
- Bayesian network structure learning via Hill Climb Search.
- Maximum Likelihood Estimation for probability distributions.
- A dataset filtered at the 10-minute mark to capture early-game statistics.
- Feature selection based on correlation with the match outcome.

## Model Evaluation
- Accuracy Assessment: Comparing predictions against actual game results.
- Visualization: Network graphs showing dependencies between variables.
- Correlation Analysis: Heatmaps showing feature importance.

## PEAS Analysis
- Performance Measure: Accuracy of match prediction.
- Environment: In-game statistics from League of Legends Platinum Ranked Games.
- Actuators: Model predictions displayed in outputs.
- Sensors: Riot League of Legends Game API providing real-time data.

## Conclusion

The first iteration of our Bayesian network model has provided promising insights into early-game win prediction. Key takeaways include:

- ...In progress

Potential Model Improvements
- Predict 2-minute intervals using a Markov Chain approach
- Incorporate new metrics, such as lane progression
- Optimize feature selection for more granularity
- Train models using smaller data subsets (e.g., individual team perspective rather than full spectator data)



## Group Members
- Jason Cheung, jac130@ucsd.edu
- Jeremy Lim, jel125@ucsd.edu 
- Kevin Zheng, kezheng@ucsd.edu 
- Daniil Katulevskiy, dkatulevskiy@ucsd.edu 
- Eric Hu, e2hu@ucsd.edu 


### Milestone 2 Conclusion
Potential model improvements:
- Because the dataset includes information at 2 minute intervals, we could increase the functionality and potentially the accuracy of the model by converting it to a time based markov chain which attempts to predict the stats in the next 2 minutes based on the stats of the current minute.
  This would require soProportionalize metrics
  Another New metric: progress along lanes
- Bayesian network ill suited since loops intrinsically present
- These loops also create some extremely strong proxy statistics. Gold technically does nothing on its own, but proxies for items, player skill, 
- Simplify due to strong proxy metrics

- Specificity consider item shop purchases (currently only have wards)
- Finer state splits
- Alternatively, we could create a slightly different agent which functions based on the information given to a certain team rather than the information available to spectators (ex. certain info such as the enemy team's gold is not present).
