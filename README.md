## Q1.1 Abstract

This AI agent is designed to predict match outcomes in *League of Legends* based on in-game statistics. Using a dataset containing high-ranked matches with detailed in-game metrics—such as KDA, gold earned, experience gained, jungle control, and objective captures—the agent analyzes early-game conditions and predicts whether a team will win or lose. 

The performance measure of the agent is the accuracy of match predicting outcomes, whereas the environment is the statistics of League of Legends Platinum Ranked Games starting with a game time of 10 minutes and taking steps of 2 minutes (i.e. 10 minutes, 12 minutes, 14 minutes, etc.) 

- **Actuators**: Display of match outcome prediction output
- **Sensors**: Riot Games' *League of Legends* Game API.

The agent operates as a **goal-based AI**, focusing solely on maximizing the accuracy of its match outcome predictions. The strong conditional dependencies among in-game variables (e.g., kills, deaths, and assists correlating with gold and experience leads) along with the binary nature of predictions (win, loss) makes probabilistic agents such as Bayesian networks a good fit.

## Q1.2 Datasets

These are the potential datasets for training and evaluating our AI model. The top-ranked (1st) dataset is preferred, while the others are alternative options:

1. **[League of Legends Solo Queue Ranked Games](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-soloq-ranked-games)**  
    - Contains statistics from approximately 25000 ranked Solo Queue games at Platinum ELO.  
    - Each game includes multiple time frames (from 10 min, 12 min, 14 min, etc.), totaling 240,000+ game frames.  
    - Features 55 attributes for the BLUE team, including kills, deaths, gold, experience, level, and more.  
    - The `hasWon` column is the target value for our classification  

2. **[League of Legends Diamond Ranked Games (10-min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min/data)**  
    - Contains statistics from  approximately 10000 ranked Solo Queue games at Diamond I to Master ELO.  
    - Includes 19 features per team (38 in total), collected at the 10-minute mark.  
    - The `blueWins` column is the target variable (1 = Blue team wins, 0 = Loss).  

3. **[League of Legends Match Dataset 2025](https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025)**  
    - Contains 94 attributes capturing comprehensive match and player data.  
    - Key columns include `game_id`, `game_start_utc`, `game_duration`, `queue_id`, `participant_id`, `kills`, `deaths`, `assists`, `final_damageDealt`, `final_goldEarned`, etc.

## Training the First Model
The first version of the model was trained using the following:
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

The first iteration of our Bayesian network model has provided promising insights into early-game win prediction. However, our observations show areas of improvement.

# Observations
- The dataset records game statistics at 2-minute intervals. This could be an opportunity to improve the model's functionality and accuracy by implementing a time-based Markov Chain model to predict statistics for the next interval of 2 minutes (e.g. 10 minutes, 12 minutes, etc.)
- Bayesian networks may not be the best choice for this problem due to the presence of loops in important variables (e.g., gold and exp/level). Gold can be used to buy items, which have a significant impact on exp gain, KDA, objectives, and many other variables, including itself. The same issue is seen in EXP and level progression. As both gold and level progression have significant impacts on the win probability, making strict conditional independence assumptions become problematic.


While gold is *theoretically* conditionally independent from win chance given item counts (the only way gold actually effects the outcome of the match).
This is especially troublesome for Level, since 

These loops also create some extremely strong proxy statistics. For example, due to low respawn times in the early game, champion kills should be more or less conditionally independent to winning outside given their effect on gold and exp. However, due to being affected by player/team skill, they are still relevant conditions by proxy.

Gold and experience acting as weighted aggregates also means that more complicated models don't necessarily. This also means certain combos such as lower gold and higher EXP have very low amounts of data. 
-  Gold technically does nothing on its own but proxies for items, player/team skill.  
- Simplify due to strong proxy metrics

- Specificity consider item shop purchases (currently only have wards)
- Finer state splits
  This would require changing how we  Proportionalize metrics
- Alternatively, we could create a slightly different agent which functions based on the information given to a certain team rather than the information available to spectators (ex. certain info such as the enemy team's gold is not present).

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
