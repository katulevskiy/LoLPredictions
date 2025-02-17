## Q1.1 Abstract

This AI agent is designed to predict match outcomes in *League of Legends* based on in-game statistics. Using a dataset containing high-ranked matches with detailed in-game metrics—such as KDA, gold earned, experience gained, jungle control, and objective captures—the agent analyzes early-game conditions and predicts whether a team will win or lose. 

The performance measure of the agent is the accuracy of match predicting outcomes, whereas the environment is the statistics of League of Legends Platinum Ranked Games starting with a game time of 10 minutes and taking steps of 2 minutes (i.e. 10 minutes, 12 minutes, 14 minutes, etc.) 

- **Actuators**: Display of match outcome prediction output
- **Sensors**: Riot Games' *League of Legends* Game API.

The agent operates as a **goal-based AI**, focusing solely on maximizing the accuracy of its match outcome predictions. The strong conditional dependencies among in-game variables (e.g., kills, deaths, and assists correlating with gold and experience leads) along with the binary nature of predictions (win, loss) makes probabilistic agents such as Bayesian networks a good fit.

## Q1.2 Datasets

Please provide links to your proposed datasets below. You can submit up to 3, but only need 1. The top one will ranked 1st and the bottom ranked last:

1. https://www.kaggle.com/datasets/bobbyscience/league-of-legends-soloq-ranked-games

This dataset contains stats of approx. 25000 ranked games (SOLO QUEUE) from a Platinium ELO. Each game is unique. The gameId can help you to fetch more attributes from the Riot API. Each game has features from different time frames from 10min to the end of the game. For example, game1 10min, game1 12min, game1 14min etc. In total there are +240000 game frames. There are 55 features collected for the BLUE team. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights. The column hasWon is the target value if you're doing classification to predict the game outcome. Otherwise you can use the gameDuration attribute if you wanna predict the game duration. Attributes starting with is* are boolean categorial values (0 or 1).

2. https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min/data

This dataset contains the first 10min. stats of approx. 10k ranked games (SOLO QUEUE) from a high ELO (DIAMOND I to MASTER). Players have roughly the same level. Each game is unique. The gameId can help you to fetch more attributes from the Riot API. There are 19 features per team (38 in total) collected after 10min in-game. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights. The column blueWins is the target value (the value we are trying to predict). A value of 1 means the blue team has won. 0 otherwise.

3. https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025

This dataset haven 94 attributes capturing comprehensive match and player data. 
Key columns: game_id, game_start_utc, game_duration, queue_id, participant_id, kills, deaths, assists, final_damageDealt, final_goldEarned, and more.

## Q1.3 Group Members

Please provide the Name and email of your group members. Up to 6.
Please don't forget to add them to gradescope assignment as well!

Jason Cheung, jac130@ucsd.edu
Jeremy Lim, jel125@ucsd.edu 
Kevin Zheng, kezheng@ucsd.edu 
Daniil Katulevskiy, dkatulevskiy@ucsd.edu 
Eric Hu, e2hu@ucsd.edu 
