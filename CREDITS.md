Grok (xAI). (2025). Hidden Markov Model Implementation for Predicting League of Legends Match Outcomes. Developed in collaboration with [Jason Cheung, Jeremy Lim, Kevin Zheng, Daniil Katulevskiy, Petra Hu] for an AI project on game outcome prediction using the lol_ranked_games.csv dataset. Code generated and refined through interactive discussions on March 16, 2025.

Prompt:
"Hi, I'm currently considering an approach to predict outcomes in League of Legends ranked games using a Hidden Markov Model (HMM), instead of a Bayesian Network (BN) like last time. What do you think? I'm thinking of to employ the Viterbi algorithm to determine the most likely sequence of hidden states, such as 'Disadvantaged,' 'Even,' or 'Advantageous,' and predict 'hasWon' based on the final state. 

Additionally, what do you think if I evaluate the model’s performance across different frame ranges, all frames from 10 onward, as well as restricted ranges of 10-20 and 10-30—to assess the impact of game timing?"