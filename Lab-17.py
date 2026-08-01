import numpy as np
import random

# Movies
movies = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"]

# User feedback
ratings = [3, 4, 2, 5, 1]

# Actor Network (Recommendation Scores)
actor = np.random.rand(len(movies))

# Critic Network (Value Estimates)
critic = np.zeros(len(movies))

learning_rate = 0.1

for episode in range(100):

    for state in range(len(movies)):

        action = actor[state]

        reward = ratings[state]

        # Critic Update
        critic[state] = critic[state] + learning_rate * (
            reward - critic[state]
        )

        # Actor Update
        actor[state] = actor[state] + learning_rate * critic[state] / 10

print("Movie Scores")

for i in range(len(movies)):
    print(movies[i], ":", round(actor[i], 2))

print("\nRecommended Movies")

order = np.argsort(actor)[::-1]

for i in order:
    print(movies[i], "Score =", round(actor[i], 2))
