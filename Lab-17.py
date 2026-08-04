# Develop a recommendation system for a streaming service to suggest movies based on user
# feedback, implemented as an MDP and trained using a Deep Deterministic Policy Gradient
# (DDPG) algorithm.
#26

import numpy as np

# Number of Movies
n = int(input("Enter Number of Movies: "))

movies = []
ratings = []

# Movie Names and Ratings
for i in range(n):
    movie = input(f"Enter Movie {i+1} Name: ")
    rating = float(input(f"Enter Rating for {movie} (1-5): "))
    movies.append(movie)
    ratings.append(rating)

# Learning Parameters
learning_rate = float(input("Enter Learning Rate (e.g., 0.1): "))
episodes = int(input("Enter Number of Episodes: "))

# Actor Network (Recommendation Scores)
actor = np.random.rand(n)

# Critic Network (Value Estimates)
critic = np.zeros(n)

# Actor-Critic Training
for episode in range(episodes):

    for state in range(n):

        action = actor[state]

        reward = ratings[state]

        # Critic Update
        critic[state] = critic[state] + learning_rate * (
            reward - critic[state]
        )

        # Actor Update
        actor[state] = actor[state] + learning_rate * critic[state] / 10

# Display Movie Scores
print("\nMovie Scores")
for i in range(n):
    print(f"{movies[i]} : {round(actor[i], 2)}")

# Display Recommended Movies
print("\nRecommended Movies")

order = np.argsort(actor)[::-1]

for i in order:
    print(f"{movies[i]}  Score = {round(actor[i], 2)}")
