# Readme
Getting into AI, this first project was done following a youtube video at first
and then I just started looking through the docs and trying to mess with things

## Goal 
Create a Model using Reinforcement Learning, trained on 
running it a ton and creating my own scoring method

### Scoring
Wanted to score it based on how long it survived (the pole remained upright)

- This was done by making it award based on how long it survived
- 0.10 points per 10 seconds survived (trying to shape the reward)
  - encourage it to survive longer by giving it smaller rewards closer by
- One point for each minute

