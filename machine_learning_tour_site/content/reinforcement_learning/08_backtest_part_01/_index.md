---
title: "DQN based backtests"
date: 2019-05-26T14:16:00-03:00
draft: false
weight: 2080
---
On this section are analyzed 18 years backtest with DQN based reinforcement learning models. As each model takes several
hours to train, and for each backtest has to be trained at least 18 models, several of the models were trained in
parallel and then are loaded with this notebook. The developed backtest infrastructures manage all the logic efficiently.
It means, if a model is ran for first time, it will be trained, but if the was trained previously, it will be loaded but
not trained.
