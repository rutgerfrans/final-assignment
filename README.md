# Reinforcement Learning - Final Assignment
Rutger de Groen - i6297772

## Introduction
Carracing environments in AI were one of the first things that intrigued me about AI. "How the hell does this 2D car learn to drive this weird maze in several 100 generations?", i was baffled, did not understand this at all. Now 6 years later, following a course called Reinforcement Learning, I got the opportunity (Read: I was lazy for 6 years) to delve deeper into this matter. This final assignment implements a DQN algorithm to the carracing environment provided by Gymnasium (CITE) to teach a car drive a circuit. 

## Problem Statement or Research Statement
Ofcours I am not the first one trying to teach this car to drive. This environment is merely a toy example compared to what Tesla is doing with their cars. But anyways, I am here to learn about RL not about full driving autonomous cars in the real world. One of the first things I noticed about several implementation of DQN on the carracing environment from gymnasium, was the fact that all of them were trained after grayscaling the frames. I asked myself, "why not in rgb?". I guess the answer is quite trivial, the car does not neceserally have to learn colors to understand shapes and edges as long as there is some difference to the road and off-road parts. And secondly grayscale means learning less information, 1 channel instead of 3 color channels, so less computational complexity. But none of the online work (CITE) I found showed rgb results or talked about this difference fundamentally. So this final assignment I am going to figure out why people choose grayscaling over rgb, and if the "trivial" answer is so "trivial" after all!

To be a little bit more specific: "Is color really not that important for this type of game? and whatever the outcome is, why?"

## Methodology
I used Torch (CITE) for the DQN's and Gymnasium for the environment. My goal is to check the difference between RGB and grayscale, so I created to preprocessing functions to handle color of frames. I used a standard DQN network as can here be seen:

On top of that I got inspired by this repo https://github.com/wiitt/DQN-Car-Racing, and also implemented Double-DQN. I know this does not really have to do any thing with measuring the difference between RGB and grayscale. But as I said in the beginning I am here to learn some cool things about RL, so why not throw some exploration in there! Double-DQN it is, so I tried this as well just to see what kind of effects it has on output results and mainly "why?".

To see wheter my implementation is somewhat correct, I matched the results using this repo: https://github.com/wiitt/DQN-Car-Racing. Note that I did not copy any code what so ever, even better, their implementation is completely Gymnasium based as where I used Torch and Gymnasium as a combination.

## Experimental Setup
I ran 4 different networks, with each 3 different seeds for trustable outcomes and reproducibility. This means 12 different configurations. They are as followed:
- DQN/Double DQN * RGB/Grayscale * 3 different seeds.

Over all configurations the network architecture and parameters stayed the same, they are as follows:
(FILL IN)

Some other settings from config.py:
(FILL IN)

I deliberatly choose to run only 1000 episodes (250k steps), since this alone took around 1.5-2hrs. The whole sweep took me around 24 hours to run. But 1000 episodes were also just enough to show decent learning results and explain difference where needed.

Other than that most of the configuration parameters are inspired by online work, to match the results so I could purely focus on the RGB vs. Grayscaling matter.

For each configuration the output returns and Runtime, were logged, saved as csv, and plotted.

All configurations where ran on a Desktop PC, with 32 GB of RAM, a GTX1070 (please sponsor me, I would love to have better GPU), and a AMD Ryzen 5 (DOUBLE CHECK). 

## Results


## Conclusion


### What did I learn?
 
## References
https://github.com/wiitt/DQN-Car-Racing
https://arxiv.org/html/2410.22766v1#Ch1.S1 (showed rgb results but it was not on dqn but on resnet)
https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN (also stating that color does not matter that much for this game but not why?)