# Super Mario AI
The original Super Mario Bros has always been a classic. One of the best part about 
older games is that they run on lower resolution, which makes it easier
to feed into a neural network.

## What this program does
This project uses a topic called reinforcement learning. The way it works is by
having a reward based system and the program learns on its own to get the best reward.
You can read a few articles on the internet to get a gist of it.
This program can play the original Mario Bros for the NES.

## Requirements
* Gym - `pip install gym`
* Gym Super Mario Bros - `pip install gym-super-mario-bros`
* NES-Py - `pip install nes-py`
<br />
<br />
As of right now, nes-py is only supported on linux so please run it on linux.
<br />
I have tried for hours to try to get it on Windows, to no avail. If you know how to, please let me know
so I can update this.

## Running

`python3 main.py <level you want to run>`

Eg. If you want to run 1-1

`python3 main.py '1-1'`

## Images

![alt text](https://github.com/vivek3141/super-mario-ai/blob/master/Images/img1.png "World 1-1")

Implementation of NEAT is still in progress.