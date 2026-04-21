# LunarLander Environment Guide

## State Space (8 values)
0. x position
1. y position
2. x velocity
3. y velocity
4. angle
5. angular velocity
6. left leg contact
7. right leg contact

## Action Space (4 actions)
0. do nothing
1. fire left orientation engine
2. fire main engine
3. fire right orientation engine

## Reward
The agent gets rewarded for:
- moving closer to the landing pad
- landing safely
- staying upright
- reducing speed
- making leg contact

The agent gets penalized for:
- crashing
- tilting badly
- wasting fuel
- moving inefficiently

## Episode Ends When
- the lander crashes
- the lander lands successfully
- the episode is truncated/terminated

## Goal
Land safely between the flags with low speed and stable balance.

## Observations
- Y position decreases as the lander falls
- Y velocity becomes more negative, means its falling faster
- X velocity fluctuates so the lander has random sideways motion
- The agent doesn't stabilize its motion, thus unstable sideways movement due to random actions
- This shows that the agent needs learning to achieve controlled descent and stable landing

- angle keeps increasing over time, which means the lander is tilting and losing balance
- angular velocity is increasing so the lander is spinning faster / sudden spike in angular velocity means lander started spinning uncontrollably 
- angular velocity suddenly dropping means the lander either crashed or the episode terminated 
- one run showed that the lander did try to stabilize itself (angle & angular velocity changed directions)

## Demo observations
- The total reward for each episode is negative so the lander is currently doing very badly
- This is ok because currently its taking random values and is not trained