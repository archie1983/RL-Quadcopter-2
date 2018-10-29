import sys
import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose, init_velocities, 
        init_angle_velocities, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        # AE: Initial state will contain pose (position and angles of the copter)
        # AE: and the copter dimensional velocities (speed)
        self.initial_state = np.concatenate((init_pose, init_velocities))
        # AE: We are not allowed to change physics_sim.py, but it will only work with a 6-dimensional
        # AE: state, so I must trim it here, but for my Neural Networks, I will use the full state.
        self.sim = PhysicsSim(self.initial_state[:6], init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * len(self.initial_state)
        self.action_low = 400
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # AE: Three main ideas for the reward:
        # AE: 1) Reward for being closer to the target
        # AE: 
        # AE: 2) I could also reward the agent for having higher
        # AE: speed at the beginning and reducing it as it gets closer. That should help it to not overshoot or undershoot.
        # AE: To achieve that, I will need to take into account the remaining distance in each dimension and each value of the
        # AE: speed vector. Perhaps I want the speed to be the same as the remaining distance. I.e. if the copter is 10m
        # AE: away from the target, then it should move towards the target at 10m/s, then after 1/10s = 100ms, it will be 9m away
        # AE: from the target and it should reduce its speed to 9m/s, then 1/9 s later it will be 8m away and so on, which 
        # AE: will increase the time it takes to reach the target (instead of 1s, it would now take: 
        # AE: 1/10 + 1/9 + 1/8 + 1/7 + 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1 = 2.93s), but it would not overshoot anymore.
        # AE:
        # AE: 3) Reward the quadcopter for being level (the angular positions being as close to 0 as possible) as that will
        # AE: ensure smooth riding with fewer falls.
        distances = (self.target_pos - self.sim.pose[:3])
        velocities = self.sim.v

        # AE: The Z axis can range within [0, 300]. So if the Z component of the distance to target is 300, 
        # AE: the reward should be 0 for Z dimension, and if the Z component of the distance to target 
        # AE: is 0, then the reward should be 1 for Z dimension.
        # AE: The X and Y coordinates can range within [-300, 300], so the distance to target
        # AE: can range from 0 to 600 for dimensions X and Y. So if X component of the distance
        # AE: to target is 600, the reward should be 0 for X and if the X component of the distance
        # AE: to target is 0, then the reward should be 1 for X. Same for Y component.
        # AE: Rewards for each of the components of the distance left to travel:
        rd_x = 1 - (1 / 600.0) * abs(distances[0])
        rd_y = 1 - (1 / 600.0) * abs(distances[1])
        rd_z = 1 - (1 / 300.0) * abs(distances[2])
        
        # AE: If velocity in all three dimensions is the same as distance left, then reward with 1.
        # AE: Give 1/3 of reward for each dimension that is moving at the
        # AE: same speed as the distance left. Don't penalise for going too fast. Instead give 
        # AE: smaller reward for higher difference between desired speed and actual speed.
        # AE:
        # AE: self.sim.time # 0..5 in 1/50 steps
        # AE: self.sim.runtime # 5
        # AE: self.sim.dt # 1/50
        # AE: 
        # AE: The total number of time steps that we have to achieve the target is: tt = (self.sim.runtime / self.sim.dt)
        # AE: The number of time steps left at any given moment is: tl = (tt - self.sim.time / self.sim.dt)
        # AE: The desired speed at any given moment is: ds = distances / tl
        # AE: Current speed is: velocities
        # AE: A term: prop = min(ds[0], velocities[0]) / max(ds[0], velocities[0]) 
        # AE: will give me the proportion of how much the speed is wrong with 1 being OK and 0 being static, but that
        # AE: will not work when ds[0] and velocities[0] are both negative or have different signs. So to generalise:
        # AE: prop = min(abs(ds[0]), abs(velocities[0])) / max(abs(ds[0]), abs(velocities[0]))
        # AE: prop_sign = 1 if (ds[0] * velocities[0]) >= 0 else -1
        # AE: Velocity reward for X axis: rv_x = prop_sign * prop
        # AE: This velocity reward should be multiplied by (1/3), because that is only one of three dimensions.
        tt = (self.sim.runtime / self.sim.dt) # AE: total simulation time
        tl = (tt - self.sim.time / self.sim.dt) # AE: total time minus flight time so far
        # AE: desired speed. Avoiding potential division by 0 with the check.
        #ds = [0.0001, 0.0001, 0.0001] if tl <= 0 else distances / tl
        ds = [0, 0, 0] if tl <= 0 else distances / tl

        # AE: the actual proportion value.
        prop_sign = 1 if (ds[0] * velocities[0]) >= 0 else -1 # AE: sign of the proportion
        prop = min(abs(ds[0]), abs(velocities[0])) / (max(abs(ds[0]), abs(velocities[0])) + 0.00001) # AE: Avoiding potential division by 0
        rv_x = (1 / 3.0) * prop_sign * prop # AE: reward based on the proportion of how much the speed is out.

        # AE: if prop is NaN (so desired speed (ds) or current speed (velocities) is infinite), then no reward
        if (np.isnan(prop)): rv_x = 0
        # AE: if prop is Inf (so desired speed (ds) and current speed (velociies) are 0), then full reward
        if (np.isinf(prop)): rv_x = 1 / 3.0

        prop_sign = 1 if (ds[1] * velocities[1]) >= 0 else -1
        prop = min(abs(ds[1]), abs(velocities[1])) / (max(abs(ds[1]), abs(velocities[1])) + 0.00001)
        rv_y = (1 / 3.0) * prop_sign * prop

        if (np.isnan(prop)): rv_y = 0
        if (np.isinf(prop)): rv_y = 1 / 3.0

        prop_sign = 1 if (ds[2] * velocities[2]) >= 0 else -1
        prop = min(abs(ds[2]), abs(velocities[2])) / (max(abs(ds[2]), abs(velocities[2])) + 0.00001)
        rv_z = (1 / 3.0) * prop_sign * prop

        if (np.isnan(prop)): rv_z = 0
        if (np.isinf(prop)): rv_z = 1 / 3.0

        # AE: I will also add in a reward for the copter not being jerky (angular positions being level)
        # AE: Maximum angular position in each dimension can be: (2 * np.pi) and that's how I can scale it.
        # AE: Perhaps I don't really care about Z axis, because that will probably not cause it to fall.
        ang_pos = self.sim.pose[3:5]
        ang_props = ang_pos / (2 * np.pi)
        ang_rew = 1 - (1 / 3.0) * ang_props.sum()

        reward = rd_x + rd_y + rd_z + rv_x + rv_y + rv_z + ang_rew
        
        if (np.isnan(reward) or np.isinf(reward)): 
            print("Bad reward- possibly NaN returned by physics_sim.py", velocities, distances, self.sim.runtime, self.sim.time, self.sim.dt)
            #[ nan  nan  nan] [ nan  nan  nan] 5.0 0.04 0.02
            reward = -np.inf
            sys.exit("Bad reward- possibly NaN returned by physics_sim.py")
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #pose_all.append(self.sim.pose)
            # AE: accomodating the new, expanded state, that also includes angular speeds
            #pose_all.append(np.concatenate((self.sim.pose, self.sim.angular_v)))
            # AE: and copter speeds
            #pose_all.append(np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v)))
            pose_all.append(np.concatenate((self.sim.pose, self.sim.v)))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        #state = np.concatenate([self.init_pose] * self.action_repeat)
        state = np.concatenate([self.initial_state] * self.action_repeat)
        return state