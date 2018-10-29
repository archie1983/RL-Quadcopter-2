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
        # AE: Initial state will contain pose (position and angles of the copter), angular velocities (copter stability)
        #self.initial_state = np.concatenate((init_pose, init_angle_velocities))
        # AE: and the copter dimensional velocities (speed)
        #self.initial_state = np.concatenate((init_pose, init_velocities, init_angle_velocities))
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
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # AE: I need to take into account how level is the quadcopter (angles- the second triplet in pose),
        # AE: the stability of the copter (angular velocities of all three dims)
        # AE: and perhaps the pose itself should include angular velocities in addition to the angular positions.
        # AE: and velocities too.
        # AE: However physics_sim.py samples the Euler angles from pose in the following manner: pose[3:] and if we added 
        # AE: anything to pose, then suddenly angles would have more than 3 dimensions. And of course we're not allowed to 
        # AE: change anything in physcis_sim.py, so we will still need to pass the old 6-dimensional pose to the physcis_sim.py
        # AE: functions, but use more dimensions here in the task.py and that of course will give more features and weights to
        # AE: the neural nets of critic and actor.
        #reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum() - .3 * (abs(self.sim.pose[3:6])).sum()
        
        # AE: Because the state now contains copter velocity and its angular velocities, I should improve the reward
        # AE: function to take that into account. The agent is already being penalised for going in the wrong direction
        # AE: for each dimension (first term in the reward function). The agent is also being penalised for making the copter
        # AE: unstable (angles not level). I could also reward the agent for having higher
        # AE: speed at the beginning and reducing it as it gets closer. That should help it to not overshoot or undershoot.
        # AE: To achieve that, I will need to take into account the remaining distance in each dimension and each value of the
        # AE: speed vector. Perhaps I want the speed to be the same as the remaining distance. I.e. if the copter is 10m
        # AE: away from the target, then it should move towards the target at 10m/s, then after 1/10s = 100ms, it will be 9m away
        # AE: from the target and it should reduce its speed to 9m/s, then 1/9 s later it will be 8m away and so on, which 
        # AE: will increase the time it takes to reach the target (instead of 1s, it would now take: 
        # AE: 1/10 + 1/9 + 1/8 + 1/7 + 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1 = 2.93s), but it would not overshoot anymore.
        distances = (self.target_pos - self.sim.pose[:3])
        velocities = self.sim.v
        
        # AE: distance from the target
        #distances_mod = abs(distances).sum()
        #t1 = .1 * np.log(abs(self.sim.pose[:3] - self.target_pos).sum() + )
        #t1 = 0.1 * 0 if distances_mod < 1 else np.log(distances_mod)
        #start_distance = self.target_pos - self.initial_state[:3]
        #t1 = -1 * np.log(np.power(distances / start_distance, 2)).sum()
        
        # AE: in-stability of the copter
        #t2 = -1 * (np.tanh(self.sim.pose[3:6])).sum()
        
        # AE: speed in each dimension must reduce when getting nearer the target
        #t3 = .1 * (abs(distances - velocities)).sum() 
        #t3 = -1 * np.log(np.power(distances - velocities, 2)).sum()
        
        # AE: Reward for moving in the Z axis and penalise for movement in X and Y axis
        #t4 = 1 * ((abs(velocities[0]) + abs(velocities[1])) - abs(velocities[2]))
        
        # AE: Jerkiness of the copter on X and Y axis. It can rotate on Z axis ok.
        #t5 = 1 * (abs(self.sim.angular_v)).sum() 
        #t6 = 1 - np.clip(abs(self.sim.pose[2] - self.target_pos[2]), 0, 1)
        #t6 = 1 if abs(self.sim.pose[2] - self.target_pos[2]) < 5 else 0
        #t6 = abs(self.sim.pose[2] - self.target_pos[2]) < 5 else 0
        
        # AE: Bonus for coming closer to the Z target Maybe X and Y too?
        #t6 = abs(self.target_pos[2] - self.initial_state[2]) - abs(self.sim.pose[2] - self.target_pos[2])
        #t6 = 0.1 * (self.target_pos - self.initial_state[:3] - abs(self.sim.pose[:3] - self.target_pos).sum())
        #print(t1, t2, t3, t4)
        #reward = np.clip((1. - t3 - t2 + t6), -1.0, 10.0)
        #reward = np.clip((1. - t1 - t2 + t6), -10.0, 10.0)
        #reward = np.clip((1. - t3 - t2 - t5 + t6), -1.0, 10.0)
        #reward = np.clip((t3 + t1 + t2), -1.0, 1.0)
        
        #reward_z = np.tanh(1 - 0.003*(abs(self.sim.pose[2] - self.target_pos[2]))).sum()
        #reward_xy = np.tanh(1 - 0.009*(abs(self.sim.pose[:2] - self.target_pos[:2]))).sum()
        #reward = reward_z + reward_xy
        
        # AE: Inspiration gotten from: https://github.com/xadahiya/RL-Quadcopter-2
        # AE: Our Z axis can range from 0 - 300. So if I am going to use tanh in the reward,
        # AE: then I want to use the tanh function in its linear range with x=[-1 to 1],
        # AE: but nah then I don't just use a linear function?
        # AE: So if the Z component of the distance to target is 300, the reward should be 0,
        # AE: and if the Z component of the distance to target is 0, then the reward should be 1.
        # AE: The X and Y coordinates can range within [-300, 300], so the distance to target
        # AE: can range from 0 to 600 for each component- X and Y. So if X component of the distance
        # AE: to target is 600, the reward should be 0 and if the X component of the distance
        # AE: to target is 0, then the reward should be 1. Same for Y component.
        #reward_z = np.tanh(1 - 0.0033 * abs(self.sim.pose[2] - self.target_pos[2]) )
        # AE: Rewards for each of the components of the distance left to travel:
        rd_x = 1 - (1 / 600.0) * abs(distances[0])
        rd_y = 1 - (1 / 600.0) * abs(distances[1])
        rd_z = 1 - (1 / 300.0) * abs(distances[2])
        
        # AE: If velocity in all three dimensions is the same as distance left, then reward with 1.
        # AE: If the velocity is opposite in all three dimensions, then don't penalise, because distance
        # AE: reward will take care of that. Give 1/3 if reward for each dimension that is moving at the
        # AE: same speed as the distance left. Don't penalise for going too fast. Give smaller reward for
        # AE: higher difference between desired speed and actual speed.
        # AE:
        # AE: self.sim.time # 0..5 in 1/50 steps
        # AE: self.sim.runtime # 5
        # AE: self.sim.dt # 1/50
        # AE: 
        # AE: The total number of time steps that we have to achieve the target is: tt = (self.sim.runtime / self.sim.dt)
        # AE: The number of time steps left at any given moment is: tl = (tt - self.sim.time / self.sim.dt)
        # AE: The desired speed at any given moment is: ds = distances / tl
        # AE: Current speed is: velocities
        # AE: A term: sp = min(ds[0], velocities[0]) / max(ds[0], velocities[0]) 
        # AE: will give me the proportion of how much the speed is wrong with 1 being OK and 0 being static, but that
        # AE: will not work when ds[0] and velocities[0] are both negative or have different signs. To generalise:
        # AE: prop = min(abs(ds[0]), abs(velocities[0])) / max(abs(ds[0]), abs(velocities[0]))
        # AE: prop_sign = 1 if (ds[0] * velocities[0]) >= 0 else -1
        # AE: Velocity reward for X axis: rv_x = prop_sign * prop
        # AE: This velocity reward should be multiplied by (1/3), because that is only one of three dimensions.
        tt = (self.sim.runtime / self.sim.dt)
        tl = (tt - self.sim.time / self.sim.dt)
        ds = distances / tl
        
        prop_sign = 1 if (ds[0] * velocities[0]) >= 0 else -1
        prop = min(abs(ds[0]), abs(velocities[0])) / (max(abs(ds[0]), abs(velocities[0])) + 0.001)
        rv_x = (1 / 3.0) * prop_sign * prop
        
        prop_sign = 1 if (ds[1] * velocities[1]) >= 0 else -1
        prop = min(abs(ds[1]), abs(velocities[1])) / (max(abs(ds[1]), abs(velocities[1])) + 0.001)
        rv_y = (1 / 3.0) * prop_sign * prop
        
        prop_sign = 1 if (ds[2] * velocities[2]) >= 0 else -1
        prop = min(abs(ds[2]), abs(velocities[2])) / (max(abs(ds[2]), abs(velocities[2])) + 0.001)
        rv_z = (1 / 3.0) * prop_sign * prop

        # AE: I will also add in a reward for the copter not being jerky (angular positions being level)
        # AE: Maximum angular position in each dimension can be: (2 * np.pi) and that's how I can scale it.
        # AE: Perhaps I don't really care about Z axis, because that will probably not cause it to fall.
        ang_pos = self.sim.pose[3:5]
        ang_props = ang_pos / (2 * np.pi)
        ang_rew = 1 - (1 / 3.0) * ang_props.sum()

        reward = rd_x + rd_y + rd_z + rv_x + rv_y + rv_z + ang_rew
        
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