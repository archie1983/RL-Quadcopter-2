import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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
        # AE: We are not allowed to change physics_sim.py, but it will only work with a 6-dimensional
        # AE: state, so I must trim it here, but for my Neural Networks, I will use the full state.
        self.sim = PhysicsSim(init_pose[:6], init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * len(init_pose)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_pose = init_pose

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
        reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum() - .3 * (abs(self.sim.pose[3:])).sum()
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
            pose_all.append(np.concatenate((self.sim.pose, self.sim.angular_v)))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = np.concatenate([self.init_pose] * self.action_repeat) 
        return state