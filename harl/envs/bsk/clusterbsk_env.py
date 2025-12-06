from harl.envs.bsk.make_cluster_bsk import make_BSK_Cluster_env, make_BSK_Walker_env, make_BSK_SAR_OPT_env
from munch import Munch
import copy
import numpy as np
from gymnasium.spaces import flatdim
from gymnasium.spaces import Discrete, Box
from harl.envs.bsk.make_single_sat_bsk import make_BSK_SingleSat_env
from harl.envs.bsk.make_cluster_bsk import make_BSK_SAR_OPT_CLOUD_env
from harl.envs.bsk.make_cluster_bsk import make_BSK_FLOCK_env
import warnings
warnings.filterwarnings("ignore")


#For recording purpose
import pandas as pd
from datetime import datetime, timedelta
import csv
import os
from pathlib import Path


class ClusterbskEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        env_args = Munch.fromDict(self.args)
        bsk_scenario = env_args.key.split('-')[0]
        print(bsk_scenario)

        if len(env_args.key.split('-')) > 1:
            task_challenge = env_args.key.split('-')[1]
            randomness_key = None

            if len(env_args.key.split('-')) == 3:
                randomness_key = env_args.key.split('-')[2]
        else:
            randomness_key = None

        if bsk_scenario == "single_sat":
            self.env = make_BSK_SingleSat_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with single satellite scenario")

        elif bsk_scenario == "walker":
            self.env = make_BSK_Walker_env(
                env_args, self.satellite_names, bsk_scenario[1])
            print("Running BSK-ENV with walker-delta scenario")

        elif bsk_scenario == "hmg_cluster":
            self.env = make_BSK_Cluster_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with 3 Optical satellites cluster scenario")

        elif bsk_scenario == "het_cluster":
            self.env = make_BSK_SAR_OPT_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with 1 SAR and 2 OPTICAL satellites cluster scenario")

        elif bsk_scenario == "hmg_flock":
            self.env = make_BSK_FLOCK_env(
                env_args, task_challenge, randomness_key)
            print("Running BSK-ENV with FLOCK OPTICAL satellites cluster scenario")

        elif bsk_scenario == "het_cloud_cluster":
            self.env = make_BSK_SAR_OPT_CLOUD_env(
                env_args, task_challenge, randomness_key)
            print(
                "Running BSK-ENV with 1-Cloud detector, 2-OPTICAL, 1 SAR satellites cluster scenario")

        else:
            print("Scenario name not available")
            NotImplementedError


        self.satellite_names = []
        for i in range(env_args.n_satellites):
            self.satellite_names.append(f"Sat-{i}")

        self.longest_action_space = max(
            self.env.action_space, key=lambda x: x.n)
        self.action_names = []
        self.action_names.append("Charge")
        self.action_names.append("Downlink")
        self.action_names.append("Desaturate")
        self.action_names.append("Drift")
        # for i in range((self.longest_action_space.n)-3):
        for i in range((self.longest_action_space.n)-4):
            self.action_names.append(f"Image_Target_{i}")

        self.n_agents = len(self.satellite_names)
        self.downlinked = {}
        self._info = {}
        self._obs = None

        for sat in self.satellite_names:
            for action_name in self.action_names:
                self._info[f'{sat}-{action_name}'] = 0
            self.downlinked[f'{sat}'] = []
        self.img_cost = []

        self.power_reward = env_args.power_reward
        self.battery_cost_scale = env_args.battery_cost_scale
        self.data_reward = env_args.data_reward
        self.data_cost_scale = env_args.data_cost_scale

        self.past_obs = self._obs

        self.longest_observation_space = max(
            self.env.observation_space, key=lambda x: x.shape
        )
        self.observation_space = [self.longest_observation_space]*self.n_agents
        self.share_observation_space = [self.get_state_size()]
        self.action_space = [self.longest_action_space]*self.n_agents

        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True
        self.avail_actions = self.get_avail_actions()

        # print(self.action_names)
        self.action_frequencies = {
                sat: {action: 0 for action in self.action_names} for i, sat in enumerate(self.satellite_names)}
        # print(self.action_frequencies)
        

        # For recording purpose
        # print(self.args)
        self.capture_records = []
        self.simulation_start_time =  datetime.strptime(self.args['start_datetime'], '%Y-%m-%d %H:%M:%S')
        self.simulation_end_time = None

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, reward, done, trunc, info = self.env.step(actions.flatten())
        print("actions: ",actions)

        self._obs = self._pad_observation(obs)
        dones = done or trunc
        s_obs = self.repeat(self.get_state())
        env_time = self.env.simulator.sim_time
        power_usage_total = 0.0
        data_downlink_total = 0.0
        for sat in self.satellite_names:
            for action_name in self.action_names:
                self._info[f'{sat}-{action_name}'] = 0

        for i, sat in enumerate(self.satellite_names):
            n_acts = len(self.action_names)
            act = actions[i][0]
            
            if act > 3: 
                print("sat: ",i, " action: ",act, " action: ", self.action_names[act])
                satellite = self.env.satellites[i]
                
                point_id = satellite.parse_target_selection(act)


                self.log_capture(
                    satellite_name=sat,
                    point_id=point_id,
                    lat=0,
                    lon=0,
                    priority=1,
                    timestamp_seconds=env_time,
                    target_name=None
                )


            power_usage_gen = obs[i][1].item()-self._past_obs[i][1].item()
            downlinked = 0.0
            if self.power_reward:
                # Add power usage reward
                if power_usage_gen < 0:
                    battery_usage = -1*power_usage_gen*100
                    battery_cost = self.battery_cost_scale * \
                        battery_usage * (1 - self._past_obs[i][1].item())
                    power_usage_total += battery_cost

            if self.data_reward:
                # Add downlinked data reward
                if obs[i][0].item() < self._past_obs[i][0].item():
                    downlinked = (
                        self._past_obs[i][0].item() - obs[i][0].item()) * 100
                    downlinked_cost = downlinked * self.data_cost_scale
                    data_downlink_total += downlinked_cost
                    self.downlinked[f'{sat}'] = downlinked

            # Track battery for satellite `sat`
            self._info[f'{sat}-batt'] = obs[i][1].item()
            self._info[f'{sat}-power_usage_gen'] = power_usage_gen
            # Track memory for satellite `sat`
            self._info[f'{sat}-mem'] = obs[i][0].item()
            self._info[f'{sat}-downlinked'] = self.downlinked[f'{sat}']
            act = [int(a) for a in actions]
            self._info[f'{sat}-{self.action_names[act[i]]}'] = 1

            # print("i: ", i)
            # print("Sat: ", sat)
            # print("Action: ",self.action_names[act[i]])
            self.action_frequencies[sat][self.action_names[act[i]]] += 1

        # self.img_cost.append(float(reward))
        # self._info[f'img_cost'] = np.mean(self.img_cost)
        self._info[f'img_cost'] = reward
        self._info[f'time'] = env_time
        reward += -1*power_usage_total + data_downlink_total
        self._past_obs = self._obs

        return (
            self._obs,
            s_obs,
            self.n_agents*[[reward]],
            self.n_agents*[dones],
            self.n_agents*[self._info],
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        obs, info = self.env.reset(seed=0)
        self._obs = self._pad_observation(obs)
        self._past_obs = self._obs
        s_obs = self.repeat(self.get_state())
        self._info = {}
        self.img_cost = []
        for sat in self.satellite_names:
            for action_name in self.action_names:
                self._info[f'{sat}-{action_name}'] = 0
            self.downlinked[f'{sat}'] = 0

        return self._obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        # print(self.env.action_space)
        # print(agent_id)
        # if(agent_id >= len(self.env.action_space)) :
        #     return None

        valid = flatdim(self.env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self.env.unwrapped, "state_size"):
            return self.env.unwrapped.state_size
        # total_length = 0
        # for obs_space in self.observation_space:
        #     total_length+=flatdim(obs_space)
        total_length = self.longest_observation_space.shape[0]*self.n_agents
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_length,),
            dtype=self.env.observation_space[0].dtype,
        )

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def render(self):
        pass
        # self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in range(self.n_agents):
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]




    def log_capture(self, satellite_name, point_id, lat, lon, priority, timestamp_seconds, target_name):
        # global capture_records, simulation_start_time
        
        capture_datetime = self.seconds_to_datetime(
            timestamp_seconds, self.simulation_start_time)

        real_sat_name = {'Sat-0': 'OPT-FLOCK 4Q-34',
                      'Sat-1': 'OPT-FLOCK 4Q-35',  
                      'Sat-2': 'OPT-FLOCK 4Q-36'  }
    #     satellite_names.append(f"OPT-FLOCK 4Q-34")
    # satellite_names.append(f"OPT-FLOCK 4Q-35")
    # satellite_names.append(f"OPT-FLOCK 4Q-36")

        record = {
            'satellite': real_sat_name[satellite_name],
            'point_id': point_id,
            'lat': round(lat, 4),
            'lon': round(lon, 4),
            'priority': round(priority, 2),
            'capture_time': capture_datetime,
            'target_name': target_name
        }
        self.capture_records.append(record)
        print(
            f"  [CAPTURED] {satellite_name} → Target {point_id} at {capture_datetime.strftime('%H:%M:%S')}")


    # def save_capture_records(self, output_dir):
    def save_capture_records(self,):
        # global capture_records, simulation_start_time, simulation_end_time

        print(os.system("pwd"))

        input_csv_path = Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent / \
            "iodata" / "WP4_input_new_flock.csv"

        # output_dir = Path(os.path.realpath(__file__)).parent / \
        #     "_dat" / "ocean" 
        output_dir = Path(os.path.realpath(__file__)).parent.parent.parent.parent.parent / \
            "iodata" 

        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(
                f"WP4_input_new_flock.csv not found at {input_csv_path}")

        df_in = pd.read_csv(input_csv_path)
        df_in['n_captured'] = 0
        df_in['t_last_capture'] = ''
        df_in['last_captured_by'] = ''


        print(df_in.head())
        print(df_in.dtypes)
        df_in = df_in.convert_dtypes(str)
        # df_in['point_id'].astype(str)
        # list_id = df_in['point_id'].astype(str)
        # print(list_id)
        print(df_in.dtypes)
        # print(df_in.iloc[25, 1])
        # print(df_in.iloc[25, 1] == '2025-01-01|1456|-32.17501|152.14967|25')
        # print(df_in[df_in['point_id'] == '2025-01-01|1456|-32.17501|152.14967|25'].index[0])

        # for rec in capture_records:
        #     print(rec['point_id'])

        n_cap = 0
        for record in self.capture_records:            
            point_id = int(record['point_id'].name)
            # row_idx =  df_in[df_in['point_id'] == point_id].index[0]
            row_idx =  df_in[df_in['point_id'] == point_id].index
            # print("point_id: ", record['point_id'].name, " index: ", row_idx)
            # point_id = point_id.replace('Target(','')
            # point_id = point_id.replace(')','')
            # df_in.loc[point_id, 'n_captured'] = df_in.loc[point_id, 'n_captured'] + 1
            # df_in.loc[point_id, 't_last_capture'] = record['capture_time']
            # df_in.loc[point_id, 'captured_by'] = record['satellite']

            df_in.iloc[row_idx, -3] = df_in.iloc[row_idx, -3] + 1
            df_in.iloc[row_idx, -2] = str(record['capture_time'])
            df_in.iloc[row_idx, -1] = record['satellite']
            n_cap = n_cap + 1

        df_cap = df_in[df_in['n_captured'] > 0]
        print("Total captured targets: ",df_cap.shape[0])
        print("N_capturing actions: ", n_cap)


        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "Op2_output_from_WP3_to_WP4_by_RL.csv")
        # output_path = os.path.join(output_dir, self.args.out_file_name)
        print(output_path)
        df_in.to_csv(output_path, index=False)
   


    def seconds_to_datetime(self, seconds, start_datetime):
        return start_datetime + timedelta(seconds=seconds)