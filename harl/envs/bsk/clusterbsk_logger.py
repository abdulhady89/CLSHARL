import numpy as np
from harl.common.base_logger import BaseLogger


class ClusterbskLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(ClusterbskLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
    def get_task_name(self):
        return self.env_args["key"]

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        eval_std_rew = np.std(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        print("Evaluation std episode reward is {}.\n".format(eval_std_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_std_rew])) + "\n"
        )
        self.log_file.flush()
    
    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)
                if k=='eval_average_episode_rewards':
                    self.writter.add_scalars('eval_std_episode_rewards', {k: np.std(v)}, self.total_num_steps)

    def log_render(self, env_infos, t):
        """Log environment information."""
        for k, v in env_infos.items():
            self.writter.add_scalars(k, {k: v}, t)
    