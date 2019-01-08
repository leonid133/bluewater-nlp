import os
import tensorflow as tf
import json
import numpy as np
from collections import namedtuple

StepValue = namedtuple('StepValue', ['step', 'value'])


class EarlyStoppingHandler:
    def __init__(self, model_dir, target_metric, no_increase_checks=3, no_increase_threshold=0.005):
        if no_increase_checks < 1:
            raise ValueError()
        self.model_dir = model_dir
        self.eval_dir = os.path.join(model_dir, 'eval')
        tf.gfile.MakeDirs(self.eval_dir)
        self.target_metric = target_metric
        self.checks_to_keep = no_increase_checks + 1
        self.no_increase_threshold = no_increase_threshold
        # state
        self._step_value_list = []
        #
        self._read_existing_eval_dicts()

    def _read_existing_eval_dicts(self):
        eval_glob = os.path.join(self.eval_dir, 'eval-dict-*.json')
        eval_json_path_list = tf.gfile.Glob(eval_glob)
        for eval_json_fp in eval_json_path_list:
            with tf.gfile.GFile(eval_json_fp) as inp:
                eval_dict = json.load(inp)
            value = eval_dict[self.target_metric]
            step = eval_dict[tf.GraphKeys.GLOBAL_STEP]
            self._step_value_list.append(StepValue(step, value))
        self._step_value_list = sorted(self._step_value_list, key=lambda sv: sv.step)
        if len(self._step_value_list) > self.checks_to_keep:
            self._step_value_list = self._step_value_list[-self.checks_to_keep:]
        tf.logging.info('Read %s previous %s values: %s',
                        len(self._step_value_list),
                        self.target_metric,
                        self._step_value_list)

    def add_eval_dict(self, eval_dict):
        eval_dict = {k: np.asscalar(v) for k, v in eval_dict.items()}
        value = eval_dict[self.target_metric]
        step = eval_dict[tf.GraphKeys.GLOBAL_STEP]
        # write
        with tf.gfile.GFile(os.path.join(self.eval_dir, 'eval-dict-%s.json' % step), mode='w') as out:
            json.dump(eval_dict, out, indent=2)
        #
        if self._step_value_list and self._step_value_list[-1].step >= step:
            raise ValueError('Attempt to add an eval dict with the same or less step')
        self._step_value_list.append(StepValue(step, value))
        while (len(self._step_value_list) > self.checks_to_keep):
            self._step_value_list.pop(0)
        tf.logging.info('Early stopping: last values for %s: %s',
                        self.target_metric,
                        self._step_value_list)

    def should_continue(self):
        if len(self._step_value_list) < self.checks_to_keep:
            return True
        for i in range(1, len(self._step_value_list)):
            cur_sv = self._step_value_list[i]
            prev_sv = self._step_value_list[i - 1]
            if (cur_sv.value - self.no_increase_threshold) > prev_sv.value:
                return True
        return False

    def get_best_step(self):
        return max(self._step_value_list, key=lambda t: t.value, default=None)
