import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np

from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.policies import nature_cnn, register_policy


class CustomRegularizedDQNFeedForwardPolicy(DQNPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu,
                 l1_regularizer=0., l2_regularizer=0.,
                 **kwargs):
        super(CustomRegularizedDQNFeedForwardPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps,
            n_batch, dueling=dueling, reuse=reuse,
            scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features

                    fc_params = {}
                    if l1_regularizer > 0 or l2_regularizer > 0:
                        fc_params['weights_regularizer'] = tf.contrib.layers.l1_l2_regularizer(
                            scale_l1=l1_regularizer,
                            scale_l2=l2_regularizer,
                        )
                    for layer_size in layers:
                        action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None, **fc_params)
                        if layer_norm:
                            action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                        action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

class CustomRegularizedDQNMlpPolicy(CustomRegularizedDQNFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True,
                 layer_norm=False, l1_regularizer=0., l2_regularizer=0.,
                 **_kwargs):
        super(CustomRegularizedDQNMlpPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
            feature_extraction="mlp", obs_phs=obs_phs, dueling=dueling,
            layer_norm=layer_norm, l1_regularizer=l1_regularizer,
            l2_regularizer=l2_regularizer, **_kwargs)

register_policy("CustomRegularizedDQNMlpPolicy", CustomRegularizedDQNMlpPolicy)
