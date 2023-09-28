from env import *
import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent
import os


experiment_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

for i in range(5,len(experiment_list)):
    # How long should training run?
    num_iterations = 15000

    # How many initial random steps, before training start, to
    # collect initial data.
    initial_collect_steps = 100

    # How many steps should we run each iteration to collect
    # data from.
    collect_steps_per_iteration = 50

    # How much data should we store for training examples.
    replay_buffer_max_length = 10000

    batch_size = 64

    # learning_rate = 1e-4
    # How often should the program provide an update.
    log_interval = 1

    # How many episodes should the program use for each evaluation.
    num_eval_episodes = 2000

    # How often should an evaluation occur.
    eval_interval = 50

    state = SC_MTL_State()

    initial_input = state.make_initial_input()

    action_dim = 2 ** (len(layers_size) - 1) + 1  # action = 0이면 head만 training 후 전송.
    # action == action dim-1 이면 No training

    action_min = [0 for t in range(action_dim)]
    action_max = [1 for t in range(action_dim)]

    env = SC_MTL_Env(initial_input, state, 'DS2CM', action_dim, action_min, action_max, experiment_list[i])
    global_step = tf.compat.v1.train.get_or_create_global_step()
    eval_env = tf_py_environment.TFPyEnvironment(env)
    train_env = tf_py_environment.TFPyEnvironment(env)

    train_env.reset()

    actor_fc_layers = (len(state.make_min_state()), len(layers_size))
    critic_obs_fc_layers = (len(state.make_min_state()),)
    critic_action_fc_layers = None
    critic_joint_fc_layers = (len(layers_size),)
    ou_stddev = 0.2
    ou_damping = 0.15
    target_update_tau = 0.05
    target_update_period = 5
    dqda_clipping = None
    td_errors_loss_fn = tf.compat.v1.losses.huber_loss
    gamma = 0.995
    reward_scale_factor = 1.0
    gradient_clipping = None

    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    debug_summaries = False
    summarize_grads_and_vars = False

    actor_net = actor_network.ActorNetwork(
        train_env.time_step_spec().observation,
        train_env.action_spec(),
        fc_layer_params=actor_fc_layers,
    )

    critic_net_input_specs = (train_env.time_step_spec().observation,
                              train_env.action_spec())

    critic_net = critic_network.CriticNetwork(
        critic_net_input_specs,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
    )

    tf_agent = ddpg_agent.DdpgAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        ou_stddev=ou_stddev,
        ou_damping=ou_damping,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_data(train_env, random_policy, replay_buffer, steps=100)

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return, avg_ec, avg_tc = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)

    action_for_static = np.zeros(6, )
    edge_only = action_for_static
    edge_only[0] = 1
    md_only = action_for_static
    md_only[-1] = 1

   
    returns = []
    edge_returns = []
    md_returns = []
    head_returns = []
    random_returns = []

    ddpg_avg_ec = []
    ddpg_avg_tc = []
    rand_avg_ec = []
    rand_avg_tc = []
    edge_avg_ec = []
    edge_avg_tc = []
    md_avg_ec = []
    md_avg_tc = []
    head_avg_ec = []
    head_avg_tc = []

    for _ in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, tf_agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience).loss

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0 and step >= 9999:
            avg_return, avg_ec, avg_tc = compute_avg_return(eval_env, tf_agent.policy, \
                                                            num_eval_episodes)
            print('Average Return for RL = {0}'.format(avg_return))
            returns.append(avg_return)
            rand_avg_return, rand_ec, rand_tc = compute_avg_return_for_static(eval_env, 'Random', num_eval_episodes)
            print('Average Return for random = {0}'.format(rand_avg_return))
            edge_only_avg_return, edge_only_ec, edge_only_tc = compute_avg_return_for_static(eval_env, 'Edge_Only',
                                                                                             num_eval_episodes)
            print('Average Return for edge only = {0}'.format(edge_only_avg_return))
            md_only_avg_return, md_only_ec, md_only_tc = compute_avg_return_for_static(eval_env, 'MD_Only',
                                                                                       num_eval_episodes)
            print('Average Return for md only = {0}'.format(md_only_avg_return))
            head_only_avg_return, head_only_ec, head_only_tc = compute_avg_return_for_static(eval_env, 'Head_Only',
                                                                                             num_eval_episodes)
            print('Average Return for head only = {0}'.format(head_only_avg_return))

            edge_returns.append(edge_only_avg_return)
            md_returns.append(md_only_avg_return)
            head_returns.append(head_only_avg_return)
            random_returns.append(rand_avg_return)

            ddpg_avg_ec.append(avg_ec)
            ddpg_avg_tc.append(avg_tc)
            rand_avg_ec.append(rand_ec)
            rand_avg_tc.append(rand_tc)
            edge_avg_ec.append(edge_only_ec)
            edge_avg_tc.append(edge_only_tc)
            md_avg_ec.append(md_only_ec)
            md_avg_tc.append(md_only_tc)
            head_avg_ec.append(head_only_ec)
            head_avg_tc.append(head_only_tc)

    np.save('Cost/result/DDPG_Cost' + str(i) + '.npy', returns)
    np.save('Cost/result/Edge_Cost' + str(i) + '.npy', edge_returns)
    np.save('Cost/result/MD_Cost' + str(i) + '.npy', md_returns)
    np.save('Cost/result/Head_Cost' + str(i) + '.npy', head_returns)
    np.save('Cost/result/RAND_Cost' + str(i) + '.npy', random_returns)

    np.save('Cost/result/DDPG_EC' + str(i) + '.npy', ddpg_avg_ec)
    np.save('Cost/result/DDPG_TC' + str(i) + '.npy', ddpg_avg_tc)
    np.save('Cost/result/EDGE_EC' + str(i) + '.npy', edge_avg_ec)
    np.save('Cost/result/EDGE_TC' + str(i) + '.npy', edge_avg_tc)
    np.save('Cost/result/Head_EC' + str(i) + '.npy', head_avg_ec)
    np.save('Cost/result/Head_TC' + str(i) + '.npy', head_avg_tc)
    np.save('Cost/result/MD_EC' + str(i) + '.npy', md_avg_ec)
    np.save('Cost/result/MD_TC' + str(i) + '.npy', md_avg_tc)
    np.save('Cost/result/RAND_EC' + str(i) + '.npy', rand_avg_ec)
    np.save('Cost/result/RAND_TC' + str(i) + '.npy', rand_avg_tc)

