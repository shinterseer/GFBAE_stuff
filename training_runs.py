def get_runs20250617(save_to='./out_dir'):
    desired_batch_size = 32
    training_runs = []

    parameter_dict = {'num_layers': 2, 'neurons_per_layer': 512, 'num_episode_batches': 800,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    return training_runs



def get_runs20250611(save_to='./out_dir'):
    desired_batch_size = 32
    training_runs = []

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 2000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 8, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.0001, .0001, .00001, .00001)}
    suffix = '_lrp'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    return training_runs


def get_runs20250610(save_to='./out_dir'):
    desired_batch_size = 32
    training_runs = []
    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 64, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 64, 'num_episode_batches': 2000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 64, 'num_episode_batches': 2000,
                      'epsilon_at_halfpoint': .05, 'learning_rates': (.001, .001, .0001)}
    suffix = '_eh05_lr3'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 64, 'num_episode_batches': 2000,
                      'epsilon_at_halfpoint': .05, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = '_eh05_lr4'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    return training_runs


def get_runs20250610_overnight(save_to='./out_dir'):
    desired_batch_size = 32
    training_runs = []

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 16, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = '_lr4'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 1000,
                      'epsilon_at_halfpoint': .05, 'learning_rates': (.001, .001, .0001)}
    suffix = '_eh05'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 2000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 4000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 8000,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    return training_runs


def get_runs20250610_dummy(save_to='./out_dir'):
    desired_batch_size = 32
    training_runs = []

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 10,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 16, 'num_episode_batches': 10,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 10,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001, .0001)}
    suffix = '_lr4'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 10,
                      'epsilon_at_halfpoint': .05, 'learning_rates': (.001, .001, .0001)}
    suffix = '_eh05'
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 20,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 40,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    parameter_dict = {'num_layers': 4, 'neurons_per_layer': 32, 'num_episode_batches': 80,
                      'epsilon_at_halfpoint': .1, 'learning_rates': (.001, .001, .0001)}
    suffix = ''
    memory_size = int(parameter_dict['num_episode_batches'] * desired_batch_size * 24 / 10)  # 24 experiences per episode, 10 times more experiences than memory size
    parameter_dict['agent_memory_size'] = memory_size
    file_stem = f'{save_to}/dqn_building_n{parameter_dict["num_layers"]}x{parameter_dict["neurons_per_layer"]}_m{parameter_dict["agent_memory_size"]}{suffix}'
    parameter_dict['file_stem'] = file_stem
    training_runs.append(parameter_dict)

    return training_runs
