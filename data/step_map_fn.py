import tensorflow as tf

TARGET_WIDTH = 160
TARGET_HEIGHT = 128


def jaco_step_map_fn(step):
    # Resize to be compatible with robo_net trajectory
    transformed_step = {}
    # Observations
    transformed_step['observation'] = {}

    transformed_step['observation']['image'] = tf.cast(tf.image.resize_with_pad(
        step['observation']['image'], target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT), tf.uint8)
    transformed_step['observation']['image'] = tf.transpose(transformed_step['observation']['image'], [2, 0, 1])

    transformed_step['observation']['natural_language_embedding'] = step['observation']['natural_language_embedding']

    # Actions
    transformed_step['action'] = {}
    transformed_step['action']['first_three'] = tf.cast(step['action']['world_vector'], tf.float32)
    transformed_step['action']['middle_three'] = tf.cast(step['action']['terminate_episode'], tf.float32)
    transformed_step['action']['final_one'] = tf.cast(step['action']['gripper_closedness_action'], tf.float32)

    transformed_step['is_first'] = step['is_first']
    transformed_step['is_last'] = step['is_last']
    transformed_step['is_terminal'] = step['is_terminal']

    return transformed_step


def berkeley_cable_routing_step_map_fn(step):
    # Resize to be compatible with robo_net trajectory
    transformed_step = {}
    # Observations
    transformed_step['observation'] = {}
    transformed_step['observation']['image'] = tf.cast(tf.image.resize_with_pad(
        step['observation']['image'], target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT), tf.uint8)
    transformed_step['observation']['image'] = tf.transpose(transformed_step['observation']['image'], [2, 0, 1])
    transformed_step['observation']['natural_language_embedding'] = step['observation']['natural_language_embedding']
    # Actions
    transformed_step['action'] = {}
    transformed_step['action']['first_three'] = tf.cast(step['action']['world_vector'], tf.float32)
    transformed_step['action']['middle_three'] = tf.cast(step['action']['rotation_delta'], tf.float32)
    transformed_step['action']['final_one'] = tf.reshape(step['action']['terminate_episode'], [1])

    transformed_step['is_first'] = step['is_first']
    transformed_step['is_last'] = step['is_last']
    transformed_step['is_terminal'] = step['is_terminal']
    return transformed_step


def bridge_step_map_fn(step):
    # Resize to be compatible with robo_net trajectory
    transformed_step = {}
    # Observations
    transformed_step['observation'] = {}

    transformed_step['observation']['image'] = tf.cast(tf.image.resize_with_pad(
        step['observation']['image'], target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT), tf.uint8)
    transformed_step['observation']['image'] = tf.transpose(transformed_step['observation']['image'], [2, 0, 1])

    transformed_step['observation']['natural_language_embedding'] = step['observation']['natural_language_embedding']

    # Actions
    transformed_step['action'] = {}
    transformed_step['action']['first_three'] = tf.cast(step['action']['world_vector'], tf.float32)
    transformed_step['action']['middle_three'] = tf.cast(step['action']['rotation_delta'], tf.float32)
    transformed_step['action']['final_one'] = tf.cast(step['action']['open_gripper'], tf.float32)

    transformed_step['is_first'] = step['is_first']
    transformed_step['is_last'] = step['is_last']
    transformed_step['is_terminal'] = step['is_terminal']

    return transformed_step


def toto_step_map_fn(step):
    # Resize to be compatible with robo_net trajectory
    transformed_step = {}
    # Observations
    transformed_step['observation'] = {}

    transformed_step['observation']['image'] = tf.cast(tf.image.resize_with_pad(
        step['observation']['image'], target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT), tf.uint8)
    transformed_step['observation']['image'] = tf.transpose(transformed_step['observation']['image'], [2, 0, 1])

    transformed_step['observation']['natural_language_embedding'] = step['observation']['natural_language_embedding']

    # Actions
    transformed_step['action'] = {}
    transformed_step['action']['first_three'] = tf.cast(step['action']['world_vector'], tf.float32)
    transformed_step['action']['middle_three'] = tf.cast(step['action']['rotation_delta'], tf.float32)
    transformed_step['action']['final_one'] = tf.cast(step['action']['open_gripper'], tf.float32)

    transformed_step['is_first'] = step['is_first']
    transformed_step['is_last'] = step['is_last']
    transformed_step['is_terminal'] = step['is_terminal']

    return transformed_step
