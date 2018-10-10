import gym
import numpy as np

TEST_ENV = gym.make("Pendulum-v0")


def test_1_linear(alg_class):
    """Test that the linear policy implementation is working correctly.

    alg_class : DirectPolicyAlgorithm
        the implementation of the algorithm to be tests

    Raises
    ------
    AssertionError
        If the actual return from the policy does not match the expected return
        (we ensure that the two should be the same by setting a random seed).
    AssertionError
        If there is a shape mismatch between the output mean value and the
        expected output mean.
    """
    # create the policy with the linear model being used
    alg = alg_class(TEST_ENV, linear=True)

    # collect a sample output for a given input state
    input_1 = [[0, 1, 2]]
    policy_mean_1 = alg.sess.run(alg.policy[0],
                                 feed_dict={alg.s_t_ph: input_1})
    expected_1 = np.array([[1.753811]])

    # collect a second output from a second input state
    input_2 = [[0, 0, 0], [0, 1, 2]]
    policy_mean_2 = alg.sess.run(alg.policy[0],
                                 feed_dict={alg.s_t_ph: input_2})
    expected_2 = np.array([[0.11094809], [1.753811]])

    if policy_mean_1.shape != expected_1.shape:
        raise AssertionError("Linear policy output shape is not correct for "
                             "input state: {}.\nExpected: {}\n Received: {}".
                             format(input_1, expected_1.shape,
                                    policy_mean_1.shape))

    if np.linalg.norm(policy_mean_1 - expected_1) > 1e-3:
        raise AssertionError("Policy return for the input state {} is not "
                             "correct.\n Expected: {}\n Received: {}".
                             format(input_1, expected_1, policy_mean_1))

    if policy_mean_2.shape != expected_2.shape:
        raise AssertionError("Linear policy output shape is not correct for "
                             "input state: {}.\nExpected: {}\n Received: {}".
                             format(input_2, expected_2.shape,
                                    policy_mean_2.shape))

    if np.linalg.norm(policy_mean_2 - expected_2) > 1e-3:
        raise AssertionError("Policy return for the input state {} is not "
                             "correct.\n Expected: {}\n Received: {}".
                             format(input_2, expected_2, policy_mean_2))

    print("Success!")


def test_1_stochastic(alg_class):
    """Test that the logstd of a stochastic policy  is working correctly.

    alg_class : DirectPolicyAlgorithm
        the implementation of the algorithm to be tests

    Raises
    ------
    AssertionError
        If the actual return from the policy logstd does not match the expected
        return (we ensure that the two should be the same by setting a random
        seed)
    AssertionError
        If there is a shape mismatch between the output logstd value and the
        expected output logstd.
    """
    # create the policy with the linear model being used
    alg = alg_class(TEST_ENV, stochastic=True)

    # collect a sample output for a given input state
    input_1 = [[0, 1, 2]]
    policy_logstd_1 = alg.sess.run(alg.policy[1],
                                   feed_dict={alg.s_t_ph: input_1})
    expected_1 = np.array([0.9644977])

    # collect a second output from a second input state
    input_2 = [[0, 0, 0], [0, 1, 2]]
    policy_logstd_2 = alg.sess.run(alg.policy[1],
                                   feed_dict={alg.s_t_ph: input_2})
    expected_2 = np.array([0.9644977])

    if policy_logstd_1.shape != expected_1.shape:
        raise AssertionError("Linear policy output shape is not correct for "
                             "input state: {}.\nExpected: {}\n Received: {}".
                             format(input_1, expected_1.shape,
                                    policy_logstd_1.shape))

    if np.linalg.norm(policy_logstd_1 - expected_1) > 1e-3:
        raise AssertionError("Policy return for the input state {} is not "
                             "correct.\n Expected: {}\n Received: {}".
                             format(input_1, expected_1, policy_logstd_1))

    if policy_logstd_2.shape != expected_2.shape:
        raise AssertionError("Linear policy output shape is not correct for "
                             "input state: {}.\nExpected: {}\n Received: {}".
                             format(input_2, expected_2.shape,
                                    policy_logstd_2.shape))

    if np.linalg.norm(policy_logstd_2 - expected_2) > 1e-3:
        raise AssertionError("Policy return for the input state {} is not "
                             "correct.\n Expected: {}\n Received: {}".
                             format(input_2, expected_2, policy_logstd_2))

    print("Success!")


def test_1_stochastic_sampling(alg_class):
    """Test that action sampling from a stochastic policy is working correctly.

    alg_class : DirectPolicyAlgorithm
        the implementation of the algorithm to be tests

    Raises
    ------
    AssertionError
        If the actual return from the policy logstd does not match the expected
        return (we ensure that the two should be the same by setting a random
        seed)
    """
    # create the policy with the linear model being used
    alg = alg_class(TEST_ENV, stochastic=True)

    # collect a sample output for a given input state
    input_1 = [[0, 1, 2]]
    policy_sample_1 = alg.compute_action(input_1)
    expected_1 = np.array([[-0.32567555]])

    # collect a second output from a second input state
    input_2 = [[0, 0, 0], [0, 1, 2]]
    policy_sample_2 = alg.compute_action(input_2)
    expected_2 = np.array([[-0.08618341], [-0.16663393]])

    if policy_sample_1.shape != expected_1.shape:
        raise AssertionError("Linear policy output shape is not correct for "
                             "input state: {}.\nExpected: {}\n Received: {}".
                             format(input_1, expected_1.shape,
                                    policy_sample_1.shape))

    if np.linalg.norm(policy_sample_1 - expected_1) > 1e-3:
        raise AssertionError("Policy return for the input state {} is not "
                             "correct.\n Expected: {}\n Received: {}".
                             format(input_1, expected_1, policy_sample_1))

    if policy_sample_2.shape != expected_2.shape:
        raise AssertionError("Linear policy output shape is not correct for "
                             "input state: {}.\nExpected: {}\n Received: {}".
                             format(input_2, expected_2.shape,
                                    policy_sample_2.shape))

    if np.linalg.norm(policy_sample_2 - expected_2) > 1e-3:
        raise AssertionError("Policy return for the input state {} is not "
                             "correct.\n Expected: {}\n Received: {}".
                             format(input_2, expected_2, policy_sample_2))

    print("Success!")


def test_2_log_likelihood(alg_class):
    """Test that action sampling from a stochastic policy is working correctly.

    alg_class : DirectPolicyAlgorithm
        the implementation of the algorithm to be tests

    Raises
    ------
    AssertionError
        If the actual return from the policy logstd does not match the expected
        return (we ensure that the two should be the same by setting a random
        seed)
    """
    # create the policy with the linear model being used
    alg = alg_class(TEST_ENV, stochastic=True)

    log_likelihoods = alg.log_likelihoods()

    # collect a sample output for a given input state
    input_s = [[0, 0, 0], [0, 1, 2], [1, 2, 3]]
    input_a = [[0], [1], [2]]
    computed = alg.sess.run(log_likelihoods,
                            feed_dict={alg.a_t_ph: input_a,
                                       alg.s_t_ph: input_s})
    expected_1 = np.array([-1.8834362, -1.9682424, -2.146737])

    if computed.shape != expected_1.shape:
        raise AssertionError("Shape of log likelihood for input state {} and "
                             "input action {} is not correct.\nExpected: {}\n "
                             "Received: {}".format(input_s, input_a,
                                                   expected_1.shape,
                                                   computed.shape))

    if np.linalg.norm(computed - expected_1) > 1e-3:
        raise AssertionError("log likelihood for input state {} and input "
                             "action {} is not correct.\nExpected: {}\n "
                             "Received: {}".format(input_s, input_a,
                                                   expected_1, computed))

    print("Success! Wow, you're really smart!")


def test_2_expected_return(alg_class):
    """Test that action sampling from a stochastic policy is working correctly.

    alg_class : DirectPolicyAlgorithm
        the implementation of the algorithm to be tests

    Raises
    ------
    AssertionError
        If the actual return from the policy logstd does not match the expected
        return (we ensure that the two should be the same by setting a random
        seed)
    """
    # create the policy with the linear model being used
    alg = alg_class(TEST_ENV, stochastic=True)
    alg.gamma = 1.0

    # 1. Test the non-normalized case
    input_1 = [{"reward": [1, 1, 1, 1]},
               {"reward": [1, 1, 1, 1]}]
    vs_1 = alg.compute_expected_return(samples=input_1)

    # convert ot numpy array if needed
    if isinstance(vs_1, list):
        vs_1 = np.array(vs_1)

    # collect a second output from a second input state
    expected_1 = np.array([4, 3, 2, 1, 4, 3, 2, 1])

    if expected_1.shape != vs_1.shape:
        raise AssertionError("Mismatch between shape of expected and actual "
                             "shape of returns for input {}.\nExpected: {}"
                             "\n Received: {}".
                             format(input_1, expected_1.shape, vs_1.shape))

    if np.linalg.norm(vs_1 - expected_1) > 1e-3:
        raise AssertionError("Expected non-normalized return for the input "
                             "state {} is not correct.\n Expected: {}\n "
                             "Received: {}".format(input_1, expected_1, vs_1))

    print("Success! And can I say, you look great today.")
