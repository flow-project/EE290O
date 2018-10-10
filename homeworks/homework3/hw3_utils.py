"""Utility objects and methods for homework 3."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.scenarios import LoopScenario, CircleGenerator
from flow.controllers import RLController, IDMController, ContinuousRouter
import matplotlib.pyplot as plt
import numpy as np

HORIZON = 750


def get_params(render=False):
    """Create flow-specific parameters for stabilizing the ring experiments.

    Parameters
    ----------
    render : bool, optional
        specifies whether the visualizer is active

    Returns
    -------
    flow.core.params.SumoParams
        sumo-specific parameters
    flow.core.params.EnvParams
        environment-speciifc parameters
    flow.scenarios.Scenario
        a flow-compatible scenario object
    """
    sumo_params = SumoParams(
        sim_step=0.4,
        sumo_binary="sumo-gui" if render else "sumo",
        seed=0
    )

    vehicles = Vehicles()
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        # speed_mode="aggressive",
        num_vehicles=1
    )
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {"noise": 0}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21
    )

    env_params = EnvParams(
        horizon=HORIZON,
        warmup_steps=int(750/4)
    )

    net_params = NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
    )

    initial_config = InitialConfig(
        spacing="uniform",
        bunching=50,
    )

    scenario = LoopScenario(
        name="stabilizing_the_ring",
        generator_class=CircleGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    return sumo_params, env_params, scenario


def plot_results(results, labels):
    """Plot the training curves of multiple algorithms.

    Parameters
    ----------
    results : list of np.ndarray
        results from each algorithms
    labels : list of str
        name of each algorithm
    """
    colors = plt.cm.get_cmap('tab10', len(labels)+1)
    fig = plt.figure(figsize=(16, 9))
    for i, (label, result) in enumerate(zip(labels, results)):
        plt.plot(np.arange(result.shape[1]), np.mean(result, 0),
                 color=colors(i), linewidth=2, label=label)
        plt.fill_between(np.arange(len(result[0])),
                         np.mean(result, 0) - np.std(result, 0),
                         np.mean(result, 0) + np.std(result, 0),
                         alpha=0.25, color=colors(i))
    plt.title("Training Performance of Different Algorithms", fontsize=25)
    plt.ylabel('Cumulative Return', fontsize=20)
    plt.xlabel('Training Iteration', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=20)

    return fig
