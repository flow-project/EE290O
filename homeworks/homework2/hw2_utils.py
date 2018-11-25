""" Utility objects and methods for homework 2. """

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.controllers import SumoCarFollowingController, GridRouter
from flow.scenarios import SimpleGridScenario
from copy import deepcopy

# time horizon of a single rollout
HORIZON = 400
# inflow rate of vehicles north-south
NORTH_SOUTH_EDGE_INFLOW = 300
# inflow rate of vehicles east-west
EAST_WEST_EDGE_INFLOW = 100
# enter speed for departing vehicles
V_ENTER = 30
# number of row of bidirectional lanes
N_ROWS = 1
# number of columns of bidirectional lanes
N_COLUMNS = 1
# length of inner edges in the grid network
INNER_LENGTH = 300
# length of final edge in route
LONG_LENGTH = 300
# length of edges that vehicles start on
SHORT_LENGTH = 300
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SumoCarFollowingController, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT+N_RIGHT)*N_COLUMNS + (N_BOTTOM+N_TOP)*N_ROWS,
    speed_mode="right_of_way"
)

# inflows of vehicles are place on all outer edges (listed here)
north_south_edges = []
north_south_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
north_south_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
east_west_edges = []
east_west_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
east_west_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in north_south_edges:
    inflow.add(veh_type="human", edge=edge,
               vehs_per_hour=NORTH_SOUTH_EDGE_INFLOW,
               departLane="free", departSpeed="max")
for edge in east_west_edges:
    inflow.add(veh_type="human", edge=edge,
               vehs_per_hour=EAST_WEST_EDGE_INFLOW,
               departLane="free", departSpeed="max")

# define the traffic light logic
tl_logic = TrafficLights(baseline=False)
phases = [{"duration": "50", # how long the light is green north-south
           "state": "GGGrrrGGGrrr"},
          {"duration": "6",
           "state": "yyyrrryyyrrr"},
          {"duration": "50", # how long the light is green east-west
           "state": "rrrGGGrrrGGG"},
          {"duration": "6",
           "state": "rrryyyrrryyy"}]

for i in range(N_ROWS*N_COLUMNS):
    tl_logic.add("center"+str(i), tls_type="static", phases=phases,
                 programID=1)

net_params = NetParams(
    in_flows=inflow,
    no_internal_links=False,
    additional_params={
        "speed_limit": V_ENTER + 5,
        "grid_array": {
            "short_length": SHORT_LENGTH,
            "inner_length": INNER_LENGTH,
            "long_length": LONG_LENGTH,
            "row_num": N_ROWS,
            "col_num": N_COLUMNS,
            "cars_left": N_LEFT,
            "cars_right": N_RIGHT,
            "cars_top": N_TOP,
            "cars_bot": N_BOTTOM,
        },
        "horizontal_lanes": 1,
        "vertical_lanes": 1,
    },
)

sumo_params_test = SumoParams(
    restart_instance=False,
    sim_step=1,
    sumo_binary="sumo-gui",
)

sumo_params_train = SumoParams(
    # set restart_instance to true, for faster execution of rollouts
    restart_instance=True,
    sim_step=1,
    # turn off the gui (again to be faster)
    sumo_binary="sumo",
)

env_params = EnvParams(
    evaluate=True,
    horizon=HORIZON,
    additional_params={
        "discrete": True,  # to have discrete actions
        "switch_time": 2.0,
        "num_observed": 2,
        "tl_type": "actuated",
    },
)

initial_config = InitialConfig(
    shuffle=True
)


def create_scenario():
    return SimpleGridScenario(
        name="grid",
        vehicles=deepcopy(vehicles),
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic
    )
