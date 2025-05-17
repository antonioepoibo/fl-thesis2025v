"""thesis-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from thesis_app.my_strategy import CustomFedAvg
from thesis_app.FedAvgEnc import FedAvgEnc
from thesis_app.task import Net, get_weights


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    weights = get_weights(Net())
    shapes = [w.shape for w in weights]
    parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAvgEnc(
        shapes=shapes,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
