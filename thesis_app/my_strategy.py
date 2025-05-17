from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.common import EvaluateIns, Parameters, GetParametersRes
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional
import numpy as np

from thesis_app.create_enc_context import load_context, decrypt_tensors, deserialize_encrypted_tensors
import tenseal as ts

# Load your TenSEAL context (with secret key for decryption)
server_context = load_context("./context_data/ckks.context")


class CustomFedAvg(FedAvg):
    def __init__(self, shapes, *args, **kwargs):
        self.shapes = shapes
        print(f"[Server] CustomFedAvg initialized with shapes: {shapes}")
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Optional[Parameters]:
        """Override this to inspect parameters after first training round."""
        print("[Server] aggregate_fit CALLED")
        
        # Only inspect on the first round
        if server_round == 1:
            self.print_parameters(results, failures)
        
        # Aggregate parameters (example: return the first client's parameters)
        if results:
            aggregated_parameters = results[0][1].parameters  # Use the first client's parameters
            aggregated_metrics = {"accuracy": 0.0}  # Example metric
            return aggregated_parameters, aggregated_metrics

        # Handle the case where no results are available
        print("[Server] No results to aggregate.")
        return None

    def print_parameters(
        self,
        results: List[Tuple[ClientProxy, GetParametersRes]],
        failures: List[BaseException],
    ):
        print("[Server] Inspecting client parameters...")

        for i, (client, fit_res) in enumerate(results):
            params: Parameters = fit_res.parameters
            print(f"Client {i}: {len(params.tensors)} tensors, type={params.tensor_type}")

            if params.tensor_type == "tenseal.ckks":
                try:
                    deserialized = deserialize_encrypted_tensors(server_context, params.tensors)
                    decrypted = decrypt_tensors(deserialized)

                    # Reshape and print shape
                    for j, tensor in enumerate(decrypted):
                        reshaped = np.array(tensor).reshape(self.shapes[j])
                        print(f"  - Tensor {j} shape: {reshaped.shape}")
                except Exception as e:
                    print(f"  - Error decrypting tensors: {e}")
            else:
                print(f"  - Skipping non-encrypted tensors from client")

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy | EvaluateIns]]:
        return super().configure_evaluate(server_round, parameters, client_manager)