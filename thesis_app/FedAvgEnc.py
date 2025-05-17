from flwr.server.strategy import FedAvg
from flwr.common import Parameters, GetParametersRes, EvaluateIns
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Optional
import numpy as np
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from logging import WARNING


from thesis_app.create_enc_context import load_context, decrypt_tensors, serialize_encrypted_tensors, deserialize_encrypted_tensors
import tenseal as ts

# Load your TenSEAL context (with secret key for decryption)
server_context = load_context("./context_data/ckks.context")


class FedAvgEnc(FedAvg):
    def __init__(self, shapes, *args, **kwargs):
        self.shapes = shapes
        print(f"[Server] FedAvgEnc initialized")
        super().__init__(*args, **kwargs)

    def print_parameters(
        self,
        results: List[Tuple[ClientProxy, GetParametersRes]],
        failures: List[BaseException],
        summed: Optional[List] = None  # Optional list of CKKSVector
    ):
        print("[Server] Inspecting client parameters...")

        # Show first 10 values of tensor 0 from each client
        for i, (client, fit_res) in enumerate(results):
            params: Parameters = fit_res.parameters
            print(f"Client {i}: {len(params.tensors)} tensors, type={params.tensor_type}")

            if params.tensor_type == "tenseal.ckks":
                try:
                    deserialized = deserialize_encrypted_tensors(server_context, params.tensors)
                    decrypted = decrypt_tensors(deserialized)

                    # Only print first tensor's first 10 values
                    if decrypted:
                        reshaped = np.array(decrypted[0]).reshape(self.shapes[0])
                        flat = reshaped.flatten()
                        print(f"  - Tensor 0 first 10 values: {flat[:10]}")
                except Exception as e:
                    print(f"  - Error decrypting tensors from client {i}: {e}")
            else:
                print(f"  - Skipping non-encrypted tensors from client")

        # Show first 10 values of first summed tensor
        if summed is not None:
            print("[Server] Decrypted summed tensors:")
            try:
                decrypted_sum = decrypt_tensors(summed)
                if decrypted_sum:
                    reshaped = np.array(decrypted_sum[0]).reshape(self.shapes[0])
                    flat = reshaped.flatten()
                    print(f"  - Summed Tensor 0 first 10 values: {flat[:10]}")
            except Exception as e:
                print(f"[Server] Error decrypting summed tensors: {e}")




    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Optional[Parameters]: 
        print(f"\n[Server] aggregate_fit round {server_round}")

        if not results:
            return None

        # Step 1: Deserialize encrypted parameters from each client
        all_encrypted = []
        for _, fit_res in results:
            if fit_res.parameters.tensor_type != "tenseal.ckks":
                continue
            encrypted = deserialize_encrypted_tensors(server_context, fit_res.parameters.tensors)
            all_encrypted.append(encrypted)

        if not all_encrypted:
            print("[Server] No usable encrypted tensors.")
            return None

        # Step 2: Homomorphic sum of encrypted tensors
        num_tensors = len(all_encrypted[0])
        summed = []

        for i in range(num_tensors):
            acc = all_encrypted[0][i]
            for tensors in all_encrypted[1:]:
                acc += tensors[i]  # Homomorphic addition
            summed.append(acc)

        # Step 3: Serialize summed encrypted tensors
        serialized = serialize_encrypted_tensors(summed)

        # Step 4: Package into FL Parameters
        parameters_aggregated = Parameters(
            tensors=serialized,
            tensor_type="tenseal.ckks"
        )

        print("[Server] Encrypted parameters summed and returned.")
        self.print_parameters(results, failures, summed)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy | EvaluateIns]]:
        return []