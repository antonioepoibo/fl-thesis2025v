import tenseal as ts
import os

# -------------------------------
# BFV Context Handling
# -------------------------------
def create_bfv_context():
    poly_mod_degree = 8192
    plain_modulus = 1032193  # Should be a large prime number

    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_mod_degree,
        plain_modulus=plain_modulus
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def save_context(context, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(context.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=True,
            save_relin_keys=True
        ))

def load_context(filepath):
    with open(filepath, "rb") as f:
        return ts.context_from(f.read())

# -------------------------------
# Encryption / Decryption (BFV)
# -------------------------------
def encrypt_bfv_vectors(context, vectors):
    return [ts.bfv_vector(context, vec) for vec in vectors]

def decrypt_bfv_vectors(encrypted_vectors):
    return [vec.decrypt() for vec in encrypted_vectors]

# -------------------------------
# Serialization / Deserialization
# -------------------------------
def serialize_encrypted_vectors(encrypted_vectors):
    return [vec.serialize() for vec in encrypted_vectors]

def deserialize_bfv_vectors(context, serialized_list):
    return [ts.bfv_vector_from(context, data) for data in serialized_list]

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Example integer vectors
    int_vectors = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    # Create and save BFV context
    bfv_context = create_bfv_context()
    save_context(bfv_context, "./context_data/bfv.context")

    # Load context
    loaded_bfv_context = load_context("./context_data/bfv.context")

    # Encrypt vectors
    encrypted = encrypt_bfv_vectors(loaded_bfv_context, int_vectors)

    # Serialize
    serialized = serialize_encrypted_vectors(encrypted)

    # Deserialize
    deserialized = deserialize_bfv_vectors(loaded_bfv_context, serialized)

    # Decrypt and print
    decrypted = decrypt_bfv_vectors(deserialized)
    for i, vec in enumerate(decrypted):
        print(f"Decrypted BFV vector {i + 1}: {vec}")
