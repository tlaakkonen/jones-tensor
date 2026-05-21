import jones_tensor
import pickle
import numpy as np
import tqdm
import os

# This test verifies that the computed value of the Jones polynomial matches with the version in the Knot Atlas, 
# for all 2261 non-trivial knots and links that have both a PD presentation and Jones polynomial available.

katlas_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "katlas_jones_pd.pkl")
katlas_data = pickle.load(open(katlas_data_path, "rb"))

ts = np.linspace(0, 2 * np.pi, 100)
qs = np.exp(1j * ts)

def eval_jones_from_terms(q, t):
    total = 0
    for c, p in t:
        total += c * q ** float(p)
    return total

for knot, (pd, jones) in tqdm.tqdm(katlas_data.items()):
    link = jones_tensor.MorseLink.from_pd(pd)
    calculated = jones_tensor.evaluate_jones(qs, link, optimize='greedy')
    
    actual = eval_jones_from_terms(qs, jones)
    if not np.allclose(calculated, actual):
        raise RuntimeError(f"Failed to reproduce Knot Atlas value for {knot}")
    
print(f"Success, verified all {len(katlas_data)} links.")
