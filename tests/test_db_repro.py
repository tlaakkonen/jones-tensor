import jones_tensor
import pickle
import csv
import gzip
import numpy as np
import tqdm
import os

# This test verifies that the computed value of the Jones polynomial matches with the versions in the
# Knot Atlas, KnotInfo, and LinkInfo databases for all non-trivial knots and links that have PD presentations,
# braid representatives, and Jones polynomials available. This amounts to about ~2000 from the Knot Atlas, 
# ~14000 from KnotInfo, and ~4000 from LinkInfo. It takes about two minutes to verify.

# NB: there is one exception, which is that the PD presentation of L2a1{1} is not correctly converted into 
# a MorseLink. The reason is that MorseLink.from_pd does not always find the correct orientation of the 
# components in the case where a component has exactly two crossings. The braid representative works fine, however.

# Knot Atlas:
katlas_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "katlas_jones_pd.pkl.gz")
katlas_data = pickle.load(gzip.open(katlas_data_path, "rb"))

ts = np.linspace(0, 2 * np.pi, 100)
qs = np.exp(1j * ts)

def eval_jones_from_terms(q, t):
    total = 0
    for c, p in t:
        total += c * q ** float(p)
    return total

print("Verifying Knot Atlas data:")
for knot, (pd, jones) in tqdm.tqdm(katlas_data.items()):
    link = jones_tensor.MorseLink.from_pd(pd)
    calculated = jones_tensor.evaluate_jones(qs, link, optimize='greedy')
    
    actual = eval_jones_from_terms(qs, jones)
    if not np.allclose(calculated, actual):
        raise RuntimeError(f"Failed to reproduce Knot Atlas value for {knot}")
print(f"Success, verified all {len(katlas_data)} links.")


# KnotInfo:
knotinfo_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "knotinfo_jones_braid_pd.csv.gz")
knotinfo_data = list(csv.DictReader(gzip.open(knotinfo_data_path, "rt")))

print("Verifying KnotInfo data:")
for row in tqdm.tqdm(knotinfo_data):
    # Hacky but I don't want to write an actual parser
    actual = eval(row['Jones'].replace("^", "**"), { "t": qs })

    pd = [[int(s) for s in c.removeprefix("[").removesuffix("]").split(";")] for c in row['PD Notation'].replace(" ","")[1:-1].split("];[")]
    link = jones_tensor.MorseLink.from_pd(pd)
    calculated = jones_tensor.evaluate_jones(qs, link, optimize='greedy')
    if not np.allclose(calculated, actual):
        raise RuntimeError(f"Failed to reproduce KnotInfo value for {row['Name']}")
    
    if ']]' in row['Braid Notation']:
        bn = row['Braid Notation'].replace(" ","").split('];[')[0][2:]
    else:
        bn = row['Braid Notation'].replace(" ","")[1:-1]
    word = [int(g) for g in bn.split(";")]
    link = jones_tensor.MorseLink.from_braid(word)
    calculated = jones_tensor.evaluate_jones(qs, link, optimize='greedy')
    if not np.allclose(calculated, actual):
        raise RuntimeError(f"Failed to reproduce KnotInfo value for {row['Name']}")
print(f"Success, verified all {len(knotinfo_data)} links.")


# LinkInfo:
linkinfo_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "linkinfo_jones_braid_pd.csv.gz")
linkinfo_data = list(csv.DictReader(gzip.open(linkinfo_data_path, "rt")))
pd_exceptions = ['L2a1{1}']

print("Verifying LinkInfo data:")
for row in tqdm.tqdm(linkinfo_data):
    # Hacky but I don't want to write an actual parser
    actual = eval(row['Jones Polynomial'].replace("^", "**"), { "x": np.sqrt(qs) })

    strands, word = row['Braid Notatation'].replace(" ","")[1:-1].split(";", maxsplit=1)
    strands = int(strands)
    word = [int(g) for g in word[1:-1].split(";")]
    link = jones_tensor.MorseLink.from_braid(word, strands)
    calculated = jones_tensor.evaluate_jones(qs, link, optimize='greedy')
    if not np.allclose(calculated, actual):
        raise RuntimeError(f"Failed to reproduce LinkInfo value for {row['Name']}")

    if row['Name'] in pd_exceptions: continue
    pd = [[int(s) for s in c.removeprefix("{").removesuffix("}").split(";")] for c in row['PD Notation (vector)'].replace(" ","")[1:-1].split("};{")]
    link = jones_tensor.MorseLink.from_pd(pd)
    calculated = jones_tensor.evaluate_jones(qs, link, optimize='greedy')
    if not np.allclose(calculated, actual):
        raise RuntimeError(f"Failed to reproduce LinkInfo value for {row['Name']}")
print(f"Success, verified all {len(linkinfo_data)} links.")
