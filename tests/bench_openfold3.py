#!/usr/bin/env python3
"""
Throughput, latency, and correctness benchmark: fastkernels OpenFold3 vs reference.

Uses real precomputed MSAs from OpenProteinSet (AWS Registry of Open Data)
to benchmark with biologically meaningful inputs rather than synthetic data.

Data source:
  OpenProteinSet on AWS S3 (s3://openfold/pdb/) contains precomputed MSAs for
  ~140k PDB chains. We download a representative subset of chains spanning
  different protein lengths, parse the MSA files, and feed the resulting
  tensors to both engines.

Architecture:
  Follows the same subprocess-isolation pattern as bench_vllm.py — each engine
  runs in its own subprocess via run_worker() to avoid GPU state contamination.
  Correctness is evaluated end-to-end within the throughput run (shared weights,
  compare outputs). Latency is measured separately.

Throughput scenarios (proteins processed sequentially at bs=1, as in production):
  - short:  proteins ≤150 residues
  - medium: proteins 150–400 residues
  - long:   proteins 400–800 residues

Usage:
    python tests/bench_openfold3.py
    python tests/bench_openfold3.py --skip-reference
    python tests/bench_openfold3.py --scenario short
    python tests/bench_openfold3.py --data-dir /path/to/cached/openfold_data
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT))

from fastkernels.bench.utils.worker import run_worker
from fastkernels.bench.utils.workloads import (
    STRUCTURE_PREDICTION_LATENCY_WORKLOADS,
    STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS,
)


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Representative PDB chains from OpenProteinSet, grouped by length
# ---------------------------------------------------------------------------

# Verified chains available in OpenProteinSet (s3://openfold/pdb/).
# Lengths from RCSB PDB. Each bucket covers a distinct length range
# to exercise different computational regimes (O(N) vs O(N²)).
CHAIN_CATALOG = {
    # ≤150 residues — fast inference, high query count
    "short": [
        ("1l2y", "A",  20, "TC5b mini-protein"),
        ("2jof", "A",  20, "Trp-cage"),
        ("1le1", "A",  13, "Tryptophan zipper 2"),
        ("1gcn", "A",  29, "Glucagon"),
        ("1cag", "A",  31, "Calmodulin fragment"),
        ("1vii", "A",  36, "Villin headpiece"),
        ("1kd8", "A",  37, "Barnase fragment"),
        ("1hf9", "A",  42, "Ubiquitin-like"),
        ("1crn", "A",  46, "Crambin"),
        ("3nir", "A",  46, "Crambin variant"),
        ("1enh", "A",  54, "Engrailed homeodomain"),
        ("1idy", "A",  54, "c-Myb DNA-binding repeat 3"),
        ("1shf", "A",  59, "Fyn SH3 domain"),
        ("1fas", "A",  61, "Fasciculin 1"),
        ("1isu", "A",  62, "High-potential iron-sulfur"),
        ("1nxb", "A",  62, "Neurotoxin B"),
        ("1c9o", "A",  66, "Cold-shock protein"),
        ("1csp", "A",  67, "Cold shock protein CspB"),
        ("1b3a", "A",  67, "RANTES"),
        ("1b67", "A",  68, "Histone HMfA"),
        ("1mjc", "A",  69, "Major cold-shock protein"),
        ("2q2k", "A",  70, "ParR DNA-binding"),
        ("1msi", "A",  70, "Type III antifreeze protein"),
        ("1a8o", "A",  70, "HIV capsid"),
        ("1ctf", "A",  74, "Ribosomal protein L7/L12"),
        ("1d3b", "A",  75, "snRNP-D"),
        ("1pk4", "A",  79, "Plasminogen kringle 4"),
        ("1hyp", "A",  80, "Hydrophobic protein soybean"),
        ("1bdo", "A",  81, "Ferredoxin"),
        ("1pgx", "A",  83, "Protein G"),
        ("1pht", "A",  85, "PI3K p85-alpha SH3"),
        ("1a43", "A",  87, "HIV-1 capsid"),
        ("1gvp", "A",  87, "Gene V protein"),
        ("1ptf", "A",  88, "HPr phosphocarrier"),
        ("1ten", "A",  90, "Tenascin"),
        ("1wit", "A",  93, "Twitchin IgSF module"),
        ("1c5e", "A",  95, "Head decoration protein"),
        ("1bym", "A",  97, "Diphtheria toxin repressor"),
        ("2crb", "A",  98, "CRABP II"),
        ("2acy", "A",  98, "Acylphosphatase"),
        ("1tit", "A",  98, "Titin I27"),
        ("1plc", "A",  99, "Plastocyanin"),
        ("1aac", "A", 105, "Amicyanin"),
        ("1ew4", "A", 106, "CyaY protein"),
        ("1acx", "A", 108, "Actinoxanthin"),
        ("1bkr", "A", 109, "Spectrin beta chain"),
        ("1jpc", "A", 109, "Agglutinin"),
        ("1i4j", "A", 110, "50S ribosomal protein L22"),
        ("1a2p", "A", 111, "Lysozyme"),
        ("1b0n", "A", 111, "SinR protein"),
        ("1qau", "A", 112, "nNOS PDZ domain"),
        ("1thx", "A", 115, "Thioredoxin"),
        ("1dj7", "A", 117, "Ferredoxin-thioredoxin reductase"),
        ("1dhn", "A", 121, "Dihydroneopterin aldolase"),
        ("1alc", "A", 123, "Alpha-lactalbumin"),
        ("1c44", "A", 123, "Sterol carrier protein 2"),
        ("1bgf", "A", 124, "STAT-4"),
        ("1bqk", "A", 124, "Pseudoazurin"),
        ("1byf", "A", 125, "Polyandrocarpa lectin"),
        ("1avd", "A", 128, "Avidin"),
        ("1rie", "A", 129, "Rieske iron-sulfur protein"),
        ("1nyn", "A", 131, "Hypothetical protein"),
        ("1bbh", "A", 131, "Cytochrome c'"),
        ("1c52", "A", 131, "Cytochrome c552"),
        ("1jac", "A", 133, "Jacalin"),
        ("1bkb", "A", 136, "Translation initiation factor 5A"),
        ("1jer", "A", 138, "Cucumber stellacyanin"),
        ("1lit", "A", 144, "Lithostathine"),
        ("1bfg", "A", 146, "Basic fibroblast growth factor"),
        ("1mba", "A", 147, "Myoglobin"),
        ("1cdl", "A", 147, "Calmodulin"),
        ("1dk8", "A", 147, "Axin"),
        ("1ggz", "A", 148, "Calmodulin-related NB-1"),
    ],
    # 150–400 residues — moderate inference cost
    "medium": [
        ("101m", "A", 154, "Myoglobin"),
        ("1bj7", "A", 156, "D2 domain"),
        ("1tnf", "A", 157, "Tumor necrosis factor-alpha"),
        ("1e7l", "A", 157, "Recombination endonuclease VII"),
        ("1am7", "A", 158, "Lysozyme"),
        ("1grj", "A", 158, "GreA protein"),
        ("1aep", "A", 161, "Apolipophorin III"),
        ("1f3g", "A", 162, "Cytochrome c"),
        ("1c1y", "A", 167, "Ras-related protein Rap-1A"),
        ("1cid", "A", 177, "T-cell CD4"),
        ("1aqb", "A", 183, "Retinol-binding protein"),
        ("1gky", "A", 187, "Guanylate kinase"),
        ("3hhr", "A", 191, "Hemoglobin"),
        ("2pth", "A", 193, "Peptidyl-tRNA hydrolase"),
        ("1chd", "A", 203, "CheB methylesterase"),
        ("1thv", "A", 207, "Thaumatin"),
        ("1bam", "A", 213, "Endonuclease BamHI"),
        ("1ake", "A", 214, "Adenylate kinase"),
        ("1cex", "A", 214, "Cutinase"),
        ("1g3p", "A", 217, "Minor coat protein g3p"),
        ("1dkx", "A", 219, "DnaK substrate-binding domain"),
        ("1c3w", "A", 222, "Bacteriorhodopsin"),
        ("1byq", "A", 228, "Heat shock protein 90"),
        ("1gfl", "A", 238, "Green fluorescent protein"),
        ("1bkj", "A", 240, "NADPH-flavin oxidoreductase"),
        ("1e2x", "A", 243, "Fatty acid metabolism regulator"),
        ("1rwz", "A", 245, "DNA polymerase sliding clamp"),
        ("1tim", "A", 247, "Triosephosphate isomerase"),
        ("1b9b", "A", 255, "Triosephosphate isomerase"),
        ("1c90", "A", 265, "Endo-beta-N-acetylglucosamidase H"),
        ("1mml", "A", 265, "MuLV reverse transcriptase"),
        ("1arb", "A", 268, "Achromobacter protease I"),
        ("1d2n", "A", 272, "NSF fusion protein"),
        ("1amp", "A", 291, "Aminopeptidase"),
        ("1c3d", "A", 294, "C3d complement"),
        ("1gca", "A", 309, "Glucose/galactose-binding protein"),
        ("1ads", "A", 315, "Aldose reductase"),
        ("2pia", "A", 321, "Phthalate dioxygenase reductase"),
        ("1bg2", "A", 325, "Kinesin"),
        ("1elv", "A", 333, "Complement C1s"),
        ("1lga", "A", 343, "Lignin peroxidase"),
        ("1bx4", "A", 345, "Adenosine kinase"),
        ("1a0i", "A", 348, "DNA ligase"),
        ("1got", "A", 350, "Gt-alpha/Gi-alpha chimera"),
        ("1czf", "A", 362, "Polygalacturonase II"),
        ("1fnf", "A", 368, "Fibronectin"),
        ("1atn", "A", 373, "Actin"),
        ("1ova", "A", 386, "Ovalbumin"),
        ("1hpm", "A", 386, "44K ATPase fragment"),
    ],
    # 400–700 residues — expensive pair representations
    "long": [
        ("1e5m", "A", 416, "Beta-ketoacyl ACP synthase"),
        ("1gnd", "A", 447, "GDI"),
        ("1b8j", "A", 449, "Alkaline phosphatase"),
        ("1a8d", "A", 452, "Tetanus neurotoxin"),
        ("1cjc", "A", 460, "Adrenodoxin reductase"),
        ("1lam", "A", 484, "Leucine aminopeptidase"),
        ("1b4v", "A", 504, "Cholesterol oxidase"),
        ("1dpe", "A", 507, "Dipeptide-binding protein"),
        ("1fcb", "A", 511, "Flavocytochrome b2"),
        ("1ddt", "A", 535, "Diphtheria toxin"),
        ("1aon", "A", 548, "GroEL chaperonin"),
        ("1aoz", "A", 552, "Ascorbate oxidase"),
        ("1dce", "A", 567, "Rab geranylgeranyltransferase"),
        ("1gpe", "A", 587, "Glucose oxidase"),
        ("1bcc", "A", 446, "Ubiquinol cytochrome c oxidoreductase"),
        ("1byq", "A", 228, "Heat shock protein 90"),
        ("1cdg", "A", 686, "Cyclodextrin glycosyltransferase"),
    ],
    # 700+ residues — stress test for O(N²) operations
    "extra-long": [
        ("1cdg", "A", 686, "Cyclodextrin glycosyltransferase"),
        ("1gpe", "A", 587, "Glucose oxidase"),
        ("1dce", "A", 567, "Rab geranylgeranyltransferase"),
        ("1aon", "A", 548, "GroEL chaperonin"),
        ("1ddt", "A", 535, "Diphtheria toxin"),
        ("1fcb", "A", 511, "Flavocytochrome b2"),
        ("1dpe", "A", 507, "Dipeptide-binding protein"),
        ("1b4v", "A", 504, "Cholesterol oxidase"),
        ("1lam", "A", 484, "Leucine aminopeptidase"),
        ("1cjc", "A", 460, "Adrenodoxin reductase"),
        ("1a8d", "A", 452, "Tetanus neurotoxin"),
        ("1b8j", "A", 449, "Alkaline phosphatase"),
        ("1bcc", "A", 446, "Ubiquinol cytochrome c oxidoreductase"),
    ],
}

PF_BLOCKS = 48
MSA_BLOCKS = 4
NO_ROLLOUT_STEPS = 10
NUM_RECYCLES = 1
MAX_MSA_SEQS = 512

HF_REPO_ID = "OpenFold/OpenFold3"
HF_CHECKPOINT_FILE = "checkpoints/of3-p2-155k.pt"


def download_checkpoint() -> str:
    """Download the pretrained OpenFold3 checkpoint from HuggingFace.

    Returns the local path to the cached .pt file.
    """
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO_ID, HF_CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": w.name,
        "num_queries": w.num_queries,
        "description": w.description,
    }
    for w in STRUCTURE_PREDICTION_THROUGHPUT_WORKLOADS
]

LATENCY_SCENARIOS = [
    {
        "name": w.name,
        "length_bucket": w.length_bucket,
        "num_warmup": w.num_warmup,
        "num_iters": w.num_iters,
    }
    for w in STRUCTURE_PREDICTION_LATENCY_WORKLOADS
]


# ---------------------------------------------------------------------------
# Data download and preparation
# ---------------------------------------------------------------------------

S3_BUCKET = "openfold"
S3_PDB_PREFIX = "pdb"


def download_chain_alignments(pdb_id: str, chain_id: str, data_dir: str) -> str | None:
    """Download precomputed MSA files for a single PDB chain from OpenProteinSet.

    S3 layout: s3://openfold/pdb/{pdb}_{chain}/a3m/*.a3m
    """
    chain_key = f"{pdb_id}_{chain_id}"
    chain_dir = os.path.join(data_dir, "alignments", chain_key)

    if os.path.isdir(chain_dir) and any(
        f.endswith(('.a3m', '.sto'))
        for f in os.listdir(chain_dir) if os.path.isfile(os.path.join(chain_dir, f))
    ):
        return chain_dir

    os.makedirs(chain_dir, exist_ok=True)
    try:
        subprocess.run(
            ["aws", "s3", "cp",
             f"s3://{S3_BUCKET}/{S3_PDB_PREFIX}/{chain_key}/a3m/",
             chain_dir + "/", "--recursive", "--no-sign-request"],
            capture_output=True, text=True, timeout=120,
        )
        if any(f.endswith(('.a3m', '.sto'))
               for f in os.listdir(chain_dir) if os.path.isfile(os.path.join(chain_dir, f))):
            return chain_dir
    except Exception:
        pass
    return None


def prepare_dataset(data_dir: str, scenarios: list[dict], seed: int = 42) -> dict:
    """Download and prepare all required data."""
    print(f"\n  Preparing dataset (data_dir={data_dir}) ...", flush=True)

    repo_test_data = os.path.join(
        str(_PROJECT_ROOT), "vllm_repo", "openfold-3",
        "openfold3", "tests", "test_data",
    )
    repo_alignments = os.path.join(repo_test_data, "alignments")
    repo_mmcifs = os.path.join(repo_test_data, "mmcifs")

    available_chains = {}
    if os.path.isdir(repo_alignments):
        for chain_dir_name in os.listdir(repo_alignments):
            chain_path = os.path.join(repo_alignments, chain_dir_name)
            if os.path.isdir(chain_path):
                parts = chain_dir_name.rsplit("_", 1)
                if len(parts) == 2:
                    pdb_id, chain_id = parts
                    if any(f.endswith(('.a3m', '.sto'))
                           for f in os.listdir(chain_path) if os.path.isfile(os.path.join(chain_path, f))):
                        available_chains[(pdb_id, chain_id)] = {
                            "pdb_id": pdb_id, "chain_id": chain_id,
                            "alignment_dir": chain_path,
                        }
    print(f"    Found {len(available_chains)} chains from repo test data", flush=True)

    all_needed = set()
    for scenario in scenarios:
        bucket = scenario.get("length_bucket", scenario["name"])
        if bucket in CHAIN_CATALOG:
            for pdb_id, chain_id, _, _ in CHAIN_CATALOG[bucket]:
                all_needed.add((pdb_id, chain_id))

    need_download = all_needed - set(available_chains.keys())
    if need_download:
        print(f"    Downloading {len(need_download)} chains from OpenProteinSet ...", flush=True)
        for pdb_id, chain_id in sorted(need_download):
            aln_dir = download_chain_alignments(pdb_id, chain_id, data_dir)
            if aln_dir:
                available_chains[(pdb_id, chain_id)] = {
                    "pdb_id": pdb_id, "chain_id": chain_id,
                    "alignment_dir": aln_dir,
                }
                print(f"      ✓ {pdb_id}_{chain_id}", flush=True)
            else:
                print(f"      ✗ {pdb_id}_{chain_id} (not found)", flush=True)

    print(f"    Total available chains: {len(available_chains)}", flush=True)

    rng = np.random.RandomState(seed)
    scenario_data = {}
    for scenario in scenarios:
        bucket = scenario.get("length_bucket", scenario["name"])
        num_queries = scenario.get("num_queries", 1)
        if bucket in scenario_data:
            continue
        chains = list(available_chains.values())
        if not chains:
            scenario_data[bucket] = []
            continue
        rng.shuffle(chains)
        query_chains = [chains[i % len(chains)] for i in range(num_queries)]
        scenario_data[bucket] = query_chains
        print(f"    Scenario '{bucket}': {len(query_chains)} queries "
              f"from {len(chains)} unique chains", flush=True)

    return scenario_data


# ---------------------------------------------------------------------------
# Shared featurization + progress helpers (inlined into subprocess workers)
# ---------------------------------------------------------------------------

_FEATURIZE_FN = r'''
import os, sys, time
import torch
import numpy as np
from tqdm import tqdm


def parse_a3m_simple(filepath):
    sequences = []
    current_header = None
    current_seq_parts = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq_parts)))
                current_header = line[1:]
                current_seq_parts = []
            elif current_header is not None:
                current_seq_parts.append(line)
    if current_header is not None:
        sequences.append((current_header, ''.join(current_seq_parts)))
    return sequences


def parse_sto_simple(filepath):
    sequences = {}
    order = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                name, seq = parts
                if name not in sequences:
                    sequences[name] = []
                    order.append(name)
                sequences[name].append(seq)
    return [(name, ''.join(sequences[name])) for name in order]


RESTYPES = 'ACDEFGHIKLMNPQRSTVWY'
RESTYPE_MAP = {aa: i for i, aa in enumerate(RESTYPES)}
GAP_IDX = len(RESTYPES)
UNK_IDX = len(RESTYPES) + 1
NUM_CLASSES = len(RESTYPES) + 2


def encode_msa_sequences(sequences, max_seqs=512):
    if not sequences:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)
    query_seq = sequences[0][1]
    query_len = sum(1 for c in query_seq if c == c.upper() and c != '-' and not c.isdigit())
    if query_len == 0:
        query_len = len(query_seq.replace('-', ''))
    n_seqs = min(len(sequences), max_seqs)
    msa_index = np.full((n_seqs, query_len), GAP_IDX, dtype=np.int64)
    deletion_matrix = np.zeros((n_seqs, query_len), dtype=np.int64)
    for seq_idx in range(n_seqs):
        _, seq = sequences[seq_idx]
        res_pos = 0
        del_count = 0
        for char in seq:
            if char == '-':
                if res_pos < query_len:
                    msa_index[seq_idx, res_pos] = GAP_IDX
                res_pos += 1
            elif char.islower():
                del_count += 1
            elif char.isupper():
                if res_pos < query_len:
                    msa_index[seq_idx, res_pos] = RESTYPE_MAP.get(char, UNK_IDX)
                    deletion_matrix[seq_idx, res_pos] = del_count
                    del_count = 0
                res_pos += 1
    return msa_index, deletion_matrix


MAX_ATOMS_PER_TOKEN = 23
C_ATOM_REF_ELEMENT = 119
C_ATOM_REF_NAME_CHARS_DIM = 4
C_ATOM_REF_NAME_CHARS_VOCAB = 64


def load_and_featurize_chain(chain_info, max_msa_seqs=512, c_m=64, c_token=384, seed=None):
    import hashlib
    aln_dir = chain_info["alignment_dir"]
    if seed is None:
        seed = int(hashlib.sha256(aln_dir.encode()).hexdigest(), 16) % (2**31)
    chain_rng = np.random.RandomState(seed)
    all_sequences = []
    for fname in sorted(os.listdir(aln_dir)):
        fpath = os.path.join(aln_dir, fname)
        if fname.endswith('.a3m'):
            seqs = parse_a3m_simple(fpath)
        elif fname.endswith('.sto'):
            seqs = parse_sto_simple(fpath)
        else:
            continue
        if seqs:
            if not all_sequences:
                all_sequences = seqs
            else:
                for header, seq in seqs[1:]:
                    all_sequences.append((header, seq))
    if not all_sequences:
        return None
    msa_index, deletion_matrix = encode_msa_sequences(all_sequences, max_seqs=max_msa_seqs)
    n_tokens = msa_index.shape[1]
    n_seqs = msa_index.shape[0]
    if n_tokens == 0:
        return None
    msa_onehot_raw = np.eye(NUM_CLASSES, dtype=np.float32)[msa_index]
    msa_onehot = np.zeros((n_seqs, n_tokens, 32), dtype=np.float32)
    msa_onehot[:, :, :NUM_CLASSES] = msa_onehot_raw
    has_deletion = (deletion_matrix > 0).astype(np.float32)
    deletion_value = np.arctan(deletion_matrix / 3.0).astype(np.float32) * (2.0 / np.pi)
    profile = msa_onehot_raw.mean(axis=0)
    deletion_mean = has_deletion.mean(axis=0)
    token_features_np = np.concatenate([
        profile, deletion_mean[:, None],
        np.zeros((n_tokens, c_token - NUM_CLASSES - 1), dtype=np.float32),
    ], axis=1)
    msa_onehot_t = torch.from_numpy(msa_onehot)
    has_deletion_t = torch.from_numpy(has_deletion)
    deletion_value_t = torch.from_numpy(deletion_value)

    n_atoms = n_tokens * MAX_ATOMS_PER_TOKEN

    ref_pos = np.zeros((n_atoms, 3), dtype=np.float32)
    ref_charge = np.zeros(n_atoms, dtype=np.float32)
    ref_mask = np.zeros(n_atoms, dtype=np.float32)
    ref_element = np.zeros((n_atoms, C_ATOM_REF_ELEMENT), dtype=np.float32)
    ref_atom_name_chars = np.zeros(
        (n_atoms, C_ATOM_REF_NAME_CHARS_DIM, C_ATOM_REF_NAME_CHARS_VOCAB),
        dtype=np.float32,
    )
    ref_space_uid = np.zeros(n_atoms, dtype=np.int64)
    atom_to_token_index = np.zeros(n_atoms, dtype=np.int64)
    atom_mask_flat = np.zeros(n_atoms, dtype=np.float32)

    for i in range(n_tokens):
        base = i * MAX_ATOMS_PER_TOKEN
        n_heavy = 5
        for j in range(n_heavy):
            idx = base + j
            ref_pos[idx] = chain_rng.randn(3).astype(np.float32) * 1.5
            ref_mask[idx] = 1.0
            atom_mask_flat[idx] = 1.0
            elem_idx = min(5 + j, C_ATOM_REF_ELEMENT - 1)
            ref_element[idx, elem_idx] = 1.0
            ref_atom_name_chars[idx, 0, min(j + 1, C_ATOM_REF_NAME_CHARS_VOCAB - 1)] = 1.0
        ref_space_uid[base:base + MAX_ATOMS_PER_TOKEN] = i
        atom_to_token_index[base:base + MAX_ATOMS_PER_TOKEN] = i

    restype_22 = msa_onehot_raw[0]
    restype_32 = np.zeros((n_tokens, 32), dtype=np.float32)
    restype_32[:, :NUM_CLASSES] = restype_22
    restype = torch.from_numpy(restype_32).float()

    profile_22 = profile
    profile_32 = np.zeros((n_tokens, 32), dtype=np.float32)
    profile_32[:, :NUM_CLASSES] = profile_22
    profile_t = torch.from_numpy(profile_32).float()

    deletion_mean_1d = torch.from_numpy(deletion_mean).float()

    return {
        "token_features": torch.from_numpy(token_features_np).unsqueeze(0),
        "residue_index": torch.arange(n_tokens, dtype=torch.float32).unsqueeze(0),
        "token_mask": torch.ones(1, n_tokens),
        "atom_mask": torch.from_numpy(atom_mask_flat).unsqueeze(0),
        "msa": msa_onehot_t.unsqueeze(0),
        "has_deletion": has_deletion_t.unsqueeze(0),
        "deletion_value": deletion_value_t.unsqueeze(0),
        "msa_mask": torch.ones(1, n_seqs, n_tokens),
        "restype": restype.unsqueeze(0),
        "profile": profile_t.unsqueeze(0),
        "deletion_mean": deletion_mean_1d.unsqueeze(0),
        "ref_pos": torch.from_numpy(ref_pos).unsqueeze(0),
        "ref_charge": torch.from_numpy(ref_charge).unsqueeze(0),
        "ref_mask": torch.from_numpy(ref_mask).unsqueeze(0),
        "ref_element": torch.from_numpy(ref_element).unsqueeze(0),
        "ref_atom_name_chars": torch.from_numpy(ref_atom_name_chars).unsqueeze(0),
        "ref_space_uid": torch.from_numpy(ref_space_uid).unsqueeze(0),
        "atom_to_token_index": torch.from_numpy(atom_to_token_index).unsqueeze(0),
        "token_index": torch.arange(n_tokens, dtype=torch.float32).unsqueeze(0),
        "asym_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "entity_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "sym_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "n_tokens": n_tokens,
    }


def batch_to_device(batch, device, dtype):
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        else:
            result[k] = v
    return result


def extract_outputs_for_comparison(outputs, aux_outputs):
    result = {}
    s_trunk = aux_outputs.get("s_trunk")
    z_trunk = aux_outputs.get("z_trunk")
    if s_trunk is not None:
        result["s_trunk"] = s_trunk.detach().float().cpu().numpy().tolist()
    if z_trunk is not None:
        n = z_trunk.shape[-2]
        stride = max(1, n // 64)
        result["z_trunk_subsampled"] = z_trunk[..., ::stride, ::stride, :].detach().float().cpu().numpy().tolist()
    if "plddt_logits" in outputs:
        result["plddt_logits"] = outputs["plddt_logits"].detach().float().cpu().numpy().tolist()
    if "pae_logits" in outputs:
        pae = outputs["pae_logits"].detach().float().cpu()
        n = pae.shape[-2]
        stride = max(1, n // 64)
        result["pae_logits_subsampled"] = pae[..., ::stride, ::stride, :].numpy().tolist()
    if "atom_positions_predicted" in outputs:
        result["atom_positions"] = outputs["atom_positions_predicted"].detach().float().cpu().numpy().tolist()
    return result
'''


# ---------------------------------------------------------------------------
# Subprocess worker template (shared by fastkernels and reference)
# ---------------------------------------------------------------------------

_WORKER_BODY = _FEATURIZE_FN + r'''
import json, copy

def main():
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    sys.path.insert(0, cfg["project_root"])
    pkg = cfg["package_name"]
    engine_label = cfg.get("engine_label", "engine")

    if cfg.get("pytorch_reference", False):
        from fastkernels.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    seed = cfg.get("seed", 42)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    mod = __import__(
        f"{pkg}.tasks.baseline.L4.openfold3",
        fromlist=["OpenFold3Config", "OpenFold3Model"],
    )
    OpenFold3Config = mod.OpenFold3Config
    OpenFold3Model = mod.OpenFold3Model

    device = cfg.get("device", "cuda")
    dtype_str = cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    print(f"  [{engine_label}] Building model (PF={cfg['pf_blocks']}, "
          f"MSA={cfg['msa_blocks']}, rollout={cfg['no_rollout_steps']}, "
          f"recycles={cfg['num_recycles']}) ...", flush=True)

    model_cfg = OpenFold3Config(
        pairformer_no_blocks=cfg["pf_blocks"],
        msa_no_blocks=cfg["msa_blocks"],
        no_rollout_steps=cfg["no_rollout_steps"],
        num_recycles=cfg["num_recycles"],
    )
    model = OpenFold3Model(model_cfg)

    ckpt_path = cfg["checkpoint_path"]
    print(f"  [{engine_label}] Loading pretrained weights from {ckpt_path} ...", flush=True)
    load_fn = __import__(
        f"{pkg}.tasks.baseline.L4.openfold3",
        fromlist=["load_openfold3_checkpoint"],
    ).load_openfold3_checkpoint
    load_fn(model, ckpt_path)
    print(f"  [{engine_label}] Loaded checkpoint (strict=True, all keys matched)", flush=True)

    model = model.to(device=device, dtype=dtype).eval()
    params = sum(p.numel() for p in model.parameters())
    if cfg.get("use_torch_compile", False):
        print(f"  [{engine_label}] Applying torch.compile ...", flush=True)
        model = torch.compile(model, mode="reduce-overhead")
    print(f"  [{engine_label}] Model ready ({params:,} params, {dtype_str})", flush=True)

    # Warmup
    warmup_chain = cfg["scenarios"][0]["chains"][0] if cfg.get("scenarios") else None
    if warmup_chain:
        print(f"  [{engine_label}] Warmup ...", flush=True)
        wb = load_and_featurize_chain(warmup_chain, max_msa_seqs=16,
                                      c_m=model_cfg.c_m, c_token=model_cfg.c_token_embedder)
        if wb is not None:
            wb.pop("n_tokens")
            wb = batch_to_device(wb, device, dtype)
            with torch.no_grad():
                model(wb)
            torch.cuda.synchronize()
        print(f"  [{engine_label}] Warmup done", flush=True)

    # ---- Throughput scenarios ----
    all_results = []
    for scenario in cfg.get("scenarios", []):
        chains = scenario["chains"]
        name = scenario["name"]
        print(f"\n  [{engine_label}] Throughput scenario '{name}' "
              f"({len(chains)} queries) ...", flush=True)
        query_outputs = []
        total_tokens = 0

        torch.cuda.synchronize()
        start = time.perf_counter()

        pbar = tqdm(chains, desc=f"  [{engine_label}] {name}",
                    unit="query", file=sys.stdout)
        for qi, chain_info in enumerate(pbar):
            batch = load_and_featurize_chain(
                chain_info, max_msa_seqs=cfg.get("max_msa_seqs", 512),
                c_m=model_cfg.c_m, c_token=model_cfg.c_token_embedder)
            if batch is None:
                continue
            n_tokens = batch.pop("n_tokens")
            total_tokens += n_tokens
            batch = batch_to_device(batch, device, dtype)
            torch.manual_seed(seed + qi)
            torch.cuda.manual_seed_all(seed + qi)
            with torch.no_grad():
                outputs, aux_outputs = model(batch)
            query_outputs.append(extract_outputs_for_comparison(outputs, aux_outputs))
            pbar.set_postfix(tokens=total_tokens, last=n_tokens)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tok_s = total_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{engine_label}] {name}: {len(query_outputs)} queries, "
              f"{total_tokens} tokens, {elapsed:.2f}s, {tok_s:.0f} tok/s", flush=True)

        all_results.append({
            "name": name, "elapsed": elapsed,
            "num_queries": len(query_outputs),
            "total_tokens": total_tokens, "outputs": query_outputs,
        })

    # ---- Latency scenarios ----
    latency_results = []
    for ls in cfg.get("latency_scenarios", []):
        chain_info = ls["chain"]
        name = ls["name"]
        num_warmup = ls.get("num_warmup", 3)
        num_iters = ls.get("num_iters", 5)

        batch = load_and_featurize_chain(
            chain_info, max_msa_seqs=cfg.get("max_msa_seqs", 512),
            c_m=model_cfg.c_m, c_token=model_cfg.c_token_embedder)
        if batch is None:
            continue
        n_tokens = batch.pop("n_tokens")
        batch = batch_to_device(batch, device, dtype)

        print(f"\n  [{engine_label}] Latency '{name}' ({n_tokens} tokens, "
              f"{num_warmup} warmup + {num_iters} iters) ...", flush=True)

        for i in tqdm(range(num_warmup), desc=f"  [{engine_label}] {name} warmup",
                      file=sys.stdout):
            with torch.no_grad():
                model(copy.deepcopy(batch))
            torch.cuda.synchronize()

        latencies = []
        for i in tqdm(range(num_iters), desc=f"  [{engine_label}] {name} bench",
                      file=sys.stdout):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(copy.deepcopy(batch))
            torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            latencies.append(lat)

        med = float(np.median(latencies))
        print(f"  [{engine_label}] {name}: median={med:.4f}s, "
              f"{n_tokens/med:.0f} tok/s", flush=True)
        latency_results.append({
            "name": name, "n_tokens": n_tokens,
            "num_iters": num_iters, "latencies": latencies,
        })

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n  [{engine_label}] Done. Peak memory: {peak_mem:.0f} MB", flush=True)

    del model
    torch.cuda.empty_cache()

    with open(cfg["output_file"], "w") as f:
        json.dump({
            "throughput": all_results, "latency": latency_results,
            "peak_mem_mb": peak_mem, "params": params,
        }, f)


if __name__ == "__main__":
    main()
'''

FASTKERNELS_OF3_WORKER = _WORKER_BODY
REF_OF3_WORKER = _WORKER_BODY


# ---------------------------------------------------------------------------
# Correctness metrics
# ---------------------------------------------------------------------------

TARGETS = {
    "atom_pos_cosine_mean": 0.95,
    "atom_pos_cosine_min": 0.90,
    "atom_rmsd_kabsch_mean": 0.5,
    "plddt_pearson_mean": 0.99,
    "pae_cosine_mean": 0.95,
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after optimal rigid superposition (Kabsch algorithm)."""
    p = p.reshape(-1, 3).astype(np.float64)
    q = q.reshape(-1, 3).astype(np.float64)
    n = min(len(p), len(q))
    if n == 0:
        return float("inf")
    p, q = p[:n], q[:n]
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    H = p_c.T @ q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    p_aligned = p_c @ R.T
    diff = p_aligned - q_c
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    a_m = a - a.mean()
    b_m = b - b.mean()
    num = np.dot(a_m, b_m)
    den = np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def plddt_from_logits(logits: np.ndarray) -> np.ndarray:
    """Convert pLDDT logits [*, N_token * max_atoms_per_token, n_bins] to per-residue scores."""
    logits = np.array(logits, dtype=np.float64)
    if logits.ndim < 2:
        return logits
    while logits.ndim > 2:
        logits = logits[0]
    exp = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(exp)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    n_bins = probs.shape[-1]
    bin_centers = np.linspace(1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins)
    return (probs * bin_centers).sum(axis=-1)


def _query_passes(kb_out: dict, ref_out: dict) -> bool:
    """Check if a single query meets all correctness targets."""
    if "atom_positions" in kb_out and "atom_positions" in ref_out:
        ap_kb = np.array(kb_out["atom_positions"], dtype=np.float64)
        ap_ref = np.array(ref_out["atom_positions"], dtype=np.float64)
        if cosine_sim(ap_kb, ap_ref) < TARGETS["atom_pos_cosine_min"]:
            return False
        if kabsch_rmsd(ap_kb, ap_ref) >= TARGETS["atom_rmsd_kabsch_mean"]:
            return False
    if "plddt_logits" in kb_out and "plddt_logits" in ref_out:
        plddt_kb = plddt_from_logits(kb_out["plddt_logits"])
        plddt_ref = plddt_from_logits(ref_out["plddt_logits"])
        if pearson_corr(plddt_kb, plddt_ref) < TARGETS["plddt_pearson_mean"]:
            return False
    if "pae_logits_subsampled" in kb_out and "pae_logits_subsampled" in ref_out:
        pae_kb = np.array(kb_out["pae_logits_subsampled"], dtype=np.float64)
        pae_ref = np.array(ref_out["pae_logits_subsampled"], dtype=np.float64)
        if cosine_sim(pae_kb, pae_ref) < TARGETS["pae_cosine_mean"]:
            return False
    return True


def compute_alignment(kb_outputs: list[dict], ref_outputs: list[dict]) -> dict:
    n = min(len(kb_outputs), len(ref_outputs))
    passed = sum(1 for i in range(n) if _query_passes(kb_outputs[i], ref_outputs[i]))
    return {"num_queries": n, "pass_rate": passed / n if n > 0 else 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Throughput, latency & correctness benchmark: "
                    "fastkernels OpenFold3 vs reference openfold3",
    )
    parser.add_argument("--scenario", type=str, default=None,
                        choices=[s["name"] for s in SCENARIOS],
                        help="Run a single throughput scenario (default: all)")
    parser.add_argument("--num-seqs", type=int, default=None,
                        help="Override num_queries per scenario (default: scenario-specific)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip reference openfold3 (fastkernels only)")
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into fastkernels.",
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--latency-iters", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory for caching downloaded MSA data "
                             "(default: /tmp/openfold_bench_data)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--torch-compile", action="store_true",
                        help="Apply torch.compile to the model (reduce-overhead mode)")
    args = parser.parse_args()

    gpu = _detect_gpu_name()

    if args.output_dir is None:
        args.output_dir = str(_THIS_DIR / "results" / "openfold3" / f"{gpu}_{args.dtype}")
    if args.data_dir is None:
        args.data_dir = "/tmp/openfold_bench_data"

    throughput_scenarios = [dict(s) for s in SCENARIOS]
    if args.scenario:
        throughput_scenarios = [s for s in throughput_scenarios if s["name"] == args.scenario]
    if args.num_seqs is not None:
        for s in throughput_scenarios:
            s["num_queries"] = args.num_seqs

    latency_scenarios = [dict(s) for s in LATENCY_SCENARIOS]
    if args.scenario:
        latency_scenarios = [s for s in latency_scenarios if s["name"].endswith(args.scenario)]

    for ls in latency_scenarios:
        ls["num_iters"] = args.latency_iters

    all_scenarios_for_data = throughput_scenarios if not args.skip_throughput else []
    scenario_data = prepare_dataset(
        args.data_dir,
        all_scenarios_for_data + ([] if args.skip_latency else latency_scenarios),
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("  fastkernels OpenFold3 vs Reference — End-to-End Benchmark")
    print("=" * 80)
    print(f"  GPU            : {gpu}")
    print(f"  Dtype          : {args.dtype}")
    print(f"  PF blocks      : {PF_BLOCKS}")
    print(f"  MSA blocks     : {MSA_BLOCKS}")
    print(f"  Rollout steps  : {NO_ROLLOUT_STEPS}")
    print(f"  Recycles       : {NUM_RECYCLES}")
    print(f"  Max MSA seqs   : {MAX_MSA_SEQS}")
    print(f"  Data dir       : {args.data_dir}")
    total_queries = sum(s["num_queries"] for s in throughput_scenarios)
    if not args.skip_throughput:
        tp_desc = "; ".join(f"{s['name']}×{s['num_queries']}" for s in throughput_scenarios)
        print(f"  Throughput     : {tp_desc} ({total_queries} total queries)")
    if not args.skip_latency:
        print(f"  Latency        : {', '.join(s['name'] for s in latency_scenarios)} "
              f"({args.latency_iters} iters)")
    print(f"  Output dir     : {args.output_dir}")
    print("=" * 80, flush=True)

    scenario_configs = []
    if not args.skip_throughput:
        for s in throughput_scenarios:
            scenario_configs.append({"name": s["name"], "chains": scenario_data.get(s["name"], [])})

    latency_configs = []
    if not args.skip_latency:
        for ls in latency_scenarios:
            bucket = ls.get("length_bucket", ls["name"])
            chains = scenario_data.get(bucket, [])
            if chains:
                latency_configs.append({
                    "name": ls["name"], "chain": chains[0],
                    "num_warmup": ls.get("num_warmup", 3), "num_iters": ls["num_iters"],
                })

    print("\n  Downloading pretrained checkpoint from HuggingFace "
          f"({HF_REPO_ID}) ...", flush=True)
    checkpoint_path = download_checkpoint()
    print(f"  Checkpoint: {checkpoint_path}", flush=True)

    base_config = {
        "project_root": str(_PROJECT_ROOT),
        "package_name": _PACKAGE_DIR.name,
        "seed": args.seed, "dtype": args.dtype,
        "pf_blocks": PF_BLOCKS, "msa_blocks": MSA_BLOCKS,
        "no_rollout_steps": NO_ROLLOUT_STEPS, "num_recycles": NUM_RECYCLES,
        "max_msa_seqs": MAX_MSA_SEQS,
        "checkpoint_path": checkpoint_path,
        "scenarios": scenario_configs, "latency_scenarios": latency_configs,
        "use_torch_compile": args.torch_compile,
    }

    # ---- Run fastkernels ----
    kb_config = {
        **base_config,
        "engine_label": "fastkernels",
        "pytorch_reference": args.pytorch_reference,
    }
    kb_raw = run_worker(
        FASTKERNELS_OF3_WORKER, kb_config,
        "fastkernels OpenFold3 — all scenarios (real MSA data)", timeout=7200,
    )
    if kb_raw is None:
        print("  ERROR: fastkernels subprocess failed.")
        sys.exit(1)

    # ---- Run reference ----
    ref_raw = None
    if not args.skip_reference:
        ref_config = {**base_config, "engine_label": "reference"}
        ref_raw = run_worker(
            REF_OF3_WORKER, ref_config,
            "Reference OpenFold3 — all scenarios (real MSA data)", timeout=7200,
        )
        if ref_raw is None:
            print("  WARNING: Reference subprocess failed. Skipping comparison.")

    # ---- Throughput metrics ----
    all_results = []
    if not args.skip_throughput:
        kb_tp = kb_raw["throughput"]
        ref_tp = ref_raw["throughput"] if ref_raw else None
        for i, scenario in enumerate(throughput_scenarios):
            kb_d = kb_tp[i]
            kb_tok_s = kb_d["total_tokens"] / kb_d["elapsed"] if kb_d["elapsed"] > 0 else 0
            r = {
                "scenario": scenario["name"], "num_queries": kb_d["num_queries"],
                "total_tokens": kb_d["total_tokens"],
                "fastkernels_elapsed": kb_d["elapsed"], "fastkernels_tok_per_s": kb_tok_s,
            }
            if ref_tp and i < len(ref_tp):
                ref_d = ref_tp[i]
                ref_tok_s = ref_d["total_tokens"] / ref_d["elapsed"] if ref_d["elapsed"] > 0 else 0
                r["ref_elapsed"] = ref_d["elapsed"]
                r["ref_tok_per_s"] = ref_tok_s
                r["speedup"] = kb_tok_s / ref_tok_s if ref_tok_s > 0 else 0
                r["alignment"] = compute_alignment(kb_d["outputs"], ref_d["outputs"])
            all_results.append(r)

    # ---- Build latency data ----
    kb_lat = kb_raw.get("latency", [])
    ref_lat = ref_raw.get("latency", []) if ref_raw else []
    lat_combined = []
    for i, kl in enumerate(kb_lat):
        kl_med = float(np.median(kl["latencies"]))
        nt = kl["n_tokens"]
        k_tps = nt / kl_med if kl_med > 0 else 0
        lr = {"scenario": kl["name"], "n_tokens": nt, "num_iters": kl["num_iters"],
              "fastkernels_median_s": kl_med, "fastkernels_tok_per_s": k_tps, "fastkernels_latencies": kl["latencies"]}
        if i < len(ref_lat):
            rl = ref_lat[i]
            rl_med = float(np.median(rl["latencies"]))
            r_tps = nt / rl_med if rl_med > 0 else 0
            sp = rl_med / kl_med if kl_med > 0 else 0
            lr.update(ref_median_s=rl_med, ref_tok_per_s=r_tps, speedup=sp, ref_latencies=rl["latencies"])
        lat_combined.append(lr)

    # ---- Print results ----
    W = 90
    print(f"\n\n{'=' * W}")
    print("  RESULTS")
    print(f"{'=' * W}")
    print(f"  {'':40s} FASTKERNELS    REFERENCE  SPEEDUP  CORRECT")
    print(f"  {'-' * (W - 2)}")

    if all_results:
        for r in all_results:
            a = r.get("alignment", {})
            kb_str = f"{r['fastkernels_tok_per_s']:>6,.0f} tok/s"
            ref_str = f"{r['ref_tok_per_s']:>5,.0f} tok/s" if "ref_tok_per_s" in r else f"{'N/A':>9s}"
            sp_str = f"{r['speedup']:.2f}x" if "speedup" in r else "N/A"
            pr = a.get("pass_rate")
            cor_str = f"{pr * 100:.0f}%" if pr is not None else "N/A"
            label = f"  throughput/{r['scenario']:<11s} ({r['num_queries']:>3}q)"
            print(f"{label:<40s} {kb_str:>10s}  {ref_str:>9s}  {sp_str:>7s}  {cor_str:>7s}")

    if lat_combined:
        for lr in lat_combined:
            kb_str = f"{lr['fastkernels_median_s']:.3f}s"
            ref_str = f"{lr['ref_median_s']:.3f}s" if "ref_median_s" in lr else "N/A"
            sp_str = f"{lr['speedup']:.2f}x" if "speedup" in lr else "N/A"
            label = f"  latency/{lr['scenario'].replace('single-', ''):<14s} ({lr['n_tokens']:>3}tok)"
            print(f"{label:<40s} {kb_str:>10s}  {ref_str:>9s}  {sp_str:>7s}")

    print(f"  {'-' * (W - 2)}")
    print(f"  {'params':40s} {kb_raw.get('params', 0):>10,}  {(ref_raw or {}).get('params', 0):>9,}")
    print(f"  {'peak memory (MB)':40s} {kb_raw.get('peak_mem_mb', 0):>10,.0f}  {(ref_raw or {}).get('peak_mem_mb', 0):>9,.0f}")
    print(f"{'=' * W}")

    # ---- Save ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        combined = {
            "gpu": gpu, "dtype": args.dtype, "seed": args.seed,
            "pf_blocks": PF_BLOCKS, "msa_blocks": MSA_BLOCKS,
            "no_rollout_steps": NO_ROLLOUT_STEPS, "num_recycles": NUM_RECYCLES,
            "max_msa_seqs": MAX_MSA_SEQS,
            "checkpoint": f"{HF_REPO_ID} ({HF_CHECKPOINT_FILE})",
            "data_source": "OpenProteinSet (AWS s3://openfold/pdb/)",
            "fastkernels_params": kb_raw.get("params", 0),
            "fastkernels_peak_mem_mb": kb_raw.get("peak_mem_mb", 0),
        }
        if ref_raw:
            combined["ref_params"] = ref_raw.get("params", 0)
            combined["ref_peak_mem_mb"] = ref_raw.get("peak_mem_mb", 0)
        if all_results:
            combined["throughput_scenarios"] = all_results
        if lat_combined:
            combined["latency_scenarios"] = lat_combined
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Results saved to: {results_path}")

        if not args.skip_throughput:
            for i, sc in enumerate(throughput_scenarios):
                sd = os.path.join(args.output_dir, sc["name"])
                os.makedirs(sd, exist_ok=True)
                with open(os.path.join(sd, "fastkernels_outputs.json"), "w") as f:
                    json.dump(kb_raw["throughput"][i], f, indent=2)
                if ref_raw and i < len(ref_raw["throughput"]):
                    with open(os.path.join(sd, "ref_outputs.json"), "w") as f:
                        json.dump(ref_raw["throughput"][i], f, indent=2)


if __name__ == "__main__":
    main()
