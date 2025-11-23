# === IMPORTS & SETUP ===
!pip install stim
!pip install ldpc

import numpy as np
import stim
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import time
from scipy.linalg import null_space
from ldpc import bplsd_decoder


# === GF(2) LINEAR ALGEBRA HELPERS ===

import numpy as np

def gf2_rref(A):
    """RREF over GF(2). Returns (R, pivot_cols)."""
    A = (A % 2).astype(np.uint8).copy()
    m, n = A.shape
    r = 0
    pivots = []
    for c in range(n):
        piv = np.where(A[r:, c] == 1)[0]
        if piv.size == 0:
            continue
        piv = piv[0] + r
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots

def gf2_rank(A):
    _, piv = gf2_rref(A)
    return len(piv)

def gf2_nullspace(A):
    A = (A % 2).astype(np.uint8)
    m, n = A.shape
    R, piv = gf2_rref(A)
    piv = list(piv)
    free = [j for j in range(n) if j not in piv]
    if len(free) == 0:
        return np.zeros((0, n), dtype=np.uint8)
    N = np.zeros((len(free), n), dtype=np.uint8)
    for k, f in enumerate(free):
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for r, c in enumerate(piv):
            if R[r, f]:
                v[c] ^= 1
        N[k, :] = v
    return N

def gf2_is_in_rowspace(v, H):
    if H.size == 0:
        return np.all((v % 2) == 0)
    H_aug = np.vstack([H % 2, (v % 2)])
    return gf2_rank(H_aug) == gf2_rank(H)

def gf2_independent_mod_rowspace(v, H, S):
    if S.size == 0:
        base = gf2_rank(H)
        added = gf2_rank(np.vstack([H, v % 2]))
        return added > base
    base = gf2_rank(np.vstack([H, S]))
    added = gf2_rank(np.vstack([H, S, v % 2]))
    return added > base

def gf2_inv(M):
    M = (M % 2).astype(np.uint8)
    k = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix to invert must be square over GF(2).")
    A = np.hstack([M.copy(), np.eye(k, dtype=np.uint8)])
    r = 0
    c = 0
    while r < k and c < k:
        piv_rows = np.where(A[r:, c] == 1)[0]
        if piv_rows.size == 0:
            c += 1
            continue
        piv = piv_rows[0] + r
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        for i in range(k):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        r += 1
        c += 1
    if not np.array_equal(A[:, :k], np.eye(k, dtype=np.uint8)):
        raise ValueError("Singular matrix over GF(2).")
    return A[:, k:]


# === CSS LOGICAL OPERATORS ===

def css_logical_ops(Hx, Hz):
    Hx = (Hx % 2).astype(np.uint8)
    Hz = (Hz % 2).astype(np.uint8)
    n = Hx.shape[1]
    assert Hz.shape[1] == n

    rx = gf2_rank(Hx)
    rz = gf2_rank(Hz)
    k_expected = n - rx - rz
    if k_expected < 0:
        raise ValueError("Inconsistent CSS checks")

    NX = gf2_nullspace(Hz)
    NZ = gf2_nullspace(Hx)

    Lx_rows = []
    for v in NX:
        if gf2_independent_mod_rowspace(v, Hx, np.vstack(Lx_rows) if Lx_rows else np.zeros((0, n), np.uint8)):
            Lx_rows.append(v.copy())
        if len(Lx_rows) == k_expected:
            break

    Lx = np.vstack(Lx_rows) if Lx_rows else np.zeros((0, n), np.uint8)

    Lz_rows = []
    for v in NZ:
        if gf2_independent_mod_rowspace(v, Hz, np.vstack(Lz_rows) if Lz_rows else np.zeros((0, n), np.uint8)):
            Lz_rows.append(v.copy())
        if len(Lz_rows) == k_expected:
            break

    Lz = np.vstack(Lz_rows) if Lz_rows else np.zeros((0, n), np.uint8)

    kx, kz = Lx.shape[0], Lz.shape[0]
    k = min(k_expected, kx, kz)
    Lx = Lx[:k, :]
    Lz = Lz[:k, :]

    if k == 0:
        return Lx, Lz

    P = (Lx @ Lz.T) % 2
    if gf2_rank(P) < k:
        raise RuntimeError("Failed to get full-rank pairing")
    Pinv = gf2_inv(P)
    Lz = ((Pinv.T @ Lz) % 2).astype(np.uint8)
    return Lx.astype(np.uint8), Lz.astype(np.uint8)


# === CIRCULANT MATRIX CONSTRUCTION ===

def poly_to_circulant(n, index, right_rotate=True):
    r0 = np.zeros(n, dtype=np.uint8)
    for i in index:
        r0[i % n] ^= 1
    M = np.zeros((n, n), dtype=np.uint8)
    row = r0.copy()
    for i in range(n):
        M[i] = row
        row = np.roll(row, 1 if right_rotate else -1)
    return M


# === SPECIFY n, a_index, b_index ===

n = 4
a_index = [0, 2]
b_index = [1, 3]


# === BUILD CSS MATRICES H_X, H_Z ===

A = poly_to_circulant(n, a_index)
B = poly_to_circulant(n, b_index)

print("A:\n", A)
print("B:\n", B)

H_X = np.concatenate([A, B], axis=1)
H_Z = np.concatenate([B.T, A.T], axis=1)

print("Hx:\n", H_X)
print("Hz:\n", H_Z)

check = np.dot(H_X, H_Z.T) % 2
if np.all(check == 0):
    print("Valid CSS construction")
    print("Dimension k=", 2*n - gf2_rank(H_X) - gf2_rank(H_Z))
else:
    print("Invalid CSS construction")


Lx, Lz = css_logical_ops(H_X, H_Z)

print("Lx shape:", Lx.shape)
print("Lz shape:", Lz.shape)
print("Lx:\n", Lx)
print("Lz:\n", Lz)

lx_hx_commutation = (Lx @ H_Z.T) % 2
lz_hz_commutation = (Lz @ H_X.T) % 2
lx_lz_pairing = (Lx @ Lz.T) % 2

print("Lx @ Hx.T:\n", lx_hx_commutation)
print("Lz @ Hz.T:\n", lz_hz_commutation)
print("Lx @ Lz.T:\n", lx_lz_pairing)


# === DIAGONAL GROUPS ===

def diagonal_groups(H):
    m, total_n = H.shape
    n = total_n // 2
    left = H[:, :n]
    right = H[:, n:]
    regions = []

    def extract(block, offset):
        m2, n2 = block.shape
        for shift in range(n2):
            diag_idx = [(i, (i + shift) % n2 + offset) for i in range(m2)]
            if all(block[i, (i + shift) % n2] == 1 for i in range(m2)):
                regions.append(diag_idx)

    extract(left, 0)
    extract(right, n)
    return regions


# === STIM BICYCLE CIRCUIT BUILDER ===

def build_bicycle_circuit(H_X, H_Z, p_depol=0.001, N_rounds=1):
    L_X, L_Z = css_logical_ops(H_X, H_Z)
    n_data = H_X.shape[1]
    n_z_checks = H_Z.shape[0]
    n_x_checks = H_X.shape[0]

    data = list(range(n_data))
    x_anc = list(range(0, n_x_checks))
    data  = list(range(n_x_checks, n_x_checks + n_data))
    z_anc = list(range(n_x_checks + n_data,
                      n_x_checks + n_data + n_z_checks))

    x_checks = [list(np.flatnonzero(H_X[i])) for i in range(n_x_checks)]
    z_checks = [list(np.flatnonzero(H_Z[i])) for i in range(n_z_checks)]
    X_groups = diagonal_groups(H_X)
    Z_groups = diagonal_groups(H_Z)
    Z_groups_rev = Z_groups[::-1]

    c = stim.Circuit()
    c.append_operation("RX",  data)
    c.append_operation("RX",  z_anc)
    c.append_operation("RX", x_anc)
    c.append_operation("Z_ERROR", data, p_depol)
    c.append_operation("Z_ERROR", z_anc, p_depol)
    c.append_operation("Z_ERROR", x_anc, p_depol)

    all_qubits = data + z_anc + x_anc
    for t in range(N_rounds):
        for k in range(len(X_groups)):
            c.append("TICK")
            active = set()

            for row, col in Z_groups_rev[k]:
                a = z_anc[row]
                q = data[col]
                c.append_operation("CNOT", [q, a])
                c.append_operation("DEPOLARIZE2", [q, a], p_depol)
                active.add(q)
                active.add(a)

            for row, col in X_groups[k]:
                a = x_anc[row]
                q = data[col]
                c.append_operation("CNOT", [a, q])
                c.append_operation("DEPOLARIZE2", [a, q], p_depol)
                active.add(q)
                active.add(a)

            for q in all_qubits:
                if q not in active:
                    c.append_operation("DEPOLARIZE1", [q], p_depol)

        c.append_operation("Z_ERROR", x_anc, p_depol)
        c.append_operation("MX", x_anc)

        for i in range(n_x_checks):
            curr = -((t + 1) * n_x_checks) + i
            if t == 0:
                c.append("DETECTOR", [stim.target_rec(curr)])
            else:
                prev = -(t * n_x_checks) + i
                c.append("DETECTOR", [
                    stim.target_rec(curr),
                    stim.target_rec(prev)
                ])

        c.append_operation("MX", data)

        stab_circuit_str = ""
        for i, l in enumerate(H_X):
            nnz = np.nonzero(l)[0]
            det_str = f"DETECTOR({i})"
            for ind in nnz:
                det_str += f" rec[{-n_data+ind}]"
            det_str += "\n"
            stab_circuit_str  += det_str
        stab_detector_circuit = stim.Circuit(stab_circuit_str )
        c += stab_detector_circuit

        log_detector_circuit_str = ""
        for i, l in enumerate(L_Z):
            nnz = np.nonzero(l)[0]
            det_str = f"OBSERVABLE_INCLUDE({i})"
            for ind in nnz:
                det_str += f" rec[{-n_data+ind}]"
            det_str += "\n"
            log_detector_circuit_str += det_str
        log_detector_circuit = stim.Circuit(log_detector_circuit_str)
        c += log_detector_circuit

    return c


# === SAMPLING FROM CIRCUIT ===

def sample_from_circuit(circuit, num_samples=1000):
    dem = circuit.detector_error_model()
    sampler = dem.compile_sampler()
    samples = sampler.sample(num_samples)
    detections = samples[0]
    observable_flips = samples[1]
    return detections, observable_flips


# === DEM → CHECK MATRICES ===

def dict_to_csc_matrix(elements_dict, shape):
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)

def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):
    DL_ids = {}
    L_map = {}
    priors_dict = {}

    def handle_error(prob, detectors, observables):
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join(
            [f"D{s}" for s in sorted(dets)] +
            [f"L{s}" for s in sorted(obs)]
        )
        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0
        hid = DL_ids[key]
        L_map[hid] = obs
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets = []
            frames = []
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)

    check_matrix = dict_to_csc_matrix(
        {
            v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")]
            for k, v in DL_ids.items()
        },
        shape=(dem.num_detectors, len(DL_ids))
    )
    observables_matrix = dict_to_csc_matrix(L_map, shape=(dem.num_observables, len(DL_ids)))
    priors = np.array([priors_dict[i] for i in range(len(DL_ids))])

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors


# === CUSTOM BP-SP DECODER ===

def bp_sp_decode(H, syndrome, p, max_iters=20, clip=20.0):
    if not isinstance(H, np.ndarray):
        H = H.toarray()

    H = (H.astype(np.uint8) & 1)
    m, n = H.shape

    if syndrome.size == 0:
        return np.zeros(n, dtype=np.uint8)

    s = syndrome.flatten().astype(np.uint8) & 1
    variable_nodes = [np.flatnonzero(H[:, j]) for j in range(n)]
    check_nodes = [np.flatnonzero(H[i, :]) for i in range(m)]

    L_prior = np.log((1.0 - p) / p)

    L_v_to_c = {}
    for v in range(n):
        for c in variable_nodes[v]:
            L_v_to_c[(v, c)] = np.clip(L_prior, -clip, clip)

    L_c_to_v = {}

    for _ in range(max_iters):
        for c in range(m):
            vars_c = check_nodes[c]
            if len(vars_c) == 0:
                continue
            th = np.tanh(0.5 * np.array([L_v_to_c[(v, c)] for v in vars_c], dtype=float))
            th = np.clip(th, -1.0 + 1e-12, 1.0 - 1e-12)
            prod_all = np.prod(th)
            for i, v in enumerate(vars_c):
                arg = prod_all / th[i]
                if s[c]:
                    arg = -arg
                arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
                L_c_to_v[(c, v)] = np.clip(2.0 * np.arctanh(arg), -clip, clip)

        L_post = np.full(n, L_prior, dtype=float)
        for v in range(n):
            checks_v = variable_nodes[v]
            if len(checks_v) == 0:
                continue
            sum_in = sum(L_c_to_v[(c, v)] for c in checks_v)
            L_post[v] += sum_in
            for c in checks_v:
                L_v_to_c[(v, c)] = np.clip(L_prior + (sum_in - L_c_to_v[(c, v)]),
                                           -clip, clip)

        e_hat = (L_post < 0).astype(np.uint8)
        s_hat = (H @ e_hat) % 2
        if np.array_equal(s_hat, s):
            break

    return e_hat


# === MONTE CARLO COMPARISON ===

def monte_carlo_bp_sd_vs_bplds_simulation(H_X, H_Z, failures, max_iters, clip=20.0):
    p_values = np.logspace(-2, -1, 10)
    ps, sp_rates, bplds_rates = [], [], []
    batch = 100

    for p_depol in p_values:
        p_eff = float(p_depol)
        circuit = build_bicycle_circuit(H_X, H_Z, p_eff)
        dem = circuit.detector_error_model()
        H_dem, O_dem, priors = dem_to_check_matrices(dem)

        decoder = bplsd_decoder.BpLsdDecoder(
            H_dem,
            error_channel=[p_eff] * H_dem.shape[1],
            lsd_method='lsd_cs',
            lsd_order=10,
            max_iter=max_iters,
            ms_scaling_factor=0.875,
            schedule="parallel"
        )

        logical_errors_sp = 0
        logical_errors_bplds = 0
        shots = 0

        print(f"\nRunning p = {p_eff:.1e}")
        while logical_errors_sp < failures or logical_errors_bplds < failures or shots < batch:
            detections, observable_flips = sample_from_circuit(circuit, batch)
            n_checks = H_dem.shape[0]
            x_syndromes = detections

            for i in range(batch):
                raw = x_syndromes[i][-n_checks:]
                syndrome = np.asarray(raw, dtype=np.uint8).reshape(-1)
                obs_flip = observable_flips[i]

                e_hat_sp = bp_sp_decode(H_dem, syndrome, p_eff, max_iters=max_iters, clip=clip)
                est_logical_sp = (e_hat_sp @ O_dem.T) % 2
                total_logical_sp = (obs_flip + est_logical_sp) % 2
                if np.any(total_logical_sp):
                    logical_errors_sp += 1

                e_hat_bplds = decoder.decode(syndrome)
                est_logical_bplds = (e_hat_bplds @ O_dem.T) % 2
                total_logical_bplds = (obs_flip + est_logical_bplds) % 2
                if np.any(total_logical_bplds):
                    logical_errors_bplds += 1

                shots += 1

            if logical_errors_sp >= failures and logical_errors_bplds >= failures:
                break

            if shots > 1e6:
                print("⚠️ Stopping early: reached max shots.")
                break

        print(logical_errors_bplds)
        print(logical_errors_sp)
        rate_sp = logical_errors_sp / shots
        rate_bplds = logical_errors_bplds / shots
        ps.append(p_eff)
        sp_rates.append(rate_sp)
        bplds_rates.append(rate_bplds)

        print(f"Custom BP-SP → {logical_errors_sp}/{shots} = {rate_sp:.3e}")
        print(f"LDPC bplds Decoder → {logical_errors_bplds}/{shots} = {rate_bplds:.3e}")

    plt.figure(figsize=(7,5))
    plt.loglog(ps, sp_rates, 'o-', label='Custom BP-SP Decoder')
    plt.loglog(ps, bplds_rates, 's--', label='LDPC bplds_decoder')
    plt.xlabel('Physical error rate (p)')
    plt.ylabel('Logical failure rate')
    plt.title('Monte Carlo Comparison – BP-SP vs bplds Decoder')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.show()

    return np.array(ps), np.array(sp_rates), np.array(bplds_rates)


# === MONTE CARLO CALL ===

ps, my_rates, ldpc_rates = monte_carlo_bp_sd_vs_bplds_simulation(
    H_X, H_Z,
    failures=20,
    max_iters=50,
    clip=20.0
)
