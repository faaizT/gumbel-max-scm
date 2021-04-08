from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.DataGenerator import DataGenerator
import sepsisSimDiabetes.MDP as simulator
import cf.counterfactual as cf
import pickle
import argparse
import numpy as np
from scipy.linalg import block_diag
import mdptoolboxSrc.mdp as mdptools
import pandas as pd


def main(args):
    DISCOUNT_Pol = 0.99  # Used for computing optimal policies
    PROB_DIAB = 0.2
    PHYS_EPSILON = 0.05  # Used for sampling using physician pol as eps greedy
    n_actions = Action.NUM_ACTIONS_TOTAL
    n_components = 2
    n_steps = args.nsteps
    n_sims = args.nsims

    with open(args.physpol, "rb") as f:
        mdict = pickle.load(f)
    tx_mat = mdict["tx_mat"]
    r_mat = mdict["r_mat"]
    p_mixture = np.array([1 - PROB_DIAB, PROB_DIAB])

    tx_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))
    r_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))

    for a in range(n_actions):
        tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a, ...])
        r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])
    fullMDP = cf.MatrixMDP(tx_mat_full, r_mat_full)
    fullPol = fullMDP.policyIteration(discount=DISCOUNT_Pol, eval_type=1)

    physPolSoft = np.copy(fullPol)
    physPolSoft[physPolSoft == 1] = 1 - PHYS_EPSILON
    physPolSoft[physPolSoft == 0] = PHYS_EPSILON / (n_actions - 1)
    dgen = DataGenerator()
    (
        states,
        actions,
        lengths,
        rewards,
        diab,
        emp_tx_totals,
        emp_r_totals,
    ) = dgen.simulate(
        n_sims,
        n_steps,
        policy=physPolSoft,
        policy_idx_type="full",
        p_diabetes=PROB_DIAB,
        output_state_idx_type="full",
        use_tqdm=False,
        noisy_state_vecs=True,
    )
    states_reshaped = np.reshape(states, ((n_steps + 1) * n_sims, 8))
    ids = np.reshape(
        [[i] * (n_steps + 1) for i in range(n_sims)],
        ((n_steps + 1) * n_sims, 1),
    )
    data = np.append(states_reshaped, ids, axis=1)
    actions = np.reshape(
        [np.append(i, -1) for i in actions], ((n_steps + 1) * n_sims, 1)
    )
    data = np.append(data, actions, axis=1)
    times = np.reshape([range(n_steps + 1)] * n_sims, ((n_steps + 1) * n_sims, 1))
    data = np.append(data, times, axis=1)
    df = pd.DataFrame(
        columns=[
            "hr_state",
            "sysbp_state",
            "percoxyg_state",
            "glucose_state",
            "antibiotic_state",
            "vaso_state",
            "vent_state",
            "diabetic_idx",
            "id",
            "A_t",
            "t",
        ],
        data=data,
    )
    df.to_csv(f"{args.exportdir}/data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exportdir", help="Dir to store csv data", type=str)
    parser.add_argument(
        "--nsteps", help="maximum number of steps in trajectory", type=int, default=20
    )
    parser.add_argument(
        "--nsims", help="number of trajectories", type=int, default=10000
    )
    parser.add_argument(
        "--physpol",
        help="file containing reward and transition matrix for physpol",
        type=str,
        default="/data/localhost/taufiq/gumbel-max-scm-exportdir/data/diab_txr_mats-replication.pkl",
    )
    args = parser.parse_args()
    main(args)