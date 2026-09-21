#!/usr/bin/env python3
"""Navigation task: fly navigates to food using the real brain wiring.

Simplified version of working-brain/main.py using the flynet library.
"""

import numpy as np
from flynet.connectome import ConnectomeLoader
from flynet.network import SpikingNetwork
from flynet.learning import RewardHebbian, TrainingLogger

# Arena parameters
ARENA = 100.0
FOOD = np.array([80.0, 80.0])
START = np.array([15.0, 20.0])
FOUND_RADIUS = 4.0
MAX_STEPS = 400
LOOKAHEAD = 9.0
ANTENNA_OFFSET = 2.7
STEER_GAIN = 6.0
MAX_TURN = 0.45
STEP_LEN = 2.0
WANDER_STEP = 0.5
WANDER_TURN = 1.5


def smell(pos):
    d = float(np.linalg.norm(FOOD - pos))
    return 2.0 / (1.0 + 0.3 * d**2)


def play_game(net, rng, max_steps=MAX_STEPS, W=None):
    pos = START.astype(float).copy()
    geo = float(rng.uniform(-np.pi, np.pi))
    path = [pos.copy()]
    events = []

    for step in range(max_steps):
        if np.linalg.norm(pos - FOOD) < FOUND_RADIUS:
            return step, path, events, True

        d = np.array([np.cos(geo), np.sin(geo)])
        s = np.array([-d[1], d[0]])

        if smell(pos + d * LOOKAHEAD) > smell(pos - d * LOOKAHEAD):
            rhoL = smell(pos + d * ANTENNA_OFFSET + s * ANTENNA_OFFSET)
            rhoR = smell(pos + d * ANTENNA_OFFSET - s * ANTENNA_OFFSET)
            drv, tr = net.turn(rhoR, rhoL)
            steering = np.clip(-STEER_GAIN * drv, -MAX_TURN, MAX_TURN)
            geo += steering
            nxt = pos + d * STEP_LEN
            a = np.mean(np.array(tr[4:]), axis=0)
            events.append((rhoR, rhoL, a))
        else:
            geo += rng.uniform(-WANDER_TURN, WANDER_TURN)
            nxt = pos + d * WANDER_STEP

        nxt = np.clip(nxt, 0.0, ARENA)
        pos = nxt
        path.append(pos.copy())

    return None, path, events, False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.6)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--eval-seeds", type=int, default=20)
    args = p.parse_args()

    loader = ConnectomeLoader()
    ids, edges, motors = loader.load_or_fetch_mini(neuron_type="DNge104")
    net = SpikingNetwork(ids, edges, motor_ids=motors)

    print(f"Brain: {len(net.ids)} neurons, {net.W.nnz} synapses")

    hebbian = RewardHebbian(eta=args.eta)
    logger = TrainingLogger()

    for ep in range(1, args.episodes + 1):
        rng = np.random.default_rng(args.seed + ep)
        steps, path, events, found = play_game(net, rng)
        logger.log_episode(1.0 if found else 0.0, steps or MAX_STEPS)

        if found and events:
            hebbian.update(net, events)

        status = f"FOUND in {steps}" if found else f"FAILED ({MAX_STEPS} steps)"
        print(f"  Episode {ep}: {status}")

    # Evaluate
    success = 0
    total_steps = []
    for s in range(args.eval_seeds):
        rng = np.random.default_rng(1000 + s)
        steps, _, _, found = play_game(net, rng)
        if found:
            success += 1
            total_steps.append(steps)

    print(f"\nEvaluation: {success}/{args.eval_seeds} success")
    if total_steps:
        print(f"Average steps: {np.mean(total_steps):.1f}")


if __name__ == "__main__":
    main()
