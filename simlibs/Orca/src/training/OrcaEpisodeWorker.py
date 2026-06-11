"""One-process-per-episode OMNeT++ worker for Orca training."""

import os
import sys
import traceback
from pathlib import Path


RAYNET_PATH = Path(os.getenv("RAYNET_PATH", "/home/james/raynet"))
sys.path.insert(0, str(RAYNET_PATH / "build"))


def _serialize_observations(observations):
    serialized = {}
    for agent_id, observation in observations.items():
        if hasattr(observation, "to_list"):
            serialized[agent_id] = observation.to_list()
        else:
            serialized[agent_id] = list(observation)
    return serialized


def run_episode(connection, ini_path, section_name, agent_name):
    """Run one OMNeT++ episode and exchange commands over a multiprocessing pipe."""
    from omnetbind import OmnetGymApi

    runner = OmnetGymApi()
    cleaned = False

    try:
        runner.initialise(ini_path, section_name)
        reset_obs = _serialize_observations(runner.reset())
        connection.send(("reset", reset_obs))

        while True:
            command, payload = connection.recv()

            if command == "step":
                obs, rewards, terminateds, info = runner.step(payload)
                obs = _serialize_observations(obs)
                rewards = {key: float(value) for key, value in rewards.items()}
                terminateds = {key: bool(value) for key, value in terminateds.items()}
                info = {key: bool(value) for key, value in info.items()}
                connection.send(("step", (obs, rewards, terminateds, info)))

                if terminateds.get(agent_name, False):
                    runner.shutdown()
                    runner.cleanup()
                    cleaned = True
                    break
                if info.get("simDone", False):
                    runner.cleanup()
                    cleaned = True
                    break

            elif command == "close":
                runner.shutdown()
                runner.cleanup()
                cleaned = True
                connection.send(("closed", None))
                break

            else:
                raise ValueError(f"Unknown episode-worker command: {command}")

    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except BaseException:
            pass
        raise
    finally:
        if not cleaned:
            try:
                runner.shutdown()
            except BaseException:
                pass
            try:
                runner.cleanup()
            except BaseException:
                pass
        connection.close()
