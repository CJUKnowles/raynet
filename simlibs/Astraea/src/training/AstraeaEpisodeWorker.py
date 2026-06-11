"""One-process-per-episode OMNeT++ worker for Astraea training."""

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


def run_episode(connection, ini_path, section_name):
    """Run one multi-agent OMNeT++ episode over a multiprocessing pipe."""
    from omnetbind import OmnetGymApi

    runner = OmnetGymApi()
    cleaned = False

    try:
        runner.initialise(ini_path, section_name)
        connection.send(("reset", _serialize_observations(runner.reset())))

        while True:
            command, payload = connection.recv()

            if command == "step":
                obs, rewards, terminateds, info = runner.step(payload)
                result = (
                    _serialize_observations(obs),
                    {key: float(value) for key, value in rewards.items()},
                    {key: bool(value) for key, value in terminateds.items()},
                    {key: bool(value) for key, value in info.items()},
                )
                connection.send(("step", result))

                if terminateds.get("__all__", False):
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
