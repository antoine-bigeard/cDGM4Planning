import sys

sys.path.insert(0, ".")

from src.utils import read_yaml_config_file, update

from src.main import main
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        default="configs_tests/update_conf_for_tests_250_small.yaml",
        required=False,
    )

    args = parser.parse_args()
    config = read_yaml_config_file(args.path_config)

    path_logs = config.get("path_logs")

    update_conf = config.get("update_conf")

    if os.path.exists(os.path.join(path_logs, "config.yaml")):
        config_run = read_yaml_config_file(os.path.join(path_logs, "config.yaml"))
    else:
        name_config = [
            f for f in os.listdir(os.path.join(path_logs)) if f[-5:] == ".yaml"
        ][0]
        config_run = read_yaml_config_file(os.path.join(path_logs, name_config))

    config_run["tensorboard_logs"]["save_dir"] = os.path.join(
        *path_logs.split("/")[:-2]
    )
    config_run["name_experiment"] = path_logs.split("/")[-2]

    config_run["checkpoint_path"] = os.path.join(
        path_logs,
        "checkpoints",
        os.listdir(os.path.join(path_logs, "checkpoints"))[0],
    )

    if update_conf is not None:
        update(config_run, update_conf)

    main(config_run)
