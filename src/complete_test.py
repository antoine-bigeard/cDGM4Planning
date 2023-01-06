import sys

sys.path.insert(0, "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS")

from src.utils import read_yaml_config_file, update

from src.main import main
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        default="configs_tests/update_conf_for_tests.yaml",
        required=False,
    )

    args = parser.parse_args()
    config = read_yaml_config_file(args.path_config)

    path_logs = config.get("path_logs")
    update_conf = config.get("update_conf")

    config_run = read_yaml_config_file(os.path.join(path_logs, "config.yaml"))
    config_run["checkpoint_path"] = os.path.join(
        path_logs,
        "checkpoints",
        os.listdir(os.path.join(path_logs, "checkpoints"))[0],
    )

    if update_conf is not None:
        update(config_run, update_conf)

    main(config_run)
    # if args.parallel:
    #     pool = multiprocessing.Pool()
    #     outs = pool.map(main, args.path_config)
    # else:
    #     for path in args.path_config:
    #         main(path)
