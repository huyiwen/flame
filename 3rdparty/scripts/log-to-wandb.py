import wandb
import datetime
from pprint import pprint
from copy import deepcopy
import os
import re
from typing import List, Dict
import json
from collections import defaultdict


def parse_config(linedata: List[str]) -> dict:
    # find line with 'Running with args: defaultdict'
    for i, line in enumerate(linedata):
        if "Running with args: defaultdict" in line:
            break
    else:
        raise ValueError("Could not find config line in log")

    # extract config
    cli_config = re.search(r"\(<class 'collections.defaultdict'>, (.*)\)", linedata[i]).group(1)
    cli_config = eval(cli_config, {"defaultdict": defaultdict})

    # find 'Building model from the config'
    for i, line in enumerate(linedata):
        if "Building model from the config" in line:
            break
    else:
        raise ValueError("Could not find model config line in log")

    # extract model config from the following lines, until the next line with '[titan]'
    st_line = i + 2
    for i, line in enumerate(linedata[st_line:], start=st_line):
        if "[titan]" in line:
            break
    else:
        raise ValueError("Could not find end of model config in log")

    ed_line = i
    while linedata[ed_line].strip() != "}":
        ed_line -= 1

    model_config = "{\n" + "\n".join(linedata[st_line:ed_line]) + "\n}"
    model_config = model_config.strip()
    pprint(model_config)
    model_config = json.loads(model_config)

    # merge the two configs
    for k, v in model_config.items():
        cli_config[f"model_config.{k}"] = v

    return cli_config


def parse_steps(linedata: List[str]) -> List[Dict]:

    data = []
    line_number = 1

    key_mapping = {
        "step": "step",
        "loss": "optim/global_avg_loss",
        "token": None,  # drop
        "lr": "optim/learning_rate",
        "gnorm": "optim/grad_norm",
        "memory": "memory/max_reserved(GiB)",
        "mfu": "speed/mfu(%)",
        "time_end_to_end": "time/end_to_end(s)"
    }
    timestamp_last = datetime.timedelta()
    for _, line in enumerate(linedata):
        line_str = f"{line_number:,}"
        if "step:" in line and "INFO" in line and line_str in line:

            # delete color codes
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)

            # find data region
            line = line.split("INFO -")[1].strip()

            # extract data with 'xxx: 1  yyy: 2' format
            state_dict = {}
            key = None
            end_to_end_time = False
            for item in re.split(r"\s+", line):

                # parse '[ 0:03:25<22 days, 19:33:51]'
                if item == "[":
                    end_to_end_time = True
                    continue

                if end_to_end_time or ("[" in item and "<" in item):
                    cur = item.replace("[", "").split("<")[0]
                    cur = datetime.timedelta(hours=int(cur.split(":")[0]), minutes=int(cur.split(":")[1]), seconds=int(cur.split(":")[2]))
                    state_dict["time_end_to_end"] = (cur - timestamp_last).total_seconds()
                    timestamp_last = cur
                    break

                if ":" in item:
                    assert item[-1] == ":", f"`{item}`"
                    assert key is None, f"key is not None: {key}"
                    key = item[:-1]
                else:
                    assert key is not None, f"key is None: key={key} item={item}"
                    state_dict[key] = item
                    key = None

            new_state_dict = {}
            for key, value in state_dict.items():
                if key in key_mapping:
                    new_key = key_mapping[key]
                    if new_key is not None:
                        new_state_dict[new_key] = value
                else:
                    raise ValueError(f"Unknown key: {key}")
            for key in key_mapping:
                if key not in state_dict:
                    raise ValueError(f"Missing key: {key} {state_dict}")

            new_state_dict["step"] = int(new_state_dict["step"].replace(",", ""))
            new_state_dict["memory/max_reserved(GiB)"] = float(new_state_dict["memory/max_reserved(GiB)"].replace("GiB", ""))
            new_state_dict["speed/mfu(%)"] = float(new_state_dict["speed/mfu(%)"].replace("%", ""))
            new_state_dict["optim/grad_norm"] = float(new_state_dict["optim/grad_norm"])
            new_state_dict["optim/learning_rate"] = float(new_state_dict["optim/learning_rate"])
            new_state_dict["optim/global_avg_loss"] = float(new_state_dict["optim/global_avg_loss"])
            data.append(new_state_dict)

            line_number += 1
    return data


def main(log_dir: str, project_name: str, log_name="train-0.log"):

    with open(os.path.join(log_dir, log_name), "r") as f:
        log = f.read()

    model, config_name, job_id = log_dir.split("/")[-3:]
    wandb_name = f"{model}.{config_name}"
    wandb_id = f"{wandb_name}-" + job_id.split("-")[1]

    linedata = log.split("\n")
    config = parse_config(linedata)
    steps = parse_steps(linedata)

    print(wandb_name, wandb_id)
    wandb.init(project=project_name, dir=log_dir, config=config, name=wandb_name, id=wandb_id)
    for step in steps:
        step_data = deepcopy(step)
        step = int(step_data.pop("step"))
        wandb.log(step_data, step)
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--project", type=str, default="fla")
    args = parser.parse_args()
    main(args.log, args.project)