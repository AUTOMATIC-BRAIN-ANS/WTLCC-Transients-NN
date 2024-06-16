from experiment_management import run_experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", action="store_true", help="Dry run")
parser.add_argument("-t", action="store_true", help="Test")

args = parser.parse_args()


if __name__ == '__main__':
    experiments = ["BOTHCNN"]
    dry_run = False
    debug_mode = False
    if args.d:
        dry_run = True
    if args.t:
        debug_mode = True
        
    project = "CorrMatToOutcome"

    for experiment in experiments:
        if not debug_mode:
            try:
                run_experiment(experiment, project, mode="online", dry_run=dry_run)
            except Exception as e:
                print(f"failed to run {experiment}, error: {e}")
        else:
            run_experiment(experiment, project, mode="disabled", dry_run=dry_run)
    