import nni

from nni_trial_moderntcn_gecco import run_trial


if __name__ == "__main__":
    params = nni.get_next_parameter()
    run_trial(params)
