import yaml
import logging.config


def config(fname):
    with open(fname) as f:
        config_dict = yaml.safe_load(f)
    logging.config.dictConfig(config_dict)
