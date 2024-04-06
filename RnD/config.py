import yaml

with open('RnD/config.yaml') as f:
    default_config  = yaml.safe_load(f)

print(default_config)







