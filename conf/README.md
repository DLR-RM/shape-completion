# Config files

Place for all config files. The `config.yaml` contains default values for all parameters and is inherited by all other config files.
Inside `dirs/default.yaml` the paths to the datasets and the log directory have to be set. When working in multiple environments,
a separate `.yaml` file can be created for each environment, e.g. `dirs/work.yaml` and `dirs/local.yaml`.
