# [APAI22-LAB08] Using the NE16 accelerator

## `parameters_generate.py` script

To see which arguments are available in the script, just run the below command:

```
python parameters_generate.py --help
```

## Producing simulation logs

To produce simulation logs run the command:

```
make clean all run runner_args="--trace=ne16"
```

If you want to save it to a file, e.g. `ne16.log`, run the command:

```
make clean all run runner_args="--trace=ne16" > ne16.log
```
