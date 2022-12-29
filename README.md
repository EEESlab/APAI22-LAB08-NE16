# [APAI22-LAB08] Using the NE16 accelerator

## `parameters_generate.py` script

To see which arguments are available in the script, just run the below command:

```
python parameters_generate.py --help
```

Arguments
``` 
  --kernel-shape {1,3}, -ks {1,3}
                        Shape of the kernel. Choices: 1 or 3. Default: 1
  --channels-in CIN, -cin CIN
                        Number of input channels. Default: 16
  --channels-out COUT, -cout COUT
                        Number of output channels. Default: 32
  --output-spatial-dimensions SPATIAL_DIMENSIONS, -osd SPATIAL_DIMENSIONS
                        Output spatial dimension. Default 3
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


## using the bash script for state counting

In the proceeding tasks, you will have to measure the number of states executed for different layers. To ease the counting, we created a script `count_states.sh`.
Quick usage guide:

If you want to use it on a file, call the script like this:

```
./count_states.sh ne16.log
```

Or you can pipe the output of the simulation directly to the script:

```
make all run runner_args=”--trace=ne16” | ./count_states.sh
```



