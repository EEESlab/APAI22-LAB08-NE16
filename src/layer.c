#include <pmsis.h>

#include "pulp_nnx.h"
#include "pulp_nnx_util.h"
#include "dims.h"
#include "input.h"
#include "output.h"
#include "weights.h"
#include "normalization_scale.h"
#include "layer.h"
#include "layer_util.h"

void layer(void *args) {
    layer_info();

    nnx_gvsoc_logging_activate();

    nnx_weights_t nnx_weights = {
        .height = WEIGHTS_KERNEL_HEIGHT,
        .width = WEIGHTS_KERNEL_WIDTH,
        .depth = WEIGHTS_CHANNEL_IN,
        .n_weights = WEIGHTS_CHANNEL_OUT,
        .bitwidth = 8,
        .offset_factor = 0,
        .offset_mode = weightOffsetModeLayerWise
    };

    nnx_feature_t nnx_input = {
        .height = INPUT_HEIGHT,
        .width = INPUT_WIDTH,
        .depth = INPUT_CHANNEL,
        .bitwidth = featureBitwidth8Bit
    };

    nnx_feature_t nnx_output = {
        .height = OUTPUT_HEIGHT,
        .width = OUTPUT_WIDTH,
        .depth = OUTPUT_CHANNEL,
        .bitwidth = featureBitwidth8Bit
    };

    // TODO: What should I set this to?
    const nnx_norm_t nnx_norm = {
        .mode  = normMode32Bit,
        .flag_bias  = FLAG_UNUSED,
        .flag_shift = FLAG_UNUSED
    };

    // TODO: What should I set this to?
    const nnx_quant_t nnx_quant = {
        .shift_amount = OUTSHIFT,
        .mode = quantMode8Bit,
        .function = quantFunctionRelu,
        .flag_rounding = FLAG_UNUSED
    };

    const nnx_padding_t nnx_padding = { 0 };

    const int nnx_stride = 1;

    nnx_task_t nnx_task;
    nnx_task_init(&nnx_task);

    int err;
    int is_depthwise = WEIGHTS_CHANNEL_IN == 1 && INPUT_CHANNEL != 1;
    if (WEIGHTS_KERNEL_WIDTH == 3 && !is_depthwise)
        err = nnx_conv_3x3(&nnx_task.cfg, nnx_weights, nnx_input, nnx_output, nnx_padding, nnx_stride);
    else if (WEIGHTS_KERNEL_WIDTH == 3 && is_depthwise)
        err = nnx_conv_3x3_dw(&nnx_task.cfg, nnx_weights, nnx_input, nnx_output, nnx_padding, nnx_stride);
    else if (WEIGHTS_KERNEL_WIDTH == 1 && !is_depthwise)
        err = nnx_conv_1x1(&nnx_task.cfg, nnx_weights, nnx_input, nnx_output, nnx_padding, nnx_stride);
    else {
        printf("Wrong layer arguments (ks:%d, dw:%s)\n", WEIGHTS_KERNEL_WIDTH, is_depthwise ? "true" : "false");
        pmsis_exit(-1);
    }

    nnx_norm_quant(&nnx_task.cfg, nnx_norm, nnx_quant);
    nnx_pad_input(&nnx_task.cfg, nnx_padding);

    if (err != 0) {
        printf("Error while setting up the nnx: %d\n", err);
        pmsis_exit(-2);
    }

    nnx_task.infeat_ptr = (uint32_t)input;
    nnx_task.outfeat_ptr = (uint32_t)output;
    nnx_task.weights_ptr = (uint32_t)weights;
    nnx_task.scale_ptr = (uint32_t)normalization_scale;
    nnx_task.scale_bias_ptr = (uint32_t)NULL;
    nnx_task.scale_shift_ptr = (uint32_t)NULL;

    nnx_init();
    nnx_acquire();
    nnx_offload(&nnx_task);

    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_stop();
    pi_perf_reset();
    pi_perf_start();

    nnx_run();

    const int cycles = pi_perf_read(PI_PERF_CYCLES);

    nnx_term();

    check_output();

    layer_stats(cycles);
}
