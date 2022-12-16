#ifndef __LAYER_UTIL_H__
#define __LAYER_UTIL_H__

#include "dims.h"

static void layer_info() {
    printf("Layer info:\n"
           " - input: (%dx%dx%d)\n"
           " - output: (%dx%dx%d)\n"
           " - weights: (%dx%dx%dx%d)\n\n",
           INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL,
           OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL,
           WEIGHTS_CHANNEL_OUT, WEIGHTS_KERNEL_HEIGHT, WEIGHTS_KERNEL_WIDTH, WEIGHTS_CHANNEL_IN);
}

static void layer_stats(const int latency) {
    const int mac_ops = OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNEL
        * WEIGHTS_KERNEL_HEIGHT * WEIGHTS_KERNEL_WIDTH * WEIGHTS_CHANNEL_IN;

    const float perf = (float)mac_ops / (float)latency;

    printf("Layer statistics:\n"
           " - operations: %d MAC\n"
           " - latency: %d cycles\n"
           " - performance: %.2f MAC/cycle\n\n",
           mac_ops, latency, perf);
}

#endif  // __LAYER_UTIL_H__
