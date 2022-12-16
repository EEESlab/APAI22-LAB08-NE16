/*
 * pulp_nnx_hal.c
 * Luka Macan <luka.macan@fer.hr>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2022 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nnx_hal.h"

static int qw, weight_d0_stride, outbytes, stride_shift;

// TODO For all the following functions we use __builtin_pulp_OffsetedWrite and
// __builtin_pulp_OffsetedRead instead of classic load/store because otherwise
// the compiler is not able to correctly factorize the NE16 base in case several
// accesses are done, ending up with twice more code

// __builtin_pulp_OffsetedX not defined - needs further investigation... (too
// old PULP toolchain? used v1.0.16) It is used inside PULP-SDK...

void nnx_init() {
  NE16_CG_ENABLE();
  NE16_SETPRIORITY_NE16();
  NE16_SET_MAXSTALL(0);
  nnx_soft_clear();
}

void nnx_term() {
  nnx_soft_clear();
  NE16_SETPRIORITY_CORE();
  NE16_RESET_MAXSTALL();
  NE16_CG_DISABLE();
}

int nnx_empty() {
  return !NE16_READ(NE16_STATUS);
}

int nnx_full() {
  return NE16_READ(NE16_STATUS) == NE16_STATUS_FULL;
}

uint8_t nnx_job_id() {
  return NE16_READ(NE16_RUNNING_JOB);
}

void nnx_soft_clear() {
  NE16_WRITE(NE16_SOFT_CLEAR, 0);
  for (volatile int i = 0; i < 10; i++)
    ;
}

int nnx_acquire_polled() {
  int job_id = -1;
  while ((job_id = NE16_READ(NE16_ACQUIRE)) < 0) printf("job id: %d\n", job_id);
  return job_id;
}

int nnx_acquire() {
  int job_id = -1;
  NE16_BARRIER_ACQUIRE(job_id);
  return job_id;
}

void nnx_offload(nnx_task_t *task) {
  int *task_data = (int *)task;
  for (int i = 0; i < sizeof(nnx_task_t) / 4; ++i) {
    NE16_WRITE_IO_REG(i * 4, task_data[i]);
  }
}

void nnx_offload_ptr(nnx_task_t *task) {
  int *task_data = (int *)task;
  for (int i = 0; i < 6; ++i) {
    NE16_WRITE_IO_REG(i * 4, task_data[i]);
  }
}

void nnx_run_async() {
  NE16_WRITE(NE16_TRIGGER, 0);
}

void nnx_run() {
  nnx_run_async();
  nnx_wait_empty();
}

void nnx_commit() {
  NE16_WRITE(NE16_TRIGGER, 1); // commit, no trigger
}

void nnx_busywait() {
  NE16_BUSYWAIT();
}

void nnx_wait_empty() {
  while(!nnx_empty()) NE16_BARRIER_NOSTATUS();
}

void nnx_wait_not_full() {
  while(nnx_full()) NE16_BARRIER_NOSTATUS();
}

void nnx_wait_on_id(const uint8_t id) {
  while(nnx_job_id() <= id) {
    eu_evt_maskWaitAndClr (1 << NE16_EVT0);
  };
}

void nnx_task_init(nnx_task_t *task) {
  memset(task, 0, sizeof(nnx_task_t));
}

int nnx_pad_input(nnx_cfg_t *cfg, const nnx_padding_t padding) {
  cfg->padding = (padding.top << 28) | (padding.right << 24) | (padding.bottom << 20) | (padding.left << 16) | padding.value;
  return 0;
}

int nnx_norm_quant(nnx_cfg_t *cfg, const nnx_norm_t norm,  const nnx_quant_t quant) {
  if (quant.shift_amount > 31) {
    printf("ERROR! quant.shift_amount > 31\n");
    return 1;
  }

  if (quant.mode == quantMode16Bit) {
    printf("ERROR! quant.mode == quantMode16Bit\n");
    return 1;
  }

  BIT_SET(cfg->conf0,
          NE16_FLAG_NORM_QUANT
          | quant.function | quant.mode | (quant.shift_amount << 16) | quant.flag_rounding << NE16_SHIFT_ROUNDING
          | norm.mode | norm.flag_bias << NE16_SHIFT_FLAG_NORM_BIAS | norm.flag_shift << NE16_SHIFT_FLAG_NORM_SHIFT);

  return 0;
}

void nnx_mask_filter(nnx_cfg_t *cfg, const uint8_t top, const uint8_t right,
            const uint8_t bottom, const uint8_t left) {
  cfg->filter_mask = ((uint32_t)top << 24) | ((uint32_t)right << 16) |
              ((uint32_t)bottom << 8) | ((uint32_t)left << 0);
}

nnx_error_code nnx_conv_1x1_update_dims(nnx_cfg_t *cfg,
    const int h_out, const int w_out, const int w_in, const int k_out, const int k_in,
    const int w_in_stride, const int w_out_stride, const nnx_padding_t padding) {

  const int num_Ko = DIVNCEIL(k_out, NE16_OUTPUT_CHANNEL_THROUGHPUT);
  const int num_Ki = DIVNCEIL(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);
  const int num_Ho = DIVNCEIL(h_out, NE16_FILTER_SIZE);
  const int num_Wo = DIVNCEIL(w_out, NE16_FILTER_SIZE);
  
  const int rem_Ko = REMAINDER(k_out, NE16_OUTPUT_CHANNEL_THROUGHPUT);
  const int rem_Ki = REMAINDER(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);
  const int rem_Ho = REMAINDER(h_out, NE16_FILTER_SIZE);
  const int rem_Wo = REMAINDER(w_out, NE16_FILTER_SIZE);
  const int rem_Hi = rem_Ho - padding.bottom;
  const int rem_Wi = rem_Wo - padding.right;

  const nnx_subtile_t subtile = {
    .number = {
      .KoKi = CONCAT_HALF(num_Ko, num_Ki),
      .HoWo = CONCAT_HALF(num_Ho, num_Wo)
    },
    .remainder = {
      .KoKi = CONCAT_HALF(rem_Ko, rem_Ki),
      .HoWo = CONCAT_HALF(rem_Ho, rem_Wo),
      .HiWi = CONCAT_HALF(rem_Hi, rem_Wi)
    }
  };
  cfg->subtile = subtile;

  // Strides
  const nnx_stride_t input_stride = {
    .d0 = k_in,
    .d1 = k_in * w_in_stride,
    .d2 = k_in * NE16_FILTER_BUFFER_SIZE * NE16_FILTER_BUFFER_SIZE
  };
  cfg->input_stride = input_stride;

  const nnx_stride_t output_stride = {
    .d0 = 32,
    .d1 = (k_out * outbytes) >> stride_shift,
    .d2 = (k_out * outbytes * w_out) >> stride_shift
  };
  cfg->output_stride = output_stride;

  const nnx_stride_t weights_stride = {
    .d0 = weight_d0_stride * qw,
    .d1 = weight_d0_stride * qw * num_Ki,
    .d2 = 0 // Unused
  };
  cfg->weights_stride = weights_stride;

  return 0;
}

nnx_error_code nnx_conv_1x1(nnx_cfg_t *cfg,
                const nnx_weights_t weights,
                const nnx_feature_t input,
                const nnx_feature_t output,
                const nnx_padding_t padding,
                const stride) {
  if (weights.bitwidth < 2 || weights.bitwidth > 8) {
    return weightBitwidthOutOfBounds;
  }

  if (weights.offset_mode != weightOffsetModeLayerWise) {
    // Currently only layer-wise mode is used.
    return unsupportedWeightOffsetMode;
  }

  if ((input.bitwidth != featureBitwidth8Bit &&
     input.bitwidth != featureBitwidth16Bit) ||
    (output.bitwidth != featureBitwidth8Bit &&
     output.bitwidth != featureBitwidth32Bit)) {
    return unsupportedFeatureBitwidth;
  }

  if (stride != 1 && stride != 2) {
    return unsupportedStride;
  }

  const int mode16 =
    input.bitwidth == 16 ? NE16_FLAG_MODE16 : NE16_FLAG_MODE_BASIC;

  BIT_SET(cfg->conf0, weights.offset_mode | NE16_FLAG_MODE_1x1 | mode16 |
                 (weights.bitwidth - 1));

  // Global static variables needed by update_dims
  outbytes = output.bitwidth / 8;
  weight_d0_stride =
    mode16 ? NE16_WEIGHT_D0_STRIDE_MODE16 : NE16_WEIGHT_D0_STRIDE_MODE8;
  qw = weights.bitwidth;

  nnx_conv_1x1_update_dims(cfg, output.height, output.width, input.width, output.depth, input.depth, input.width, output.width, padding);

  cfg->weight_offset_factor = weights.offset_factor;

  return 0;
}

nnx_error_code nnx_conv_3x3_update_dims(nnx_cfg_t *cfg,
    const int h_out, const int w_out, const int w_in, const int k_out, const int k_in,
    const int w_in_stride, const int w_out_stride, const nnx_padding_t padding) {

  const int num_Ko = DIVNCEIL(k_out, NE16_OUTPUT_CHANNEL_THROUGHPUT);
  const int num_Ki = DIVNCEIL(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);
  const int num_Ho = DIVNCEIL(h_out, NE16_FILTER_SIZE);
  const int num_Wo = DIVNCEIL(w_out, NE16_FILTER_SIZE);

  const int rem_Ko = REMAINDER(k_out, NE16_OUTPUT_CHANNEL_THROUGHPUT);
  const int rem_Ki = REMAINDER(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);
  const int rem_Ho = REMAINDER(h_out, NE16_FILTER_SIZE);
  const int rem_Wo = REMAINDER(w_out, NE16_FILTER_SIZE);
  const int rem_Hi = rem_Ho + 2 - padding.bottom;
  const int rem_Wi = rem_Wo + 2 - padding.right;

  const nnx_subtile_t subtile = {
    .number = {
      .KoKi = CONCAT_HALF(num_Ko, num_Ki),
      .HoWo = CONCAT_HALF(num_Ho, num_Wo)
    },
    .remainder = {
      .KoKi = CONCAT_HALF(rem_Ko, rem_Ki),
      .HoWo = CONCAT_HALF(rem_Ho, rem_Wo),
      .HiWi = CONCAT_HALF(rem_Hi, rem_Wi)
    }
  };
  cfg->subtile = subtile;

  // Strides
  const nnx_stride_t input_stride = {
    .d0 = k_in,
    .d1 = k_in * w_in_stride,
    .d2 = k_in * NE16_FILTER_BUFFER_SIZE * NE16_FILTER_BUFFER_SIZE
  };
  cfg->input_stride = input_stride;

  const nnx_stride_t output_stride = {
    .d0 = 32,
    .d1 = (k_out * outbytes) >> stride_shift,
    .d2 = (k_out * outbytes * w_out_stride) >> stride_shift
  };
  cfg->output_stride = output_stride;

  const nnx_stride_t weights_stride = {
    .d0 = NE16_FILTER_SIZE * NE16_FILTER_SIZE * weight_d0_stride,
    .d1 = NE16_FILTER_SIZE * NE16_FILTER_SIZE * weight_d0_stride * qw * num_Ki,
    .d2 = 0  // Unused
  };
  cfg->weights_stride = weights_stride;

  return 0;
}

nnx_error_code nnx_conv_3x3(nnx_cfg_t *cfg,
                const nnx_weights_t weights,
                const nnx_feature_t input,
                const nnx_feature_t output,
                const nnx_padding_t padding,
                const stride) {
  if (weights.bitwidth < 2 || weights.bitwidth > 8) {
    return weightBitwidthOutOfBounds;
  }

  if (weights.offset_mode != weightOffsetModeLayerWise) {
    // Currently only layer-wise mode is used.
    return unsupportedWeightOffsetMode;
  }

  if ((input.bitwidth != featureBitwidth8Bit &&
     input.bitwidth != featureBitwidth16Bit) ||
    (output.bitwidth != featureBitwidth8Bit &&
     output.bitwidth != featureBitwidth32Bit)) {
    return unsupportedFeatureBitwidth;
  }

  if (stride != 1 && stride != 2) {
    return unsupportedStride;
  }

  const int mode16 =
    input.bitwidth == 16 ? NE16_FLAG_MODE16 : NE16_FLAG_MODE_BASIC;

  const flag_stride2x2 = stride == 2 ? NE16_FLAG_STRIDE_2x2 : 0;

  BIT_SET(cfg->conf0, weights.offset_mode | NE16_FLAG_MODE_3x3 | mode16 |
                 (weights.bitwidth - 1) | flag_stride2x2);

  // Global static variables needed by update_dims
  outbytes = output.bitwidth / 8;
  weight_d0_stride =
    mode16 ? NE16_WEIGHT_D0_STRIDE_MODE16 : NE16_WEIGHT_D0_STRIDE_MODE8;
  qw = weights.bitwidth;
  stride_shift = stride == 2 ? 1 : 0;

  nnx_conv_3x3_update_dims(cfg, output.height, output.width, input.width, output.depth, input.depth, input.width, output.width, padding);
  
  cfg->weight_offset_factor = weights.offset_factor;

  return 0;
}

nnx_error_code nnx_conv_3x3_dw_update_dims(nnx_cfg_t *cfg,
    const int h_out, const int w_out, const int w_in, const int k_out, const int k_in,
    const int w_in_stride, const int w_out_stride, const nnx_padding_t padding) {

  const int num_Ko = DIVNCEIL(k_out, NE16_INPUT_CHANNEL_THROUGHPUT);
  const int num_Ki = num_Ko;
  const int num_Ho = DIVNCEIL(h_out, NE16_FILTER_SIZE);
  const int num_Wo = DIVNCEIL(w_out, NE16_FILTER_SIZE);

  const int rem_Ko = REMAINDER(k_out, NE16_INPUT_CHANNEL_THROUGHPUT);
  const int rem_Ki = rem_Ko;
  const int rem_Ho = REMAINDER(h_out, NE16_FILTER_SIZE);
  const int rem_Wo = REMAINDER(w_out, NE16_FILTER_SIZE);
  const int rem_Hi = rem_Ho + 2 - padding.bottom;
  const int rem_Wi = rem_Wo + 2 - padding.right;

  const nnx_subtile_t subtile = {
    .number = {
      .KoKi = CONCAT_HALF(num_Ko, num_Ki),
      .HoWo = CONCAT_HALF(num_Ho, num_Wo)
    },
    .remainder = {
      .KoKi = CONCAT_HALF(rem_Ko, rem_Ki),
      .HoWo = CONCAT_HALF(rem_Ho, rem_Wo),
      .HiWi = CONCAT_HALF(rem_Hi, rem_Wi)
    }
  };
  cfg->subtile = subtile;

  // Strides
  const nnx_stride_t input_stride = {
    .d0 = k_out,
    .d1 = k_out * w_in_stride,
    .d2 = 0 // Unused
  };
  cfg->input_stride = input_stride;

  const nnx_stride_t output_stride = {
    .d0 = 32,
    .d1 = (k_out * outbytes) >> stride_shift,
    .d2 = (k_out * outbytes * w_out_stride) >> stride_shift
  };
  cfg->output_stride = output_stride;

  const nnx_stride_t weights_stride = {
    .d0 = NE16_FILTER_SIZE * NE16_FILTER_SIZE * weight_d0_stride,
    .d1 = 0,
    .d2 = 0  // Unused
  };
  cfg->weights_stride = weights_stride;

  return 0;
}

nnx_error_code nnx_conv_3x3_dw(nnx_cfg_t *cfg,
                const nnx_weights_t weights,
                const nnx_feature_t input,
                const nnx_feature_t output,
                const nnx_padding_t padding,
                const stride) {
  if (weights.bitwidth < 2 || weights.bitwidth > 8) {
    return weightBitwidthOutOfBounds;
  }

  if (weights.offset_mode != weightOffsetModeLayerWise) {
    // Currently only layer-wise mode is used.
    return unsupportedWeightOffsetMode;
  }

  if ((input.bitwidth != featureBitwidth8Bit &&
     input.bitwidth != featureBitwidth16Bit) ||
    (output.bitwidth != featureBitwidth8Bit &&
     output.bitwidth != featureBitwidth32Bit)) {
    return unsupportedFeatureBitwidth;
  }

  if (stride != 1 && stride != 2) {
    return unsupportedStride;
  }

  const int mode16 =
    input.bitwidth == 16 ? NE16_FLAG_MODE16 : NE16_FLAG_MODE_BASIC;

  BIT_SET(cfg->conf0, weights.offset_mode | NE16_FLAG_MODE_3x3_DW | mode16 |
                 (weights.bitwidth - 1));

  // Global static variables needed by update_dims
  outbytes = output.bitwidth / 8;
  weight_d0_stride =
    mode16 ? NE16_WEIGHT_D0_STRIDE_MODE16 : NE16_WEIGHT_D0_STRIDE_MODE8;
  qw = weights.bitwidth;
  stride_shift = stride == 2 ? 1 : 0;

  nnx_conv_3x3_dw_update_dims(cfg, output.height, output.width, input.width, output.depth, input.depth, input.width, output.width, padding);
  
  cfg->weight_offset_factor = weights.offset_factor;

  return 0;
}
