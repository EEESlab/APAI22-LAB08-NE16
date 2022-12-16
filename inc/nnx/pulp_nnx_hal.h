/*
 * pulp_nnx_hal.h
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

// IDEA: instead of calling separate functions, user
// initializes structures he wants and sends pointers to the init_struct
// function. If the pointer is NULL, the user doesn't want that feature (e.g.
// padding). All the error checking can be done in the init function. Current
// functions can be reused to make the init function smaller.

#ifndef __NE16_H__
#define __NE16_H__

#include <stdint.h>
#include "pmsis.h"

#include "pulp_nnx_defs.h"
#include "pulp_nnx_error_codes.h"

#define BIT_SET(var, bits) var |= bits

#define NE16_WRITE(offset, value) \
  *(int volatile *)(NE16_BASE_ADDR + (offset)) = (value)
#define NE16_WRITE_BE(offset, value, be) \
  *(char volatile *)(NE16_BASE_ADDR + (offset) + (be)) = (value)
#define NE16_READ(offset) *(int volatile *)(NE16_BASE_ADDR + (offset))

#define NE16_WRITE_IO_REG(offset, value) \
  NE16_WRITE(NE16_REGISTER_OFFSET + (offset), (value))
#define NE16_WRITE_IO_REG_BE(offset, value, be) \
  NE16_WRITE_BE(NE16_REGISTER_OFFSET + (offset), (value), (be))
#define NE16_READ_IO_REG(offset) NE16_READ(NE16_REGISTER_OFFSET + (offset))

#define NE16_BARRIER_NOSTATUS()      eu_evt_maskWaitAndClr (1 << NE16_EVT0)
#define NE16_BARRIER()               do { eu_evt_maskWaitAndClr (1 << NE16_EVT0); } while((*(int volatile *)(NE16_BASE_ADDR + NE16_STATUS)) != 0)
#define NE16_BUSYWAIT()              do {                                         } while((*(int volatile *)(NE16_BASE_ADDR + NE16_STATUS)) != 0)
#define NE16_BARRIER_ACQUIRE(job_id) job_id = NE16_READ(NE16_ACQUIRE); \
                                     while(job_id < 0) { eu_evt_maskWaitAndClr (1 << NE16_EVT0); job_id = NE16_READ(NE16_ACQUIRE); };
#define NE16_NOBARRIER_ACQUIRE(job_id) job_id = NE16_READ(NE16_ACQUIRE); \
                                       while(job_id < 0) { job_id = NE16_READ(NE16_ACQUIRE); };

/* CLUSTER */
#define NE16_CG_ENABLE()  *(volatile int*) (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS) |=  CLUSTER_CTRL_HWPE_CG_EN_MASK
#define NE16_CG_DISABLE() *(volatile int*) (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS) &= ~CLUSTER_CTRL_HWPE_CG_EN_MASK

#define NE16_SETPRIORITY_CORE() *(volatile int*) (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS) &= ~CLUSTER_CTRL_HWPE_HCI_PRIO_MASK
#define NE16_SETPRIORITY_NE16() *(volatile int*) (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS) |=  CLUSTER_CTRL_HWPE_HCI_PRIO_MASK

#define NE16_RESET_MAXSTALL()  *(volatile int*) (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS) &= ~CLUSTER_CTRL_HWPE_HCI_MAXSTALL_MASK
#define NE16_SET_MAXSTALL(val) *(volatile int*) (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS) |=  (val & CLUSTER_CTRL_HWPE_HCI_MAXSTALL_MASK)


#define DIVNCEIL(A,B)  ( (((A) - 1) / (B)) + 1 )
#define REMAINDER(A,B) ( (((A) - 1) % (B)) + 1 )
#define CONCAT_HALF(A,B) ( (((A) & 0xffff) << 16) | ((B) & 0xffff) )

#define NNX_CONTEXT_SIZE NE16_CONTEXT_SIZE

#define FLAG_USED   (1)
#define FLAG_UNUSED (0)

typedef enum {
    weightOffsetModeSymmetric = NE16_FLAG_WEIGHT_OFFSET_SYMMETRIC,
    weightOffsetModeLayerWise = NE16_FLAG_WEIGHT_OFFSET_LAYER_WISE
} nnx_weight_offset_mode_e;

typedef struct {
    void *data;
    uint16_t height;
    uint16_t width;
    uint16_t depth;
    uint16_t n_weights;
    uint32_t bitwidth;
    int32_t offset_factor;
    nnx_weight_offset_mode_e offset_mode;
} nnx_weights_t;

typedef enum {
    featureBitwidth8Bit = 8,
    featureBitwidth16Bit = 16,
    featureBitwidth32Bit = 32
} nnx_feature_bitwidth_e;

typedef struct {
    void *data;
    uint16_t height;
    uint16_t width;
    uint16_t depth;
    nnx_feature_bitwidth_e bitwidth;
} nnx_feature_t;

typedef enum {
    normMode8Bit = NE16_NORM_MODE_8BIT,
    normMode16Bit = NE16_NORM_MODE_16BIT,
    normMode32Bit = NE16_NORM_MODE_32BIT
} nnx_norm_mode_e;

typedef struct {
    nnx_norm_mode_e mode;
    int flag_bias;
    int flag_shift;
} nnx_norm_t;

typedef enum {
    quantMode8Bit = NE16_QUANT_MODE_8BIT,
    quantMode16Bit = NE16_QUANT_MODE_16BIT,
    quantMode32Bit = NE16_QUANT_MODE_32BIT
} nnx_quant_mode_e;

typedef enum {
    quantFunctionIdentity = NE16_FLAG_QUANT_FUNCTION_IDENTITY,
    quantFunctionRelu = NE16_FLAG_QUANT_FUNCTION_RELU
} nnx_quant_function_e;

// TODO: add rounding to quant. Should also be an enum? Best boolean...
typedef struct {
    // Shift amount must be in range 0x00-0x1F
    unsigned shift_amount;
    nnx_quant_mode_e mode;
    nnx_quant_function_e function;
    int flag_rounding;
} nnx_quant_t;

typedef struct {
    int top;
    int right;
    int bottom;
    int left;
    uint16_t value;
} nnx_padding_t;

typedef struct {
    uint32_t d0;
    uint32_t d1;
    uint32_t d2;
} nnx_stride_t;

typedef struct {
    uint32_t KoKi;
    uint32_t HoWo;
    uint32_t HiWi;
} nnx_subtile_remainder_t;

typedef struct {
    uint32_t KoKi;
    uint32_t HoWo;
} nnx_subtile_number_t;

typedef struct {
    nnx_subtile_remainder_t remainder;
    nnx_subtile_number_t number;
} nnx_subtile_t;

typedef struct {
    nnx_stride_t input_stride;
    nnx_stride_t output_stride;
    nnx_stride_t weights_stride;
    nnx_subtile_t subtile;
    uint32_t padding;
    uint32_t weight_offset_factor;
    uint32_t filter_mask;
    uint32_t conf0;
} nnx_cfg_t;

typedef struct {
    uint32_t weights_ptr;
    uint32_t infeat_ptr;
    uint32_t outfeat_ptr;
    uint32_t scale_ptr;
    uint32_t scale_shift_ptr;
    uint32_t scale_bias_ptr;
    nnx_cfg_t cfg;
} nnx_task_t;

uint8_t nnx_job_id();
int  nnx_empty();
int  nnx_full();
void nnx_soft_clear();
int  nnx_acquire();
void nnx_offload(nnx_task_t *task);
void nnx_offload_ptr(nnx_task_t *task);
void nnx_run_async();
void nnx_run();
void nnx_commit();
void nnx_wait_empty();
void nnx_wait_not_full();
void nnx_wait_on_id(const uint8_t id);
void nnx_busywait();
void nnx_init();
void nnx_term();

void nnx_task_init(nnx_task_t *task);
int nnx_pad_input(nnx_cfg_t *cfg, nnx_padding_t padding);
int nnx_norm_quant(nnx_cfg_t *cfg, nnx_norm_t norm, nnx_quant_t quant);
void nnx_mask_filter(nnx_cfg_t *cfg, uint8_t top, uint8_t right, uint8_t bottom, uint8_t left);
nnx_error_code nnx_conv_1x1(nnx_cfg_t *cfg, nnx_weights_t weights, nnx_feature_t input, nnx_feature_t output, nnx_padding_t padding, const stride);
nnx_error_code nnx_conv_1x1_update_dims(nnx_cfg_t *cfg, int h_out, int w_out, int w_in, int k_out, int k_in, int w_in_stride, int w_out_stride, nnx_padding_t padding);
nnx_error_code nnx_conv_3x3(nnx_cfg_t *cfg, nnx_weights_t weights, nnx_feature_t input, nnx_feature_t output, nnx_padding_t padding, const stride);
nnx_error_code nnx_conv_3x3_update_dims(nnx_cfg_t *cfg, int h_out, int w_out, int w_in, int k_out, int k_in, int w_in_stride, int w_out_stride, nnx_padding_t padding);
nnx_error_code nnx_conv_3x3_dw(nnx_cfg_t *cfg, nnx_weights_t weights, nnx_feature_t input, nnx_feature_t output, nnx_padding_t padding, const stride);
nnx_error_code nnx_conv_3x3_dw_update_dims(nnx_cfg_t *cfg, int h_out, int w_out, int w_in, int k_out, int k_in, int w_in_stride, int w_out_stride, nnx_padding_t padding);

#endif /* __NE16_H__ */
