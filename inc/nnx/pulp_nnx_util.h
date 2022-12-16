#ifndef __PULP_NNX_UTIL__
#define __PULP_NNX_UTIL__

#include "pulp_nnx_hal.h"

void nnx_gvsoc_logging_activate() {
  NE16_WRITE_IO_REG(sizeof(nnx_task_t), 3);
  NE16_WRITE_IO_REG(sizeof(nnx_task_t)+4, 0); // or 3
}

void nnx_gvsoc_logging_deactivate() {
  NE16_WRITE_IO_REG(sizeof(nnx_task_t), 0);
}

#endif /* __PULP_NNX_UTIL__ */
