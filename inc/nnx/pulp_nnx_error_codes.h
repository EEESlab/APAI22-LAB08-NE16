#ifndef __NE16_ERROR_CODES_H__
#define __NE16_ERROR_CODES_H__

typedef enum {
    success = 0,
    weightBitwidthOutOfBounds,
    unsupportedWeightOffsetMode,
    unsupportedFeatureBitwidth,
    unsupportedStride,
    dimensionMismatch
} nnx_error_code;

#endif  // __NE16_ERROR_CODES_H__