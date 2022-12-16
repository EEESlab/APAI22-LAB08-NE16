# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
# 
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import torch
import torch.nn.functional as F
from Ne16 import *

def license(filename):
    return \
"""/*
 * {filename}
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

""".format(filename=filename)

def header_guard_begin(filename):
    guard = filename.replace('.', '_')
    return \
"""#ifndef __{GUARD}__
#define __{GUARD}__

""".format(GUARD=guard.upper())

def header_guard_end(filename):
    guard = filename.replace('.', '_')
    return "#endif  // __{GUARD}__\n".format(GUARD=guard.upper())

def includes():
    return "#include <pmsis.h>\n\n"

def define(name, expr):
    if isinstance(expr, int):
        expr = f'({expr})'
    return f'#define {name.upper()} {expr}\n'

def vector_size(data):
    if hasattr(data, 'numel'):
        return data.numel()
    elif hasattr(data, 'size'):
        return data.size
    else:
        return len(data)

def vector_declaration(name, size):
    retval = ""
    retval += define(f'{name}_size', size)
    retval += f"PI_L1 uint8_t {name}[{name.upper()}_SIZE]"
    return retval

def vector_initial_value(data, elements_per_row=10, spaces=4):
    indent = ' ' * spaces
    size = vector_size(data)

    if hasattr(data, 'flatten'):
        data = data.flatten()

    retval = ""
    retval += " = {"
    for i, element in enumerate(data):
        if i % elements_per_row == 0:
            retval += '\n' + indent
        retval += '{value:#04x}'.format(value=int(element))
        if i < size - 1:
            retval += ', '
    retval += '\n}'
    return retval

def vector_end():
    return ';\n\n'

def render_vector(name, init=None, size=None, elements_per_row=10, spaces=4):
    size_ = vector_size(init) if init is not None else size
    retval = ""
    retval += vector_declaration(name, size_)
    if init is not None:
        retval += vector_initial_value(init, elements_per_row, spaces)
    retval += vector_end()
    return retval

def check(name):
    return \
f"""static void check_{name}() {{
    printf("Checking the {name} vector:\\n");

    int n_err = 0;
    for (int i = 0; i < {name.upper()}_SIZE; i++) {{
        if ({name}[i] != golden_{name}[i]) {{
            printf("ERROR: wrong value of {name} @ %d: %d vs. golden: %d\\n", i, {name}[i], golden_{name}[i]);
            n_err++;
        }}
    }}

    if (n_err == 0)
        printf("> Success! No errors found.\\n\\n");
    else
        printf("> Failure! Found %d/%d errors.\\n\\n", n_err, {name.upper()}_SIZE);
}}

"""

def generate_header(name, path, body):
    filename = name + '.h'
    filepath = os.path.join('inc', path, filename)

    print(f'Generating header file -> {filepath}')

    filerender = license(filename)              \
                 + header_guard_begin(filename) \
                 + body                         \
                 + header_guard_end(filename)

    with open(filepath, 'w') as file:
        file.write(filerender)

def generate_vector_header(name, data, golden=None):
    bodyrender = ""
    bodyrender += includes()
    bodyrender += render_vector(name, init=data, size=vector_size(golden) if golden is not None else None)

    if golden is not None:
        bodyrender += render_vector('golden_' + name, init=golden)
        bodyrender += check(name)
        
    generate_header(name, 'data', bodyrender)

def render_dims(name, dims):
    retval = ""
    for dim_name, dim_value in zip(dims["names"], dims["shape"]):
        retval += define(f'{name}_{dim_name}', dim_value)
    return retval

def render_dummy(name, data):
    return ""

def generate_dims_header(name, info):
    bodyrender = ""
    for piece_of_info in info:
        if piece_of_info["type"] == "dims":
            render_func = render_dims
        elif piece_of_info["type"] == "def":
            render_func = define
        else:
            print(f"Render function not implemented for type {type}")
            render_func = render_dummy
        bodyrender += render_func(piece_of_info["name"], piece_of_info["data"])
        bodyrender += '\n'

    generate_header(name, 'data', bodyrender)

def borders(bits, signed = False):
    low = -(2 ** (bits-1)) if signed else 0
    high = 2 ** (bits-1) - 1 if signed else 2 ** bits - 1
    return low, high

def clip(x, bits, signed=False):
    low, high = borders(bits, signed)
    x[x > high] = high
    x[x < low] = low
    return x

def create_input(channels, spatial_dim):
    size = (1, channels, spatial_dim, spatial_dim)
    return torch.randint(low=0, high=100, size=size, dtype=torch.int32)

def create_weights(shape):
    """ Create weights

    Shape is of layout (Cout, H, W, Cin)
    """
    size = (shape[0], shape[3], shape[1], shape[2])  # Torch expects layout (Cout, Cin, H, W)
    return torch.randint(low=0, high=5, size=size, dtype=torch.int32)

def create_layer(cin, cout, spatial_dim, kernel_shape, outshift=8):
    x = create_input(cin, spatial_dim + kernel_shape - 1)
    x_save = x.permute(0, 2, 3, 1).type(torch.int32)
    generate_vector_header("input", x_save)

    w = create_weights((cout, kernel_shape, kernel_shape, cin))
    w_save = Ne16().conv_unroll(w.numpy(), 8, layout="CoutCinK", dw=False)
    generate_vector_header("weights", w_save)

    #norm_scale = torch.ones((1, channels, 1, 1), dtype=torch.int32)
    norm_scale = np.ones((1, cout, 1, 1), dtype='<i4')
    generate_vector_header("normalization_scale", norm_scale.tobytes())
    
    y = F.conv2d(x, w).type(torch.int32)
    y = torch.from_numpy(norm_scale) * y
    y = clip(y >> outshift, 8)
    y_save = y.permute(0, 2, 3, 1).type(torch.int32)
    generate_vector_header("output", None, golden=y_save)

    generate_dims_header('dims',
                         [
                             {"type":"dims", "name": "input",    "data": {"shape": x_save.shape[1:], "names": ["height", "width", "channel"]}},
                             {"type":"dims", "name": "output",   "data": {"shape": y_save.shape[1:], "names": ["height", "width", "channel"]}},
                             {"type":"dims", "name": "weights",  "data": {"shape": w.shape,          "names": ["channel_out", "channel_in", "kernel_height", "kernel_width"]}},
                             {"type":"def",  "name": "outshift", "data": outshift}
                         ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-shape', '-ks', dest='kernel_shape', type=int, choices=[1, 3], default=1,
                        help='Shape of the kernel. Choices: 1 or 3. Default: 1')
    parser.add_argument('--channels-in', '-cin', dest='cin', type=int, default=16,
                        help='Number of input channels. Default: 16')
    parser.add_argument('--channels-out', '-cout', dest='cout', type=int, default=32,
                        help='Number of output channels. Default: 32')
    parser.add_argument('--output-spatial-dimensions', '-osd', dest='spatial_dimensions', type=int, default=3,
                        help='Output spatial dimension. Default 3')
    args = parser.parse_args()

    # All the generated headers will go into 'inc/data' so create directory first
    os.makedirs('inc/data', exist_ok=True)

    create_layer(args.cin, args.cout, args.spatial_dimensions, args.kernel_shape)
