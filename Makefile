# Makefile
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

# Set runner_args="--trace=ne16" if you want to trace what is happening in the ne16

CORE ?= 1

APP = main
APP_SRCS := $(wildcard src/*.c)
APP_CFLAGS += -DNUM_CORES=$(CORE) -Iinc -Iinc/data -Iinc/nnx -O2 -w

include $(RULES_DIR)/pmsis_rules.mk
