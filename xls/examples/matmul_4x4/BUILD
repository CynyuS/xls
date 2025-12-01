# Copyright 2024 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "matmul_4x4_dslx",
    srcs = ["matmul_4x4.x"],
)

xls_dslx_test(
    name = "matmul_4x4_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":matmul_4x4_dslx",
)

xls_dslx_opt_ir(
    name = "matmul_4x4_opt_ir",
    dslx_top = "matmul_4x4",
    ir_conv_args = {
        # Set which proc is the top entity
        # "top": "matmul_4x4",
        # Set default FIFO config for internal streaming channels
        "default_fifo_config": "depth: 1, bypass: true, register_push_outputs: false, register_pop_outputs: false",
        # Don't use proc scoped channels - we want package-level channels
        "proc_scoped_channels": "false",
    },
    library = ":matmul_4x4_dslx",
)

# Combinational version targeting a single PE proc
xls_ir_verilog(
    name = "matmul_4x4_pe_comb",
    src = ":matmul_4x4_opt_ir.opt.ir",
    codegen_args = {
        "generator": "combinational",
        "use_system_verilog": "true",
        "module_name": "matmul_pe",
        "top": "__matmul_4x4__matmul_4x4__matmul_0__4_4_next",
    },
    verilog_file = "matmul_4x4_pe_comb.sv",
)

# Combinational version with top proc specified
xls_ir_verilog(
    name = "matmul_4x4_comb",
    src = ":matmul_4x4_opt_ir.opt.ir",
    codegen_args = {
        "generator": "combinational",
        "use_system_verilog": "true",
        "module_name": "matmul_4x4_comb",
        "top": "__matmul_4x4__matmul_4x4_0_next",
        "multi_proc": "true",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_valid_suffix": "_valid",
        "streaming_channel_ready_suffix": "_ready",
    },
    verilog_file = "matmul_4x4_comb.sv",
)

# Pipeline version for proc-based streaming design
xls_ir_verilog(
    name = "matmul_4x4_pipeline",
    src = ":matmul_4x4_opt_ir.opt.ir",
    codegen_args = {
        "generator": "pipeline",
        "pipeline_stages": "2",
        "reset": "rst",
        "reset_data_path": "false",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "use_system_verilog": "true",
        "module_name": "matmul_4x4",
        "top": "__matmul_4x4__matmul_4x4_0_next",
        "multi_proc": "true",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_valid_suffix": "_valid",
        "streaming_channel_ready_suffix": "_ready",
        "fifo_module": "",
    },
    verilog_file = "matmul_4x4_pipeline.sv",
)

# Convenience target to build all versions
filegroup(
    name = "all_verilog",
    srcs = [
        # ":matmul_4x4_comb",
        ":matmul_4x4_pipeline",
    ],
)