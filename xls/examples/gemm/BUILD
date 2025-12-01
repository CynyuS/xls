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
    licenses = ["notice"],
)

xls_dslx_library(
    name = "gemm_dslx",
    srcs = ["gemm.x"],
)

xls_dslx_test(
    name = "gemm_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":gemm_dslx",
)

xls_dslx_opt_ir(
    name = "gemm_opt_ir",
    dslx_top = "gemm",
    library = ":gemm_dslx",
)

# Combinational version
xls_ir_verilog(
    name = "gemm_comb",
    src = ":gemm_opt_ir.opt.ir",
    codegen_args = {
        "generator": "combinational",
        "use_system_verilog": "true",
        "module_name": "gemm_comb",
    },
    verilog_file = "gemm_comb.sv",
)

# Pipeline version with minimal stages
xls_ir_verilog(
    name = "gemm_pipeline",
    src = ":gemm_opt_ir.opt.ir",
    codegen_args = {
        "generator": "pipeline",
        "pipeline_stages": "2",
        "delay_model": "unit",
        "reset": "rst",
        "reset_data_path": "false",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "use_system_verilog": "true",
        "module_name": "gemm_pipeline",
    },
    verilog_file = "gemm_pipeline.sv",
)

# Pipeline version with more stages for higher frequency
xls_ir_verilog(
    name = "gemm_pipeline_deep",
    src = ":gemm_opt_ir.opt.ir",
    codegen_args = {
        "generator": "pipeline",
        "pipeline_stages": "5",
        "delay_model": "asap7",
        "reset": "rst",
        "reset_data_path": "false",
        "flop_inputs": "true",
        "flop_outputs": "true",
        "use_system_verilog": "true",
        "module_name": "gemm_deep",
    },
    verilog_file = "gemm_pipeline_deep.sv",
)

# Generate signature file for simulation
genrule(
    name = "gemm_signature",
    srcs = [":gemm_opt_ir.opt.ir"],
    outs = ["gemm_signature.textproto"],
    cmd = "$(location //xls/tools:codegen_main) " +
          "--generator=combinational " +
          "--output_signature_path=$@ " +
          "--use_system_verilog=true " +
          "$(location :gemm_opt_ir.opt.ir) > /dev/null",
    tools = ["//xls/tools:codegen_main"],
)

filegroup(
    name = "all_verilog",
    srcs = [
        ":gemm_comb",
        ":gemm_pipeline",
        ":gemm_pipeline_deep",
    ],
)
