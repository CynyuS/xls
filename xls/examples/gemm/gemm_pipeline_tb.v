`timescale 1ns / 1ps

module gemm_pipeline_tb;
  reg clk;
  reg rst;
  reg [32767:0] arg0;  // Matrix A (32x32 of 32-bit values)
  reg [32767:0] arg1;  // Matrix B (32x32 of 32-bit values)
  wire [32767:0] out;  // Matrix C (32x32 of 32-bit values)
  
  // Instantiate the DUT
  gemm_pipeline dut (
    .clk(clk),
    .rst(rst),
    .arg0(arg0),
    .arg1(arg1),
    .out(out)
  );
  
  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100MHz clock
  end
  
  // Test vectors - compute [[1,1],[1,1]] * [[2,2],[2,2]] = [[4,4],[4,4]]
  initial begin
    $dumpfile("gemm_pipeline_tb.vcd");
    $dumpvars(0, gemm_pipeline_tb);
    
    // Initialize all matrix elements to 0
    arg0 = 0;
    arg1 = 0;
    
    // Set up simple 2x2 test in top-left corner: A = [[1,1],[1,1]], B = [[2,2],[2,2]]
    // arg0[31:0]     = A[0][0] = 1
    // arg0[63:32]    = A[0][1] = 1
    // arg0[1055:1024] = A[1][0] = 1
    // arg0[1087:1056] = A[1][1] = 1
    arg0[31:0]      = 32'd1;
    arg0[63:32]     = 32'd1;
    arg0[1055:1024] = 32'd1;
    arg0[1087:1056] = 32'd1;
    
    // arg1[31:0]     = B[0][0] = 2
    // arg1[63:32]    = B[0][1] = 2
    // arg1[1055:1024] = B[1][0] = 2
    // arg1[1087:1056] = B[1][1] = 2
    arg1[31:0]      = 32'd2;
    arg1[63:32]     = 32'd2;
    arg1[1055:1024] = 32'd2;
    arg1[1087:1056] = 32'd2;
    
    // Reset
    rst = 1;
    #20;
    rst = 0;
    #20;
    
    // Wait for pipeline latency (3 cycles) + some extra cycles
    repeat(10) @(posedge clk);
    
    // Check results: C[0][0], C[0][1], C[1][0], C[1][1] should all be 4
    $display("Result Matrix (first 2x2):");
    $display("C[0][0] = %d (expected 4)", out[31:0]);
    $display("C[0][1] = %d (expected 4)", out[63:32]);
    $display("C[1][0] = %d (expected 4)", out[1055:1024]);
    $display("C[1][1] = %d (expected 4)", out[1087:1056]);
    
    if (out[31:0] == 32'd4 && out[63:32] == 32'd4 && 
        out[1055:1024] == 32'd4 && out[1087:1056] == 32'd4) begin
      $display("TEST PASSED!");
    end else begin
      $display("TEST FAILED!");
    end
    
    #100;
    $finish;
  end
  
endmodule
