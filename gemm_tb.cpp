#include "Vgemm_pipeline.h"
#include "verilated.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vgemm_pipeline* dut = new Vgemm_pipeline;
    
    // Seed random number generator
    srand(time(NULL));
    
    // Create random 32x32 matrices A and B, and compute expected result C
    uint32_t A[32][32], B[32][32], C_expected[32][32];
    
    // Initialize matrices with random values (using small values to avoid overflow)
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            A[i][j] = rand() % 10;  // Random values 0-9
            B[i][j] = rand() % 10;  // Random values 0-9
            C_expected[i][j] = 0;
        }
    }
    
    // Compute expected result: C = A * B
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                C_expected[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    // Initialize DUT inputs
    dut->clk = 0;
    dut->rst = 1;
    
    // Load matrices into DUT (flattened arrays)
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            dut->arg0[i * 32 + j] = A[i][j];
            dut->arg1[i * 32 + j] = B[i][j];
        }
    }
    
    // Run simulation
    bool test_passed = true;
    for (int i = 0; i < 30; i++) {
        dut->clk = 0;
        dut->eval();
        
        dut->clk = 1;
        dut->eval();
        
        if (i == 2) dut->rst = 0; // Deassert reset after 2 cycles
        
        // Check results after pipeline latency
        if (i >= 10) {
            // Verify all elements
            for (int row = 0; row < 32; row++) {
                for (int col = 0; col < 32; col++) {
                    uint32_t result = dut->out[row * 32 + col];
                    if (result != C_expected[row][col]) {
                        std::cout << "MISMATCH at C[" << row << "][" << col << "]: ";
                        std::cout << "Expected " << C_expected[row][col];
                        std::cout << ", Got " << result << std::endl;
                        test_passed = false;
                    }
                }
            }
            
            if (test_passed) {
                std::cout << "Cycle " << i << ": All elements match! TEST PASSED" << std::endl;
            }
            break;  // Only check once after pipeline is done
        }
    }
    
    // Print sample values for verification
    std::cout << "\nSample values:" << std::endl;
    std::cout << "A[0][0]=" << A[0][0] << ", B[0][0]=" << B[0][0] 
              << ", C_expected[0][0]=" << C_expected[0][0] 
              << ", C_actual[0][0]=" << dut->out[0] << std::endl;
    std::cout << "A[15][15]=" << A[15][15] << ", B[15][15]=" << B[15][15] 
              << ", C_expected[15][15]=" << C_expected[15][15] 
              << ", C_actual[15][15]=" << dut->out[15*32+15] << std::endl;
    
    delete dut;
    return test_passed ? 0 : 1;
}