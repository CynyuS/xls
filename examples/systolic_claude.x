// 2x2 Systolic Array for Matrix Multiplication
// Computes C = A * B where A is MxK, B is KxN, and C is MxN
// This implementation uses M=2, N=2, K=2

import float32;

type F32 = float32::F32;
const F32_ZERO = float32::zero(false);

// Processing Element (PE) for the systolic array
// Each PE performs: accumulate += a * b
// and forwards a (right) and b (down) to neighbors
proc PE {
    a_in: chan<F32> in;
    b_in: chan<F32> in;
    a_out: chan<F32> out;
    b_out: chan<F32> out;
    result_out: chan<F32> out;

    config(a_in: chan<F32> in, b_in: chan<F32> in,
           a_out: chan<F32> out, b_out: chan<F32> out,
           result_out: chan<F32> out) {
        (a_in, b_in, a_out, b_out, result_out)
    }

    init { F32_ZERO }

    next(state: F32) {
        let tok = join();
        let (tok_a, a_val) = recv(tok, a_in);
        let (tok_b, b_val) = recv(tok, b_in);
        
        // Compute: accumulate = accumulate + a * b
        let product = float32::mul(a_val, b_val);
        let new_state = float32::add(state, product);
        
        // Forward values to neighbors
        let tok = join(tok_a, tok_b);
        let tok = send(tok, a_out, a_val);
        let tok = send(tok, b_out, b_val);
        let tok = send(tok, result_out, new_state);
        
        new_state
    }
}

// A feeder proc that sends matrix A rows to the systolic array
proc AFeeder {
    a_out_0: chan<F32> out;
    a_out_1: chan<F32> out;

    config(a_out_0: chan<F32> out, a_out_1: chan<F32> out) {
        (a_out_0, a_out_1)
    }

    init { u32:0 }

    next(state: u32) {
        // For a 2x2 matrix multiplication with K=2
        // We need to feed each row K times
        // Row 0: A[0,0], A[0,1]
        // Row 1: A[1,0], A[1,1]
        
        let tok = join();
        
        // This is a simplified version - in practice, you'd read from input channels
        // For now, sending zeros as placeholder
        let tok = send(tok, a_out_0, F32_ZERO);
        let tok = send(tok, a_out_1, F32_ZERO);
        
        state + u32:1
    }
}

// B feeder proc that sends matrix B columns to the systolic array
proc BFeeder {
    b_out_0: chan<F32> out;
    b_out_1: chan<F32> out;

    config(b_out_0: chan<F32> out, b_out_1: chan<F32> out) {
        (b_out_0, b_out_1)
    }

    init { u32:0 }

    next(state: u32) {
        let tok = join();
        
        // Feed columns of B
        let tok = send(tok, b_out_0, F32_ZERO);
        let tok = send(tok, b_out_1, F32_ZERO);
        
        state + u32:1
    }
}

// Drain proc that consumes forwarded values at array boundaries
proc Drain {
    drain_in: chan<F32> in;

    config(drain_in: chan<F32> in) {
        (drain_in,)
    }

    init { () }

    next(state: ()) {
        let tok = join();
        let (tok, _) = recv(tok, drain_in);
    }
}

// Top-level systolic array with 2x2 PEs
proc SystolicArray2x2 {
    // Input channels for A matrix (2 rows)
    a_in_0: chan<F32> in;
    a_in_1: chan<F32> in;
    
    // Input channels for B matrix (2 columns)  
    b_in_0: chan<F32> in;
    b_in_1: chan<F32> in;
    
    // Output channels for C matrix results (2x2 = 4 values)
    c_out_00: chan<F32> out;
    c_out_01: chan<F32> out;
    c_out_10: chan<F32> out;
    c_out_11: chan<F32> out;

    config(a_in_0: chan<F32> in, a_in_1: chan<F32> in,
           b_in_0: chan<F32> in, b_in_1: chan<F32> in,
           c_out_00: chan<F32> out, c_out_01: chan<F32> out,
           c_out_10: chan<F32> out, c_out_11: chan<F32> out) {
        
        // Internal channels for connecting PEs
        // Horizontal channels (A values flow left to right)
        let (a_00_01_p, a_00_01_c) = chan<F32>("a_00_to_01");
        let (a_10_11_p, a_10_11_c) = chan<F32>("a_10_to_11");
        
        // Vertical channels (B values flow top to bottom)
        let (b_00_10_p, b_00_10_c) = chan<F32>("b_00_to_10");
        let (b_01_11_p, b_01_11_c) = chan<F32>("b_01_to_11");
        
        // Drain channels for boundary outputs
        let (a_01_drain_p, a_01_drain_c) = chan<F32>("a_01_drain");
        let (a_11_drain_p, a_11_drain_c) = chan<F32>("a_11_drain");
        let (b_10_drain_p, b_10_drain_c) = chan<F32>("b_10_drain");
        let (b_11_drain_p, b_11_drain_c) = chan<F32>("b_11_drain");
        
        // Spawn PE[0,0] (top-left)
        spawn PE(a_in_0, b_in_0, a_00_01_p, b_00_10_p, c_out_00);
        
        // Spawn PE[0,1] (top-right)
        spawn PE(a_00_01_c, b_in_1, a_01_drain_p, b_01_11_p, c_out_01);
        
        // Spawn PE[1,0] (bottom-left)
        spawn PE(a_in_1, b_00_10_c, a_10_11_p, b_10_drain_p, c_out_10);
        
        // Spawn PE[1,1] (bottom-right)
        spawn PE(a_10_11_c, b_01_11_c, a_11_drain_p, b_11_drain_p, c_out_11);
        
        // Spawn drain procs for boundary outputs
        spawn Drain(a_01_drain_c);
        spawn Drain(a_11_drain_c);
        spawn Drain(b_10_drain_c);
        spawn Drain(b_11_drain_c);
        
        (a_in_0, a_in_1, b_in_0, b_in_1,
         c_out_00, c_out_01, c_out_10, c_out_11)
    }

    init { () }

    next(state: ()) {
        // Top-level proc just coordinates - PEs do the work
        ()
    }
}

// Test proc for the systolic array with general matrix multiplication
// Tests: A = [[1, 2], [3, 4]] * B = [[5, 6], [7, 8]]
// Expected C = [[19, 22], [43, 50]]
#[test_proc]
proc systolic_array_test {
    a_in_0_s: chan<F32> out;
    a_in_1_s: chan<F32> out;
    b_in_0_s: chan<F32> out;
    b_in_1_s: chan<F32> out;
    c_out_00_r: chan<F32> in;
    c_out_01_r: chan<F32> in;
    c_out_10_r: chan<F32> in;
    c_out_11_r: chan<F32> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (a_in_0_s, a_in_0_r) = chan<F32>("a_in_0");
        let (a_in_1_s, a_in_1_r) = chan<F32>("a_in_1");
        let (b_in_0_s, b_in_0_r) = chan<F32>("b_in_0");
        let (b_in_1_s, b_in_1_r) = chan<F32>("b_in_1");
        let (c_out_00_s, c_out_00_r) = chan<F32>("c_out_00");
        let (c_out_01_s, c_out_01_r) = chan<F32>("c_out_01");
        let (c_out_10_s, c_out_10_r) = chan<F32>("c_out_10");
        let (c_out_11_s, c_out_11_r) = chan<F32>("c_out_11");
        
        spawn SystolicArray2x2(a_in_0_r, a_in_1_r, b_in_0_r, b_in_1_r,
                               c_out_00_s, c_out_01_s, c_out_10_s, c_out_11_s);
        
        (a_in_0_s, a_in_1_s, b_in_0_s, b_in_1_s,
         c_out_00_r, c_out_01_r, c_out_10_r, c_out_11_r, terminator)
    }

    next(state: ()) {
        let tok = join();
        
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        
        let F32_ONE = float32::one(false);
        let F32_TWO = float32::add(F32_ONE, F32_ONE);
        let F32_THREE = float32::add(F32_TWO, F32_ONE);
        let F32_FOUR = float32::add(F32_TWO, F32_TWO);
        let F32_FIVE = float32::add(F32_FOUR, F32_ONE);
        let F32_SIX = float32::add(F32_FIVE, F32_ONE);
        let F32_SEVEN = float32::add(F32_SIX, F32_ONE);
        let F32_EIGHT = float32::add(F32_SEVEN, F32_ONE);
        
        // First iteration (K=0): multiply first column of A with first row of B
        // A[:,0] = [1, 3], B[0,:] = [5, 6]
        let tok = send(tok, a_in_0_s, F32_ONE);     // A[0,0] = 1
        let tok = send(tok, a_in_1_s, F32_THREE);   // A[1,0] = 3
        let tok = send(tok, b_in_0_s, F32_FIVE);    // B[0,0] = 5
        let tok = send(tok, b_in_1_s, F32_SIX);     // B[0,1] = 6
        
        // Receive partial results: C[i,j] += A[i,0] * B[0,j]
        let (tok, c00_partial) = recv(tok, c_out_00_r);  // Should be 1*5 = 5
        let (tok, c01_partial) = recv(tok, c_out_01_r);  // Should be 1*6 = 6
        let (tok, c10_partial) = recv(tok, c_out_10_r);  // Should be 3*5 = 15
        let (tok, c11_partial) = recv(tok, c_out_11_r);  // Should be 3*6 = 18
        
        // Second iteration (K=1): multiply second column of A with second row of B
        // A[:,1] = [2, 4], B[1,:] = [7, 8]
        let tok = send(tok, a_in_0_s, F32_TWO);     // A[0,1] = 2
        let tok = send(tok, a_in_1_s, F32_FOUR);    // A[1,1] = 4
        let tok = send(tok, b_in_0_s, F32_SEVEN);   // B[1,0] = 7
        let tok = send(tok, b_in_1_s, F32_EIGHT);   // B[1,1] = 8
        
        // Receive final results: previous + A[i,1] * B[1,j]
        let (tok, c00_final) = recv(tok, c_out_00_r);  // Should be 5 + 2*7 = 19
        let (tok, c01_final) = recv(tok, c_out_01_r);  // Should be 6 + 2*8 = 22
        let (tok, c10_final) = recv(tok, c_out_10_r);  // Should be 15 + 4*7 = 43
        let (tok, c11_final) = recv(tok, c_out_11_r);  // Should be 18 + 4*8 = 50
        
        // Create expected values
        let F32_19 = float32::add(
            float32::mul(F32_ONE, F32_FIVE),
            float32::mul(F32_TWO, F32_SEVEN)
        );
        let F32_22 = float32::add(
            float32::mul(F32_ONE, F32_SIX),
            float32::mul(F32_TWO, F32_EIGHT)
        );
        let F32_43 = float32::add(
            float32::mul(F32_THREE, F32_FIVE),
            float32::mul(F32_FOUR, F32_SEVEN)
        );
        let F32_50 = float32::add(
            float32::mul(F32_THREE, F32_SIX),
            float32::mul(F32_FOUR, F32_EIGHT)
        );
        
        // Verify final results
        assert_eq(c00_final, F32_19);
        assert_eq(c01_final, F32_22);
        assert_eq(c10_final, F32_43);
        assert_eq(c11_final, F32_50);
        
        let tok = send(tok, terminator, true);
    }
}

// Simple smoke test with zeros
#[test_proc]
proc smoke_test {
    a_in_0_s: chan<F32> out;
    a_in_1_s: chan<F32> out;
    b_in_0_s: chan<F32> out;
    b_in_1_s: chan<F32> out;
    c_out_00_r: chan<F32> in;
    c_out_01_r: chan<F32> in;
    c_out_10_r: chan<F32> in;
    c_out_11_r: chan<F32> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (a_in_0_s, a_in_0_r) = chan<F32>("a_in_0");
        let (a_in_1_s, a_in_1_r) = chan<F32>("a_in_1");
        let (b_in_0_s, b_in_0_r) = chan<F32>("b_in_0");
        let (b_in_1_s, b_in_1_r) = chan<F32>("b_in_1");
        let (c_out_00_s, c_out_00_r) = chan<F32>("c_out_00");
        let (c_out_01_s, c_out_01_r) = chan<F32>("c_out_01");
        let (c_out_10_s, c_out_10_r) = chan<F32>("c_out_10");
        let (c_out_11_s, c_out_11_r) = chan<F32>("c_out_11");
        
        spawn SystolicArray2x2(a_in_0_r, a_in_1_r, b_in_0_r, b_in_1_r,
                               c_out_00_s, c_out_01_s, c_out_10_s, c_out_11_s);
        
        (a_in_0_s, a_in_1_s, b_in_0_s, b_in_1_s,
         c_out_00_r, c_out_01_r, c_out_10_r, c_out_11_r, terminator)
    }

    next(state: ()) {
        let tok = join();
        
        // Test with all zeros - should produce all zero outputs
        // First iteration (K=0)
        let tok = send(tok, a_in_0_s, F32_ZERO);
        let tok = send(tok, a_in_1_s, F32_ZERO);
        let tok = send(tok, b_in_0_s, F32_ZERO);
        let tok = send(tok, b_in_1_s, F32_ZERO);
        
        let (tok, c00) = recv(tok, c_out_00_r);
        let (tok, c01) = recv(tok, c_out_01_r);
        let (tok, c10) = recv(tok, c_out_10_r);
        let (tok, c11) = recv(tok, c_out_11_r);
        
        assert_eq(c00, F32_ZERO);
        assert_eq(c01, F32_ZERO);
        assert_eq(c10, F32_ZERO);
        assert_eq(c11, F32_ZERO);
        
        // Second iteration (K=1)
        let tok = send(tok, a_in_0_s, F32_ZERO);
        let tok = send(tok, a_in_1_s, F32_ZERO);
        let tok = send(tok, b_in_0_s, F32_ZERO);
        let tok = send(tok, b_in_1_s, F32_ZERO);
        
        let (tok, c00) = recv(tok, c_out_00_r);
        let (tok, c01) = recv(tok, c_out_01_r);
        let (tok, c10) = recv(tok, c_out_10_r);
        let (tok, c11) = recv(tok, c_out_11_r);
        
        assert_eq(c00, F32_ZERO);
        assert_eq(c01, F32_ZERO);
        assert_eq(c10, F32_ZERO);
        assert_eq(c11, F32_ZERO);
        
        let tok = send(tok, terminator, true);
    }
}