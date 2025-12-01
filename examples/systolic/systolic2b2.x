// Complete 2x2 Systolic Array with Functional Tests
// Self-contained file for testing matrix multiplication on a 2x2 PE grid
//
// Architecture: 2x2 grid of Processing Elements (PEs)
// Computes C = A @ B where:
//   A: 2x4 matrix (2 rows, K=4 columns)
//   B: 4x2 matrix (K=4 rows, 2 columns)
//   C: 2x2 matrix (2 rows, 2 columns)

// ============================================================================
// PE - Processing Element with MAC (Multiply-Accumulate)
// ============================================================================

proc PE<K: u32> {
  a_in: chan<u32> in;
  b_in: chan<u32> in;
  a_out: chan<u32> out;
  b_out: chan<u32> out;
  c_out: chan<u32> out;

  config(a_in: chan<u32> in, b_in: chan<u32> in, a_out: chan<u32> out, b_out: chan<u32> out, c_out: chan<u32> out) {
    (a_in, b_in, a_out, b_out, c_out)
  }

  init { (u32:0, u32:0) }

  next(state: (u32, u32)) {
    let (accum, k) = state;
    let (tok, a) = recv(join(), a_in);
    let (tok, b) = recv(tok, b_in);
    let prod = a * b;
    let new_accum = accum + prod;
    let tok = send(tok, a_out, a);
    let tok = send(tok, b_out, b);
    let new_k = k + u32:1;
    let tok = if new_k == K { send(tok, c_out, new_accum) } else { tok };
    let new_state = if new_k == K { (u32:0, u32:0) } else { (new_accum, new_k) };
    new_state
  }
}

// ============================================================================
// SystolicArray_2x2_u32 - Top-level 2x2 systolic array
// ============================================================================

pub proc SystolicArray_2x2_u32 {
  a_row_0: chan<u32> in;
  a_row_1: chan<u32> in;
  b_col_0: chan<u32> in;
  b_col_1: chan<u32> in;
  c_result: chan<u32[2][2]> out;

  config(a_row_0: chan<u32> in, a_row_1: chan<u32> in, b_col_0: chan<u32> in, b_col_1: chan<u32> in, c_result: chan<u32[2][2]> out) {
    (a_row_0, a_row_1, b_col_0, b_col_1, c_result)
  }

  init { () }

  next(state: ()) {
    // Internal horizontal channels (A flows left-to-right)
    let (a_fifo_0_0_s, a_fifo_0_0_r) = chan<u32>("a_fifo_0_0");
    let (a_fifo_0_1_s, a_fifo_0_1_r) = chan<u32>("a_fifo_0_1");
    let (a_fifo_0_2_s, a_fifo_0_2_r) = chan<u32>("a_fifo_0_2");
    let (a_fifo_1_0_s, a_fifo_1_0_r) = chan<u32>("a_fifo_1_0");
    let (a_fifo_1_1_s, a_fifo_1_1_r) = chan<u32>("a_fifo_1_1");
    let (a_fifo_1_2_s, a_fifo_1_2_r) = chan<u32>("a_fifo_1_2");

    // Internal vertical channels (B flows top-to-bottom)
    let (b_fifo_0_0_s, b_fifo_0_0_r) = chan<u32>("b_fifo_0_0");
    let (b_fifo_0_1_s, b_fifo_0_1_r) = chan<u32>("b_fifo_0_1");
    let (b_fifo_0_2_s, b_fifo_0_2_r) = chan<u32>("b_fifo_0_2");
    let (b_fifo_1_0_s, b_fifo_1_0_r) = chan<u32>("b_fifo_1_0");
    let (b_fifo_1_1_s, b_fifo_1_1_r) = chan<u32>("b_fifo_1_1");
    let (b_fifo_1_2_s, b_fifo_1_2_r) = chan<u32>("b_fifo_1_2");

    // Output channels from each PE
    let (c_out_0_0_s, c_out_0_0_r) = chan<u32>("c_out_0_0");
    let (c_out_0_1_s, c_out_0_1_r) = chan<u32>("c_out_0_1");
    let (c_out_1_0_s, c_out_1_0_r) = chan<u32>("c_out_1_0");
    let (c_out_1_1_s, c_out_1_1_r) = chan<u32>("c_out_1_1");

    // Spawn 2x2 grid of PEs (K=4)
    spawn PE<u32:4>(a_fifo_0_0_r, b_fifo_0_0_r, a_fifo_0_1_s, b_fifo_0_1_s, c_out_0_0_s);
    spawn PE<u32:4>(a_fifo_0_1_r, b_fifo_1_0_r, a_fifo_0_2_s, b_fifo_1_1_s, c_out_0_1_s);
    spawn PE<u32:4>(a_fifo_1_0_r, b_fifo_0_1_r, a_fifo_1_1_s, b_fifo_0_2_s, c_out_1_0_s);
    spawn PE<u32:4>(a_fifo_1_1_r, b_fifo_1_1_r, a_fifo_1_2_s, b_fifo_1_2_s, c_out_1_1_s);

    let tok = join();

    // Feed A matrix (2 rows x 4 elements each)
    let (tok, a_0_0) = recv(tok, a_row_0);
    let tok = send(tok, a_fifo_0_0_s, a_0_0);
    let (tok, a_0_1) = recv(tok, a_row_0);
    let tok = send(tok, a_fifo_0_0_s, a_0_1);
    let (tok, a_0_2) = recv(tok, a_row_0);
    let tok = send(tok, a_fifo_0_0_s, a_0_2);
    let (tok, a_0_3) = recv(tok, a_row_0);
    let tok = send(tok, a_fifo_0_0_s, a_0_3);
    let (tok, a_1_0) = recv(tok, a_row_1);
    let tok = send(tok, a_fifo_1_0_s, a_1_0);
    let (tok, a_1_1) = recv(tok, a_row_1);
    let tok = send(tok, a_fifo_1_0_s, a_1_1);
    let (tok, a_1_2) = recv(tok, a_row_1);
    let tok = send(tok, a_fifo_1_0_s, a_1_2);
    let (tok, a_1_3) = recv(tok, a_row_1);
    let tok = send(tok, a_fifo_1_0_s, a_1_3);

    // Feed B matrix (2 columns x 4 elements each)
    let (tok, b_0_0) = recv(tok, b_col_0);
    let tok = send(tok, b_fifo_0_0_s, b_0_0);
    let (tok, b_1_0) = recv(tok, b_col_0);
    let tok = send(tok, b_fifo_0_0_s, b_1_0);
    let (tok, b_2_0) = recv(tok, b_col_0);
    let tok = send(tok, b_fifo_0_0_s, b_2_0);
    let (tok, b_3_0) = recv(tok, b_col_0);
    let tok = send(tok, b_fifo_0_0_s, b_3_0);
    let (tok, b_0_1) = recv(tok, b_col_1);
    let tok = send(tok, b_fifo_1_0_s, b_0_1);
    let (tok, b_1_1) = recv(tok, b_col_1);
    let tok = send(tok, b_fifo_1_0_s, b_1_1);
    let (tok, b_2_1) = recv(tok, b_col_1);
    let tok = send(tok, b_fifo_1_0_s, b_2_1);
    let (tok, b_3_1) = recv(tok, b_col_1);
    let tok = send(tok, b_fifo_1_0_s, b_3_1);

    // Collect results
    let (tok, c_0_0) = recv(tok, c_out_0_0_r);
    let (tok, c_0_1) = recv(tok, c_out_0_1_r);
    let (tok, c_1_0) = recv(tok, c_out_1_0_r);
    let (tok, c_1_1) = recv(tok, c_out_1_1_r);

    // Drain edges
    let (tok, _) = recv(tok, a_fifo_0_2_r);
    let (tok, _) = recv(tok, a_fifo_0_2_r);
    let (tok, _) = recv(tok, a_fifo_0_2_r);
    let (tok, _) = recv(tok, a_fifo_0_2_r);
    let (tok, _) = recv(tok, a_fifo_1_2_r);
    let (tok, _) = recv(tok, a_fifo_1_2_r);
    let (tok, _) = recv(tok, a_fifo_1_2_r);
    let (tok, _) = recv(tok, a_fifo_1_2_r);
    let (tok, _) = recv(tok, b_fifo_0_2_r);
    let (tok, _) = recv(tok, b_fifo_0_2_r);
    let (tok, _) = recv(tok, b_fifo_0_2_r);
    let (tok, _) = recv(tok, b_fifo_0_2_r);
    let (tok, _) = recv(tok, b_fifo_1_2_r);
    let (tok, _) = recv(tok, b_fifo_1_2_r);
    let (tok, _) = recv(tok, b_fifo_1_2_r);
    let (tok, _) = recv(tok, b_fifo_1_2_r);

    // Build and send result
    let C = u32[2][2]:[[c_0_0, c_0_1], [c_1_0, c_1_1]];
    let tok = send(tok, c_result, C);
  }
}

// ============================================================================
// Test 1: Zero Matrix
// ============================================================================
// A = [[0,0,0,0], [0,0,0,0]]
// B = [[0,0], [0,0], [0,0], [0,0]]
// Expected: C = [[0,0], [0,0]]

#[test_proc]
proc test_zero_matrix {
  terminator: chan<bool> out;

  config(terminator: chan<bool> out) { (terminator,) }

  init { () }

  next(state: ()) {
    let (a_row_0_s, a_row_0_r) = chan<u32>("test_a_row_0");
    let (a_row_1_s, a_row_1_r) = chan<u32>("test_a_row_1");
    let (b_col_0_s, b_col_0_r) = chan<u32>("test_b_col_0");
    let (b_col_1_s, b_col_1_r) = chan<u32>("test_b_col_1");
    let (c_result_s, c_result_r) = chan<u32[2][2]>("test_c_result");

    spawn SystolicArray_2x2_u32(a_row_0_r, a_row_1_r, b_col_0_r, b_col_1_r, c_result_s);

    let tok = join();

    // Send all zeros for A
    let tok = send(tok, a_row_0_s, u32:0);
    let tok = send(tok, a_row_0_s, u32:0);
    let tok = send(tok, a_row_0_s, u32:0);
    let tok = send(tok, a_row_0_s, u32:0);
    let tok = send(tok, a_row_1_s, u32:0);
    let tok = send(tok, a_row_1_s, u32:0);
    let tok = send(tok, a_row_1_s, u32:0);
    let tok = send(tok, a_row_1_s, u32:0);

    // Send all zeros for B
    let tok = send(tok, b_col_0_s, u32:0);
    let tok = send(tok, b_col_0_s, u32:0);
    let tok = send(tok, b_col_0_s, u32:0);
    let tok = send(tok, b_col_0_s, u32:0);
    let tok = send(tok, b_col_1_s, u32:0);
    let tok = send(tok, b_col_1_s, u32:0);
    let tok = send(tok, b_col_1_s, u32:0);
    let tok = send(tok, b_col_1_s, u32:0);

    let (tok, C) = recv(tok, c_result_r);

    // Verify all zeros
    assert_eq(C[0][0], u32:0);
    assert_eq(C[0][1], u32:0);
    assert_eq(C[1][0], u32:0);
    assert_eq(C[1][1], u32:0);

    let tok = send(tok, terminator, true);
  }
}

// ============================================================================
// Test 2: Identity Matrix
// ============================================================================
// A = [[1,0,0,0], [0,1,0,0]]
// B = [[1,0], [0,1], [0,0], [0,0]]
// Expected: C = [[1,0], [0,1]]

#[test_proc]
proc test_identity_matrix {
  terminator: chan<bool> out;

  config(terminator: chan<bool> out) { (terminator,) }

  init { () }

  next(state: ()) {
    let (a_row_0_s, a_row_0_r) = chan<u32>("test_a_row_0");
    let (a_row_1_s, a_row_1_r) = chan<u32>("test_a_row_1");
    let (b_col_0_s, b_col_0_r) = chan<u32>("test_b_col_0");
    let (b_col_1_s, b_col_1_r) = chan<u32>("test_b_col_1");
    let (c_result_s, c_result_r) = chan<u32[2][2]>("test_c_result");

    spawn SystolicArray_2x2_u32(a_row_0_r, a_row_1_r, b_col_0_r, b_col_1_r, c_result_s);

    let tok = join();

    // A row 0: [1,0,0,0]
    let tok = send(tok, a_row_0_s, u32:1);
    let tok = send(tok, a_row_0_s, u32:0);
    let tok = send(tok, a_row_0_s, u32:0);
    let tok = send(tok, a_row_0_s, u32:0);

    // A row 1: [0,1,0,0]
    let tok = send(tok, a_row_1_s, u32:0);
    let tok = send(tok, a_row_1_s, u32:1);
    let tok = send(tok, a_row_1_s, u32:0);
    let tok = send(tok, a_row_1_s, u32:0);

    // B col 0: [1,0,0,0]
    let tok = send(tok, b_col_0_s, u32:1);
    let tok = send(tok, b_col_0_s, u32:0);
    let tok = send(tok, b_col_0_s, u32:0);
    let tok = send(tok, b_col_0_s, u32:0);

    // B col 1: [0,1,0,0]
    let tok = send(tok, b_col_1_s, u32:0);
    let tok = send(tok, b_col_1_s, u32:1);
    let tok = send(tok, b_col_1_s, u32:0);
    let tok = send(tok, b_col_1_s, u32:0);

    let (tok, C) = recv(tok, c_result_r);

    // Verify identity
    assert_eq(C[0][0], u32:1);
    assert_eq(C[0][1], u32:0);
    assert_eq(C[1][0], u32:0);
    assert_eq(C[1][1], u32:1);

    let tok = send(tok, terminator, true);
  }
}

// ============================================================================
// Test 3: General Matrix Multiplication
// ============================================================================
// A = [[1,2,3,4], [5,6,7,8]]
// B = [[1,2], [3,4], [5,6], [7,8]]
// Expected: C = [[50,60], [114,140]]

#[test_proc]
proc test_general_matmul {
  terminator: chan<bool> out;

  config(terminator: chan<bool> out) { (terminator,) }

  init { () }

  next(state: ()) {
    let (a_row_0_s, a_row_0_r) = chan<u32>("test_a_row_0");
    let (a_row_1_s, a_row_1_r) = chan<u32>("test_a_row_1");
    let (b_col_0_s, b_col_0_r) = chan<u32>("test_b_col_0");
    let (b_col_1_s, b_col_1_r) = chan<u32>("test_b_col_1");
    let (c_result_s, c_result_r) = chan<u32[2][2]>("test_c_result");

    spawn SystolicArray_2x2_u32(a_row_0_r, a_row_1_r, b_col_0_r, b_col_1_r, c_result_s);

    let tok = join();

    // A row 0: [1,2,3,4]
    let tok = send(tok, a_row_0_s, u32:1);
    let tok = send(tok, a_row_0_s, u32:2);
    let tok = send(tok, a_row_0_s, u32:3);
    let tok = send(tok, a_row_0_s, u32:4);

    // A row 1: [5,6,7,8]
    let tok = send(tok, a_row_1_s, u32:5);
    let tok = send(tok, a_row_1_s, u32:6);
    let tok = send(tok, a_row_1_s, u32:7);
    let tok = send(tok, a_row_1_s, u32:8);

    // B col 0: [1,3,5,7]
    let tok = send(tok, b_col_0_s, u32:1);
    let tok = send(tok, b_col_0_s, u32:3);
    let tok = send(tok, b_col_0_s, u32:5);
    let tok = send(tok, b_col_0_s, u32:7);

    // B col 1: [2,4,6,8]
    let tok = send(tok, b_col_1_s, u32:2);
    let tok = send(tok, b_col_1_s, u32:4);
    let tok = send(tok, b_col_1_s, u32:6);
    let tok = send(tok, b_col_1_s, u32:8);

    let (tok, C) = recv(tok, c_result_r);

    // Verify: C[0][0] = 1*1 + 2*3 + 3*5 + 4*7 = 50
    assert_eq(C[0][0], u32:50);
    // C[0][1] = 1*2 + 2*4 + 3*6 + 4*8 = 60
    assert_eq(C[0][1], u32:60);
    // C[1][0] = 5*1 + 6*3 + 7*5 + 8*7 = 114
    assert_eq(C[1][0], u32:114);
    // C[1][1] = 5*2 + 6*4 + 7*6 + 8*8 = 140
    assert_eq(C[1][1], u32:140);

    let tok = send(tok, terminator, true);
  }
}