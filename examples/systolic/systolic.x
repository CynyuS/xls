// Processing Element as a parametric proc
proc PE<row: u32, col: u32, K: u32> {
  a_in: chan<u32> in;
  b_in: chan<u32> in;
  a_out: chan<u32> out;
  b_out: chan<u32> out;
  c_out: chan<u32> out;

  config(a_in: chan<u32> in, b_in: chan<u32> in,
         a_out: chan<u32> out, b_out: chan<u32> out,
         c_out: chan<u32> out) {
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
    let tok = if new_k == K {
      send(tok, c_out, new_accum)
    } else {
      tok
    };

    let new_state = if new_k == K {
      (u32:0, u32:0)
    } else {
      (new_accum, new_k)
    };

    new_state
  }
}

// Main systolic array proc for 2x2 matrices
pub proc SystolicGEMM {
  a_data: chan<u32[2][2]> in;
  b_data: chan<u32[2][2]> in;
  c_result: chan<u32[2][2]> out;

  config(a_data: chan<u32[2][2]> in, 
         b_data: chan<u32[2][2]> in,
         c_result: chan<u32[2][2]> out) {
    (a_data, b_data, c_result)
  }

  init { () }

  next(state: ()) {
    let (tok, A) = recv(join(), a_data);
    let (tok, B) = recv(tok, b_data);

    let (a_00_01_s, a_00_01_r) = chan<u32>("a_00_01");
    let (a_01_drain_s, a_01_drain_r) = chan<u32>("a_01_drain");
    let (a_10_11_s, a_10_11_r) = chan<u32>("a_10_11");
    let (a_11_drain_s, a_11_drain_r) = chan<u32>("a_11_drain");
    
    let (b_00_10_s, b_00_10_r) = chan<u32>("b_00_10");
    let (b_01_11_s, b_01_11_r) = chan<u32>("b_01_11");
    let (b_10_drain_s, b_10_drain_r) = chan<u32>("b_10_drain");
    let (b_11_drain_s, b_11_drain_r) = chan<u32>("b_11_drain");
    
    let (a_in_00_s, a_in_00_r) = chan<u32>("a_in_00");
    let (a_in_10_s, a_in_10_r) = chan<u32>("a_in_10");
    let (b_in_00_s, b_in_00_r) = chan<u32>("b_in_00");
    let (b_in_01_s, b_in_01_r) = chan<u32>("b_in_01");
    
    let (c_out_00_s, c_out_00_r) = chan<u32>("c_out_00");
    let (c_out_01_s, c_out_01_r) = chan<u32>("c_out_01");
    let (c_out_10_s, c_out_10_r) = chan<u32>("c_out_10");
    let (c_out_11_s, c_out_11_r) = chan<u32>("c_out_11");
    
    spawn PE<u32:0, u32:0, u32:2>(a_in_00_r, b_in_00_r, a_00_01_s, b_00_10_s, c_out_00_s);
    spawn PE<u32:0, u32:1, u32:2>(a_00_01_r, b_in_01_r, a_01_drain_s, b_01_11_s, c_out_01_s);
    spawn PE<u32:1, u32:0, u32:2>(a_in_10_r, b_00_10_r, a_10_11_s, b_10_drain_s, c_out_10_s);
    spawn PE<u32:1, u32:1, u32:2>(a_10_11_r, b_01_11_r, a_11_drain_s, b_11_drain_s, c_out_11_s);
    
    let tok = send(tok, a_in_00_s, A[0][0]);
    let tok = send(tok, a_in_00_s, A[0][1]);
    let tok = send(tok, a_in_10_s, A[1][0]);
    let tok = send(tok, a_in_10_s, A[1][1]);
    let tok = send(tok, b_in_00_s, B[0][0]);
    let tok = send(tok, b_in_00_s, B[1][0]);
    let tok = send(tok, b_in_01_s, B[0][1]);
    let tok = send(tok, b_in_01_s, B[1][1]);
    
    let (tok, c00) = recv(tok, c_out_00_r);
    let (tok, c01) = recv(tok, c_out_01_r);
    let (tok, c10) = recv(tok, c_out_10_r);
    let (tok, c11) = recv(tok, c_out_11_r);
    
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    
    let C = u32[2][2]:[u32[2]:[c00, c01], u32[2]:[c10, c11]];
    let tok = send(tok, c_result, C);
  }
}

// Test: Simple identity-like matrices
// A = [[1,0],[0,1]], B = [[2,0],[0,2]]
// Expected: C = [[2,0],[0,2]]
#[test_proc]
proc test_systolic_simple {
  terminator: chan<bool> out;
  
  // Inputs to PE grid
  a_in_00_s: chan<u32> out;
  a_in_10_s: chan<u32> out;
  b_in_00_s: chan<u32> out;
  b_in_01_s: chan<u32> out;
  
  // Outputs from PE grid
  c_out_00_r: chan<u32> in;
  c_out_01_r: chan<u32> in;
  c_out_10_r: chan<u32> in;
  c_out_11_r: chan<u32> in;
  
  // Drains
  a_01_drain_r: chan<u32> in;
  a_11_drain_r: chan<u32> in;
  b_10_drain_r: chan<u32> in;
  b_11_drain_r: chan<u32> in;
  
  init { () }

  config(terminator: chan<bool> out) {
    // Create all internal channels
    let (a_00_01_s, a_00_01_r) = chan<u32>("a_00_01");
    let (a_01_drain_s, a_01_drain_r) = chan<u32>("a_01_drain");
    let (a_10_11_s, a_10_11_r) = chan<u32>("a_10_11");
    let (a_11_drain_s, a_11_drain_r) = chan<u32>("a_11_drain");
    
    let (b_00_10_s, b_00_10_r) = chan<u32>("b_00_10");
    let (b_01_11_s, b_01_11_r) = chan<u32>("b_01_11");
    let (b_10_drain_s, b_10_drain_r) = chan<u32>("b_10_drain");
    let (b_11_drain_s, b_11_drain_r) = chan<u32>("b_11_drain");
    
    let (a_in_00_s, a_in_00_r) = chan<u32>("a_in_00");
    let (a_in_10_s, a_in_10_r) = chan<u32>("a_in_10");
    let (b_in_00_s, b_in_00_r) = chan<u32>("b_in_00");
    let (b_in_01_s, b_in_01_r) = chan<u32>("b_in_01");
    
    let (c_out_00_s, c_out_00_r) = chan<u32>("c_out_00");
    let (c_out_01_s, c_out_01_r) = chan<u32>("c_out_01");
    let (c_out_10_s, c_out_10_r) = chan<u32>("c_out_10");
    let (c_out_11_s, c_out_11_r) = chan<u32>("c_out_11");
    
    // Spawn all PEs
    spawn PE<u32:0, u32:0, u32:2>(a_in_00_r, b_in_00_r, a_00_01_s, b_00_10_s, c_out_00_s);
    spawn PE<u32:0, u32:1, u32:2>(a_00_01_r, b_in_01_r, a_01_drain_s, b_01_11_s, c_out_01_s);
    spawn PE<u32:1, u32:0, u32:2>(a_in_10_r, b_00_10_r, a_10_11_s, b_10_drain_s, c_out_10_s);
    spawn PE<u32:1, u32:1, u32:2>(a_10_11_r, b_01_11_r, a_11_drain_s, b_11_drain_s, c_out_11_s);
    
    (terminator, 
     a_in_00_s, a_in_10_s, b_in_00_s, b_in_01_s,
     c_out_00_r, c_out_01_r, c_out_10_r, c_out_11_r,
     a_01_drain_r, a_11_drain_r, b_10_drain_r, b_11_drain_r)
  }
  
  next(state: ()) {
    // A = [[1,0],[0,1]] (identity), B = [[2,0],[0,2]]
    let tok = join();
    
    // Send A[0][:] = [1,0] to PE[0][0]
    let tok = send(tok, a_in_00_s, u32:1);
    let tok = send(tok, a_in_00_s, u32:0);
    
    // Send A[1][:] = [0,1] to PE[1][0]
    let tok = send(tok, a_in_10_s, u32:0);
    let tok = send(tok, a_in_10_s, u32:1);
    
    // Send B[:][0] = [2,0] to PE[0][0]
    let tok = send(tok, b_in_00_s, u32:2);
    let tok = send(tok, b_in_00_s, u32:0);
    
    // Send B[:][1] = [0,2] to PE[0][1]
    let tok = send(tok, b_in_01_s, u32:0);
    let tok = send(tok, b_in_01_s, u32:2);
    
    // Receive results
    let (tok, c00) = recv(tok, c_out_00_r);
    let (tok, c01) = recv(tok, c_out_01_r);
    let (tok, c10) = recv(tok, c_out_10_r);
    let (tok, c11) = recv(tok, c_out_11_r);
    
    // Drain unused outputs
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    
    // Check: Expected C = [[2,0],[0,2]]
    let success = 
      (c00 == u32:2) && (c01 == u32:0) &&
      (c10 == u32:0) && (c11 == u32:2);
    
    let tok = send(tok, terminator, success);
  }
}

// Test: All ones
// A = [[1,1],[1,1]], B = [[1,1],[1,1]]
// Expected: C = [[2,2],[2,2]] (each element = 1*1 + 1*1 = 2)
#[test_proc]
proc test_systolic_ones {
  terminator: chan<bool> out;
  
  // Inputs to PE grid
  a_in_00_s: chan<u32> out;
  a_in_10_s: chan<u32> out;
  b_in_00_s: chan<u32> out;
  b_in_01_s: chan<u32> out;
  
  // Outputs from PE grid
  c_out_00_r: chan<u32> in;
  c_out_01_r: chan<u32> in;
  c_out_10_r: chan<u32> in;
  c_out_11_r: chan<u32> in;
  
  // Drains
  a_01_drain_r: chan<u32> in;
  a_11_drain_r: chan<u32> in;
  b_10_drain_r: chan<u32> in;
  b_11_drain_r: chan<u32> in;
  
  init { () }

  config(terminator: chan<bool> out) {
    // Create all internal channels
    let (a_00_01_s, a_00_01_r) = chan<u32>("a_00_01");
    let (a_01_drain_s, a_01_drain_r) = chan<u32>("a_01_drain");
    let (a_10_11_s, a_10_11_r) = chan<u32>("a_10_11");
    let (a_11_drain_s, a_11_drain_r) = chan<u32>("a_11_drain");
    
    let (b_00_10_s, b_00_10_r) = chan<u32>("b_00_10");
    let (b_01_11_s, b_01_11_r) = chan<u32>("b_01_11");
    let (b_10_drain_s, b_10_drain_r) = chan<u32>("b_10_drain");
    let (b_11_drain_s, b_11_drain_r) = chan<u32>("b_11_drain");
    
    let (a_in_00_s, a_in_00_r) = chan<u32>("a_in_00");
    let (a_in_10_s, a_in_10_r) = chan<u32>("a_in_10");
    let (b_in_00_s, b_in_00_r) = chan<u32>("b_in_00");
    let (b_in_01_s, b_in_01_r) = chan<u32>("b_in_01");
    
    let (c_out_00_s, c_out_00_r) = chan<u32>("c_out_00");
    let (c_out_01_s, c_out_01_r) = chan<u32>("c_out_01");
    let (c_out_10_s, c_out_10_r) = chan<u32>("c_out_10");
    let (c_out_11_s, c_out_11_r) = chan<u32>("c_out_11");
    
    // Spawn all PEs
    spawn PE<u32:0, u32:0, u32:2>(a_in_00_r, b_in_00_r, a_00_01_s, b_00_10_s, c_out_00_s);
    spawn PE<u32:0, u32:1, u32:2>(a_00_01_r, b_in_01_r, a_01_drain_s, b_01_11_s, c_out_01_s);
    spawn PE<u32:1, u32:0, u32:2>(a_in_10_r, b_00_10_r, a_10_11_s, b_10_drain_s, c_out_10_s);
    spawn PE<u32:1, u32:1, u32:2>(a_10_11_r, b_01_11_r, a_11_drain_s, b_11_drain_s, c_out_11_s);
    
    (terminator, 
     a_in_00_s, a_in_10_s, b_in_00_s, b_in_01_s,
     c_out_00_r, c_out_01_r, c_out_10_r, c_out_11_r,
     a_01_drain_r, a_11_drain_r, b_10_drain_r, b_11_drain_r)
  }
  
  next(state: ()) {
    // A = [[1,1],[1,1]], B = [[1,1],[1,1]]
    let tok = join();
    
    // Send A[0][:] = [1,1] to PE[0][0]
    let tok = send(tok, a_in_00_s, u32:1);
    let tok = send(tok, a_in_00_s, u32:1);
    
    // Send A[1][:] = [1,1] to PE[1][0]
    let tok = send(tok, a_in_10_s, u32:1);
    let tok = send(tok, a_in_10_s, u32:1);
    
    // Send B[:][0] = [1,1] to PE[0][0]
    let tok = send(tok, b_in_00_s, u32:1);
    let tok = send(tok, b_in_00_s, u32:1);
    
    // Send B[:][1] = [1,1] to PE[0][1]
    let tok = send(tok, b_in_01_s, u32:1);
    let tok = send(tok, b_in_01_s, u32:1);
    
    // Receive results
    let (tok, c00) = recv(tok, c_out_00_r);
    let (tok, c01) = recv(tok, c_out_01_r);
    let (tok, c10) = recv(tok, c_out_10_r);
    let (tok, c11) = recv(tok, c_out_11_r);
    
    // Drain unused outputs
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    
    // Check: Expected C = [[2,2],[2,2]]
    let success = 
      (c00 == u32:2) && (c01 == u32:2) &&
      (c10 == u32:2) && (c11 == u32:2);
    
    let tok = send(tok, terminator, success);
  }
}

// Test: Known result
// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
// Expected: C = [[19,22],[43,50]]
#[test_proc]
proc test_systolic_known {
  terminator: chan<bool> out;
  
  // Inputs to PE grid
  a_in_00_s: chan<u32> out;
  a_in_10_s: chan<u32> out;
  b_in_00_s: chan<u32> out;
  b_in_01_s: chan<u32> out;
  
  // Outputs from PE grid
  c_out_00_r: chan<u32> in;
  c_out_01_r: chan<u32> in;
  c_out_10_r: chan<u32> in;
  c_out_11_r: chan<u32> in;
  
  // Drains
  a_01_drain_r: chan<u32> in;
  a_11_drain_r: chan<u32> in;
  b_10_drain_r: chan<u32> in;
  b_11_drain_r: chan<u32> in;
  
  init { () }

  config(terminator: chan<bool> out) {
    // Create all internal channels
    let (a_00_01_s, a_00_01_r) = chan<u32>("a_00_01");
    let (a_01_drain_s, a_01_drain_r) = chan<u32>("a_01_drain");
    let (a_10_11_s, a_10_11_r) = chan<u32>("a_10_11");
    let (a_11_drain_s, a_11_drain_r) = chan<u32>("a_11_drain");
    
    let (b_00_10_s, b_00_10_r) = chan<u32>("b_00_10");
    let (b_01_11_s, b_01_11_r) = chan<u32>("b_01_11");
    let (b_10_drain_s, b_10_drain_r) = chan<u32>("b_10_drain");
    let (b_11_drain_s, b_11_drain_r) = chan<u32>("b_11_drain");
    
    let (a_in_00_s, a_in_00_r) = chan<u32>("a_in_00");
    let (a_in_10_s, a_in_10_r) = chan<u32>("a_in_10");
    let (b_in_00_s, b_in_00_r) = chan<u32>("b_in_00");
    let (b_in_01_s, b_in_01_r) = chan<u32>("b_in_01");
    
    let (c_out_00_s, c_out_00_r) = chan<u32>("c_out_00");
    let (c_out_01_s, c_out_01_r) = chan<u32>("c_out_01");
    let (c_out_10_s, c_out_10_r) = chan<u32>("c_out_10");
    let (c_out_11_s, c_out_11_r) = chan<u32>("c_out_11");
    
    // Spawn all PEs
    spawn PE<u32:0, u32:0, u32:2>(a_in_00_r, b_in_00_r, a_00_01_s, b_00_10_s, c_out_00_s);
    spawn PE<u32:0, u32:1, u32:2>(a_00_01_r, b_in_01_r, a_01_drain_s, b_01_11_s, c_out_01_s);
    spawn PE<u32:1, u32:0, u32:2>(a_in_10_r, b_00_10_r, a_10_11_s, b_10_drain_s, c_out_10_s);
    spawn PE<u32:1, u32:1, u32:2>(a_10_11_r, b_01_11_r, a_11_drain_s, b_11_drain_s, c_out_11_s);
    
    (terminator, 
     a_in_00_s, a_in_10_s, b_in_00_s, b_in_01_s,
     c_out_00_r, c_out_01_r, c_out_10_r, c_out_11_r,
     a_01_drain_r, a_11_drain_r, b_10_drain_r, b_11_drain_r)
  }
  
  next(state: ()) {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    let tok = join();
    
    // Send A[0][:] = [1,2] to PE[0][0]
    let tok = send(tok, a_in_00_s, u32:1);
    let tok = send(tok, a_in_00_s, u32:2);
    
    // Send A[1][:] = [3,4] to PE[1][0]
    let tok = send(tok, a_in_10_s, u32:3);
    let tok = send(tok, a_in_10_s, u32:4);
    
    // Send B[:][0] = [5,7] to PE[0][0]
    let tok = send(tok, b_in_00_s, u32:5);
    let tok = send(tok, b_in_00_s, u32:7);
    
    // Send B[:][1] = [6,8] to PE[0][1]
    let tok = send(tok, b_in_01_s, u32:6);
    let tok = send(tok, b_in_01_s, u32:8);
    
    // Receive results
    let (tok, c00) = recv(tok, c_out_00_r);
    let (tok, c01) = recv(tok, c_out_01_r);
    let (tok, c10) = recv(tok, c_out_10_r);
    let (tok, c11) = recv(tok, c_out_11_r);
    
    // Drain unused outputs
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_01_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, a_11_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_10_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    let (tok, _) = recv(tok, b_11_drain_r);
    
    // Check: C[0][0] = 1*5 + 2*7 = 19
    //        C[0][1] = 1*6 + 2*8 = 22
    //        C[1][0] = 3*5 + 4*7 = 43
    //        C[1][1] = 3*6 + 4*8 = 50
    let success = 
      (c00 == u32:19) && (c01 == u32:22) &&
      (c10 == u32:43) && (c11 == u32:50);
    
    let tok = send(tok, terminator, success);
  }
}