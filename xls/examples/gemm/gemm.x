fn gemm(arg0: u32[32][32], arg1: u32[32][32]) -> u32[32][32] {
  let C_init = u32[32][32]:[u32[32]:[0, ...], ...];
  let C = for (i, C): (u32, u32[32][32]) in u32:0..u32:32 {
    for (j, C): (u32, u32[32][32]) in u32:0..u32:32 {
      for (k, C): (u32, u32[32][32]) in u32:0..u32:32 {
        update(C, i, update(C[i], j, (C[i][j] + (arg0[i][k] * arg1[k][j]))))
      }(C)
    }(C)
  }(C_init);
  C
}

#[test]
fn test_gemm() {
  // Test with simple 32x32 identity-like matrices
  let A = u32[32][32]:[u32[32]:[1, ...], ...];
  let B = u32[32][32]:[u32[32]:[2, ...], ...];
  let result = gemm(A, B);
  
  // When multiplying matrix of all 1s by matrix of all 2s,
  // each element should be 32 * 1 * 2 = 64
  assert_eq(result[u32:0][u32:0], u32:64);
  assert_eq(result[u32:15][u32:15], u32:64);
  assert_eq(result[u32:31][u32:31], u32:64);
}

#[test]
fn test_gemm_small() {
  // Create small test matrices with known values
  // A: first row [1,2,0,0,...], second row [3,4,0,0,...], rest zeros
  let row0 = u32[32]:[u32:1, u32:2, u32:0, ...];
  let row1 = u32[32]:[u32:3, u32:4, u32:0, ...];
  let row_zero = u32[32]:[u32:0, ...];
  let A = u32[32][32]:[row0, row1, row_zero, ...];
  
  // B: first column [5,6,0,0,...], second column [7,8,0,0,...], rest zeros
  let col0 = u32[32]:[u32:5, u32:6, u32:0, ...];
  let col1 = u32[32]:[u32:7, u32:8, u32:0, ...];
  let col_zero = u32[32]:[u32:0, ...];
  
  // To make B with those columns, we need to construct it differently
  // B[0] = [5, 7, 0, 0, ...]
  // B[1] = [6, 8, 0, 0, ...]
  let B_row0 = u32[32]:[u32:5, u32:7, u32:0, ...];
  let B_row1 = u32[32]:[u32:6, u32:8, u32:0, ...];
  let B = u32[32][32]:[B_row0, B_row1, col_zero, ...];
  
  let result = gemm(A, B);
  
  // result[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] = 1*5 + 2*6 = 17
  assert_eq(result[u32:0][u32:0], u32:17);
  
  // result[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] = 1*7 + 2*8 = 23
  assert_eq(result[u32:0][u32:1], u32:23);
  
  // result[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] = 3*5 + 4*6 = 39
  assert_eq(result[u32:1][u32:0], u32:39);
  
  // result[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] = 3*7 + 4*8 = 53
  assert_eq(result[u32:1][u32:1], u32:53);
}