// worker.js — Web Worker for CPU IDCT fallback

function cpuIDCT1D(input, N) {
  const output = new Float32Array(N);
  const alpha0 = Math.sqrt(1 / N);
  const alpha = Math.sqrt(2 / N);
  const pi = Math.PI;
  for (let i = 0; i < N; i++) {
    let sum = alpha0 * input[0];
    const theta = pi * (2 * i + 1) / (2 * N);
    for (let f = 1; f < N; f++) {
      sum += alpha * input[f] * Math.cos(f * theta);
    }
    output[i] = sum;
  }
  return output;
}

self.onmessage = function(e) {
  const { type, grid, n, k, startIdx, endIdx } = e.data;

  if (type === 'idct-columns') {
    // Process columns startIdx..endIdx of an n×k grid
    // grid is row-major: grid[row * k + col]
    const results = new Float32Array((endIdx - startIdx) * n);
    for (let j = startIdx; j < endIdx; j++) {
      const col = new Float32Array(n);
      for (let i = 0; i < n; i++) col[i] = grid[i * k + j];
      const out = cpuIDCT1D(col, n);
      const off = (j - startIdx) * n;
      for (let i = 0; i < n; i++) results[off + i] = out[i];
    }
    self.postMessage({ type: 'idct-columns', startIdx, endIdx, results }, [results.buffer]);

  } else if (type === 'idct-rows') {
    // Process rows startIdx..endIdx of an n×k grid
    const results = new Float32Array((endIdx - startIdx) * k);
    for (let i = startIdx; i < endIdx; i++) {
      const row = new Float32Array(k);
      for (let j = 0; j < k; j++) row[j] = grid[i * k + j];
      const out = cpuIDCT1D(row, k);
      const off = (i - startIdx) * k;
      for (let j = 0; j < k; j++) results[off + j] = out[j];
    }
    self.postMessage({ type: 'idct-rows', startIdx, endIdx, results }, [results.buffer]);
  }
};
