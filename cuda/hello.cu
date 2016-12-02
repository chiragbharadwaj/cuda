#include <stdio.h>

const int N = 7;
const int blocksize = 7;

/* Adds the an integer from the [b] array to a character in the same position
 * in the [a] array and stores the result back in [a]. Uses a multithreaded
 * pattern to add the two (each thread modifies a different index in parallel).
 * 
 * Requires: |a| = |b|.
 */
__global__
void hello(char* a, int* b) {
  a[threadIdx.x] += b[threadIdx.x];
}

/* Initializes the arrays and prints out the actual hello-world program.
 *
 * Requires: Computer supports CUDA hardware for multithreaded code to run.
 */
int main(int argc, char* argv[]) {
  char a[N] = "Hello ";
  int  b[N] = {47, 10, 6, 0, -11, 1, 0}; // Diffs between "Hello " and "world!"

  char* ad;
  int*  bd;
  const int csize = N * sizeof(char);
  const int isize = N * sizeof(int);

  printf("%s", a);   // Print out "Hello "

  cudaMalloc((void**) &ad, csize);
  cudaMalloc((void**) &bd, isize);
  
  cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);                   // Set up blocks to copy.
  hello<<<dimGrid, dimBlock>>>(ad, bd); // Works in theory, but requires CUDA.
  cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);

  cudaFree(ad);
  cudaFree(bd);

  printf("%s\n", a); // Print out "world!"
  return EXIT_SUCCESS;
}
