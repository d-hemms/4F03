#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <string>

using namespace std;

MPI_Status status;
double start = 0,end = 0;

//assig2.x n m d b c
int main(int argc, char* argv[]) {

	int n = strtol(argv[1], NULL, 10);
	int m = strtol(argv[2], NULL, 10);
	int dist = strtol(argv[3], NULL, 10);
	int block_size = strtol(argv[4], NULL, 10);
	int scheme = strtol(argv[5], NULL, 10);

	int comm_sz;
	int my_rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//use 1-d arrays to simulate 2-d setup
	int *A = (int *) malloc(sizeof(int)*n*m);
	int *B = (int *) malloc(sizeof(int)*m*n);
	int *C = (int *) malloc(sizeof(int)*n*n);

	switch(dist) {
		case 0:
			block_size = (int)ceil((double)n/(double)comm_sz);
			break;
		case 1:
			block_size = 1;
			break;
		case 2:
			break;
		default:
			printf("Choice of data distribution not defined!\n");
	}

	if (my_rank == 0) {
		//generate array values if process 0
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				A[i*m + j] = (int)(drand48()*4);
				B[i*m + j] = (int)(drand48()*4);
			}
		}

		start = MPI_Wtime();
	}
	MPI_Bcast(B, m*n, MPI_INT, 0, MPI_COMM_WORLD); //send B matrix to all processors

	int *a_rows = (int *) malloc(sizeof(int)*block_size*m);
	int *c_rows = (int *) malloc(sizeof(int)*block_size*n);

	//scatter in blocks of comm_sz*block_size rows until all have been delegated
	//if block distribution, iterations will be 1
	int iterations = (int)ceil((double)n/(double)(comm_sz*block_size));
	for (int iter = 0; iter < iterations; ++iter) {
		//boundary cases where non-equal row distribution to all processors. eg 7 rows to 4 processors. Use Send rather than Scatter
		if ((iter == iterations - 1) && n%(comm_sz*block_size) != 0) {
			int remaining = n - (iter*block_size*comm_sz); //remaining rows to be delegated
			int proc = 0;	//processor to get next block of rows
			while (remaining > 0) {

				//update all processors with new remaining and proc values
				MPI_Bcast(&remaining, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(&proc, 1, MPI_INT, 0, MPI_COMM_WORLD);

				if (remaining > block_size) {
					//send whole block of rows to proc
					if (my_rank == 0) {
						if (proc != 0) {
							MPI_Send(&A[(iter*comm_sz + proc)*block_size*m], block_size*m, MPI_INT, proc, 0, MPI_COMM_WORLD);
							MPI_Recv(&C[(iter*comm_sz + proc)*block_size*n], block_size*n, MPI_INT, proc, 0, MPI_COMM_WORLD, &status);
						}
						else {
							for (int i = 0; i < block_size; ++i) { //iterate over each row in the chunk
								for (int j = 0; j < n; ++j) {
									C[(iter*comm_sz*block_size + i)*n + j] = 0;
									for (int k = 0; k < m; ++k) {
										if (scheme) {
											C[(iter*comm_sz*block_size + i)*n + j] += A[(iter*comm_sz*block_size + i)*m + k] * B[j*m + k];
										}
										else {
											C[(iter*comm_sz*block_size + i)*n + j] += A[(iter*comm_sz*block_size + i)*m + k] * B[k*n + j];
										}
									}
								}
							}
						}
					}
					else if (my_rank == proc) {
						MPI_Recv(a_rows, block_size*m, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
						for (int i = 0; i < block_size; ++i) { //iterate over each row in the chunk
							for (int j = 0; j < n; ++j) {
									c_rows[i*n + j] = 0;
								for (int k = 0; k < m; ++k) {
									if (scheme) {
										c_rows[i*n + j] += a_rows[i*m + k] * B[j*m + k];
									}
									else {
										c_rows[i*n + j] += a_rows[i*m + k] * B[k*n + j];
									}
								}
							}
						}
						MPI_Send(c_rows, block_size*n, MPI_INT, 0, 0, MPI_COMM_WORLD);
					}
					if (my_rank == 0) {
						remaining -= block_size;
						proc++;
					}
				}
				else {
					//send remaining rows
					if (my_rank == 0) {
						if (proc != 0) {
							MPI_Send(&A[(iter*comm_sz + proc)*block_size*m], remaining*m, MPI_INT, proc, 0, MPI_COMM_WORLD);
							MPI_Recv(&C[(iter*comm_sz + proc)*block_size*n], remaining*n, MPI_INT, proc, 0, MPI_COMM_WORLD, &status);
						}
						else {
							for (int i = 0; i < remaining; ++i) { //iterate over each row in the block
								for (int j = 0; j < n; ++j) {
									C[(iter*comm_sz*block_size + i)*n + j] = 0;
									for (int k = 0; k < m; ++k) {
										if (scheme) {
											C[(iter*comm_sz*block_size + i)*n + j] += A[(iter*comm_sz*block_size + i)*m + k] * B[j*m + k];
										}
										else {
											C[(iter*comm_sz*block_size + i)*n + j] += A[(iter*comm_sz*block_size + i)*m + k] * B[k*n + j];
										}
									}
								}
							}
						}
					}
					else if (my_rank == proc) {
						MPI_Recv(a_rows, remaining*m, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
						for (int i = 0; i < remaining; ++i) { //iterate over each row in the block
							for (int j = 0; j < n; ++j) {
								c_rows[i*n + j] = 0;
								for (int k = 0; k < m; ++k) {
									if (scheme) {
										c_rows[i*n + j] += a_rows[i*m + k] * B[j*m + k];
									}
									else {
										c_rows[i*n + j] += a_rows[i*m + k] * B[k*n + j];
									}
								}
							}
						}
						MPI_Send(c_rows, remaining*n, MPI_INT, 0, 0, MPI_COMM_WORLD);
					}
					remaining = 0;
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}
		}
		else {
			MPI_Scatter(&A[iter*comm_sz*block_size*m], block_size*m, MPI_INT, a_rows, block_size*m, MPI_INT, 0, MPI_COMM_WORLD); //scatter next comm_sz*block_size rows of A
			for (int i = 0; i < block_size; ++i) { //iterate over each row in the block
				for (int j = 0; j < n; ++j) {
					c_rows[i*n + j] = 0;
					for (int k = 0; k < m; ++k) {
						if (scheme) { //(column iterator * row size) + row iteration
							c_rows[i*n + j] += a_rows[i*m + k] * B[j*m + k];
						}
						else {
							c_rows[i*n + j] += a_rows[i*m + k] * B[k*n + j];
						}
					}
				}
			}
			MPI_Gather(c_rows, block_size*n, MPI_INT, &C[iter*comm_sz*block_size*n], block_size*n, MPI_INT, 0, MPI_COMM_WORLD); //gather next comm_sz*block_size into C
		}
	}

	free(A);
	free(B);
	free(C);
	free(a_rows);
	free(c_rows);

	if (my_rank == 0) {
		end = MPI_Wtime();
		/*for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				printf("%d ",A[i*m + j]);
			}
			printf("\n");
		}
		if (scheme) {
			printf("\n\n");
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					printf("%d ",B[j*m + i]); //(column iterator * row size) + row iteration
				}
				printf("\n");
			}
		}
		else {
			printf("\n\n");
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					printf("%d ",B[i*n + j]); //(row iterator * column size) + column iteration
				}
				printf("\n");
			}
		}
		printf("\n\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d ",C[i*n + j]);
			}
			printf("\n");
		}*/
		printf("%f\n", end-start);
	}
	
	MPI_Finalize();

	return 0;
}
