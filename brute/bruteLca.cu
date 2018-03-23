#include <cstdio>

extern "C" {
__global__ void cuInit(int V, int *father, int *next, int *depth, int *notAllDone) {
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thid >= V) return;

    next[thid] = father[thid];
    depth[thid] = 0;
}

__global__ void cuCalcDepthRead(int V, int *father, int *next, int *depth, int *tmp, int *notAllDone) {
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thid == 0) *notAllDone = 0;

    if (thid >= V || next[thid] == -1) return;

    tmp[thid] = depth[next[thid]] + 1;

}

__global__ void cuCalcDepthWrite(int V, int *father, int *next, int *depth, int *tmp, int *notAllDone) {
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thid == 0) *notAllDone = 0;

    if (thid >= V || next[thid] == -1) return;

    depth[thid] += tmp[thid];
}

__global__ void cuMoveNextRead(int V, int *father, int *next, int *depth, int *tmp, int *notAllDone) {
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thid >= V || next[thid] == -1) return;

    tmp[thid] = next[next[thid]];
}

__global__ void cuMoveNextWrite(int V, int *father, int *next, int *depth, int *tmp, int *notAllDone) {
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thid >= V || next[thid] == -1) return;

    next[thid] = tmp[thid];

    *notAllDone = 1;
}

__global__ void cuCalcQueries(int Q, int *father, int *depth, int *queries, int *answers) {
    int thid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thid >= Q) return;

    int p = queries[thid * 2];
    int q = queries[thid * 2 + 1];

    if (p == q) answers[thid] = p;

    while (depth[p] != depth[q]) {
        if (depth[p] > depth[q])
            p = father[p];
        else
            q = father[q];
    }

    while (p != q) {
        p = father[p];
        q = father[q];
    }

    answers[thid] = p;
}
}