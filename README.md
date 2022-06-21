# MySGEMM
cuda practice project

# Enviroment
device: RTX 2060  
nvcc: cuda 10.2

# Bash
```bash
nvcc -std=c++11 --gpu-architecture=compute_75 --gpu-code=sm_75 -O2 -lcublas -lcurand sgemm.cu -o sgemm
./sgemm
./sgemm 1024
./sgemm 1024 512 1024
```

# Optimizing Goal
- [x] coalesce gmem access (strided access)
- [x] smem bank conflict
- [x] DRAM diagonal access (useless)
- [x] use float4 shared mem for lower LSU utilization
- [x] prefetch (LDG to register -> other works -> STS from register) to hide the LDG latency and reduce stall
- [ ] prefetch (next frag load -> this frag calculate) to hide the LDS latency (failed due to the register limit)
- [x] double buffer smem to reduce sync
- [x] adjust parameters, consider hardware constraints and theoretical occupancy
- [ ] optimize for small grid (low wave coverage)


# Performance (compare with cublas)
|M=N=K|512|1024|2048|4096|8192|16384|
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
|performance|68.36%|83.66%|90.08%|90.72%|91.13%|90.89%|
