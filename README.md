# MySGEMM
cuda practice project

# Enviroment
device: RTX 2060  
nvcc: cuda 10.2

# Bash
```bash
nvcc -std=c++11 --gpu-architecture=compute_75 --gpu-code=sm_75 -O2 -lcublas -lcurand sgemm.cu -o sgemm
./sgemm
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
- [ ] use tensor core


# Performance (compare with cublas)
|M=N=K|512|1024|2048|4096|8192|16384|
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
|performance|91.87%|91.55%|91.73%|91.90%|91.82%|91.86%|
