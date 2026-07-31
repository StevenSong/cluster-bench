# ---- fabric selection ----
# control plane: one interface, routable from every node
export NCCL_SOCKET_IFNAME=ens80f0np0        # your admin/mgmt NIC, exact name
export GLOO_SOCKET_IFNAME=ens80f0np0

# data plane: unchanged, both rails, per-rank affinity
export NCCL_IB_HCA=rocep80s0,rocep203s0
export NCCL_IB_DISABLE=0

# ---- THE key setting for a rail-aligned fabric ----
# 0 = a ring must use the same NIC at both ends of a cross-node hop.
# Default (2) lets NCCL try cross-rail paths that don't physically exist here.
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring

# ---- GPUDirect RDMA ----
export NCCL_NET_GDR_LEVEL=PIX           # NIC + GPU pair share a PCIe switch
export NCCL_IB_GID_INDEX=3              # RoCEv2 only; omit for InfiniBand
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10
export NCCL_IB_QPS_PER_CONNECTION=4     # helps saturate a single rail

# ---- intra-node ----
export NCCL_NVLS_ENABLE=0               # no NVSwitch, skip the probe
export NCCL_BUFFSIZE=8388608

# ---- torch / allocator ----
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# ---- first run only ----
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
