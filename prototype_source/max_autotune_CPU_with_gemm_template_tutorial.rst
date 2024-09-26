Max-autotune Support on CPU with GEMM Template Tutorial
==============================================================

**Author**: `Jiong Gong <https://github.com/jgong5>`__, Leslie Fang <https://github.com/leslie-fang-intel>`__, `Chunyuan Wu <https://github.com/chunyuan-w>`_

Prerequisites:
----------------
-  `torch.compile and TorchInductor concepts in PyTorch <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

Introduction
------------
"max-autotune" mode for the Inductor CPU backend in torch.compile profiles multiple implementations of operations at compile time and selects the best-performing one,
trading longer compilation times for improved runtime performance. This enhancement is particularly beneficial for GEMM-related operations.
In the Inductor CPU backend, we’ve introduced a C++ template-based GEMM implementation as an alternative to the ATen-based approach that relies on oneDNN and MKL libraries.
This is similar to the max-autotune mode on CUDA, where implementations from ATen, Triton, and CUTLASS are considered.


API
------------
The API to turn on the "max-autotune" mode: `compiled = torch.compile(model, mode='max-autotune')`.

By setting `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=CPP`, the selected backend is forced to be CPP template.


Example code
------------
Lauch the below code by setting `TORCHINDUCTOR_FREEZING=1`, we can find the autotuning log similar to (the performance numbers are for demonstration purpose):

The below means that CPP template outperforms ATen kernel and it will be selected.
TODO: perf numbers?
```
AUTOTUNE linear_unary(64x16, 32x16, 32)
  cpp_packed_gemm_0 0.2142 ms 100.0% 
  _linear_pointwise 0.2441 ms 87.7% 
```


```
import torch
from torch._inductor import config
config.trace.log_autotuning_results = True # enable the log of autotuning results

class M(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        **kwargs,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features,
            out_features,
            bias,
            **kwargs,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

amp_enabled = True
batch_size = 64
in_features = 16
out_features = 32
bias = True

x = torch.randn(batch_size, in_features)
model = M(in_features, out_features, bias)

with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
    compiled = torch.compile(model, mode="max-autotune") # turn on "max-autotune" mode
    y = compiled(x)
```    

To check the generated output code, set `TORCH_LOGS="+output_code"`.
TODO: output code too long
```
cpp_fused__to_copy_relu_1 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'bfloat16*'], '''
#include "/tmp/torchinductor_chunyuan/xw/cxww3s7wxrujoyxna7mlcjktid2uu6nntixqwm542xfkd756gl3x.h"
#include <c10/util/Unroll.h>



template <bool accum>
inline void kernel_micro_gemm_amx_kernel_32_2(
    AMXState& amx_state,
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint8_t tilecfg_rows
) {
    // TODO(jgong5): add prefetch hint for A, B, C
    auto loadconfig = [](const amx_tilecfg& cfg) {
        _tile_loadconfig(&cfg);
    };
    const auto last_k_offset = K / 32 * 32;
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 64, 32 / 16, 2, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 32 / 16, 2, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(1, C + 0 * ldc + 16, ldc * sizeof(float));
        _tile_loadd(2, C + 16 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(3, C + 16 * ldc + 16, ldc * sizeof(float));
    };
    auto zero_c = [&]() {
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
        _tile_stream_loadd(4, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(6, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(0, 4, 6);
        _tile_loadd(7, B + k * ldb + 32, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(1, 4, 7);
        _tile_stream_loadd(5, A + 16 * lda + k, lda * sizeof(bfloat16));
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_stored(1, C + 0 * ldc + 16, ldc * sizeof(float));
        _tile_stored(2, C + 16 * ldc + 0, ldc * sizeof(float));
        _tile_stored(3, C + 16 * ldc + 16, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 32 / 16, 2, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}
template <bool accum>
inline void kernel_micro_gemm_amx_kernel_16_2(
    AMXState& amx_state,
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    float* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint8_t tilecfg_rows
) {
    // TODO(jgong5): add prefetch hint for A, B, C
    auto loadconfig = [](const amx_tilecfg& cfg) {
        _tile_loadconfig(&cfg);
    };
    const auto last_k_offset = K / 32 * 32;
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 64, 16 / 16, 2, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 16 / 16, 2, loadconfig);
    }
    auto load_c = [&]() {
        _tile_loadd(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_loadd(1, C + 0 * ldc + 16, ldc * sizeof(float));
    };
    auto zero_c = [&]() {
        _tile_zero(0);
        _tile_zero(1);
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
        _tile_stream_loadd(2, A + 0 * lda + k, lda * sizeof(bfloat16));
        _tile_loadd(3, B + k * ldb + 0, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(0, 2, 3);
        _tile_loadd(4, B + k * ldb + 32, ldb * 2 * sizeof(bfloat16));
        _tile_dpbf16ps(1, 2, 4);
    };

    #pragma GCC unroll 4
    for (int k = 0; k < last_k_offset; k += 32) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
        _tile_stored(0, C + 0 * ldc + 0, ldc * sizeof(float));
        _tile_stored(1, C + 0 * ldc + 16, ldc * sizeof(float));
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof(bfloat16), 16 / 16, 2, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}

template <bool accum>
inline void kernel_micro_gemm(
    AMXState& amx_state,
    const bfloat16* __restrict__ A,
    const bfloat16* __restrict__ B,
    float* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    TORCH_CHECK(N % 32 == 0, "N dimension must be multiple of 32");
    TORCH_CHECK(K % 2 == 0, "K dimension must be multiple of 2");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += 32) {
        int64_t block_m = std::min<int64_t>(M - m, 32);
        int64_t m_tail = m;
        for (int64_t n = 0; n < N; n += 32) {
            if (block_m >= 32) {
                kernel_micro_gemm_amx_kernel_32_2<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    16
                );
                block_m -= 32;
                m_tail += 32;
            }
            else
            if (block_m >= 16) {
                kernel_micro_gemm_amx_kernel_16_2<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    16
                );
                block_m -= 16;
                m_tail += 16;
            }
            if (block_m > 0) {
                kernel_micro_gemm_amx_kernel_16_2<accum>(
                    amx_state,
                    A + m_tail * lda,
                    B + n,
                    C + m_tail * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    block_m
                );
            }
        }
    }
}

extern "C" 
void kernel(const bfloat16* X, const bfloat16* W, const bfloat16* inp, bfloat16* Y)
{

    constexpr int64_t num_threads = 240;
    constexpr int64_t N = 32;
    constexpr int64_t K = 16;
    constexpr int64_t Mr = 32;
    constexpr int64_t Nr = 32;
    constexpr int64_t Kr = 32;
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
    constexpr int64_t M = static_cast<int64_t>(64L);
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
    constexpr int64_t Mt_blocks = 1;
    constexpr int64_t Nt_blocks = 1;
    constexpr int64_t Kt_blocks = 1;
    constexpr int64_t Mc_blocks = 1;
    constexpr int64_t Nc_blocks = 1;
    constexpr int64_t Kc_blocks = 1;
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    constexpr int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    constexpr int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;

    // make sure all partitions are assigned
    TORCH_CHECK(
        Mt_blocks * Nt_blocks * Kt_blocks * 240 >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
    #pragma omp parallel num_threads(240)
    {
        const int tid = omp_get_thread_num();
        const int64_t k_group_id = tid / num_Kt_blocks;
        const int64_t k_slice_id = tid % num_Kt_blocks;
        const int64_t n_group_id = k_group_id / num_Nt_blocks;
        const int64_t n_slice_id = k_group_id % num_Nt_blocks;
        const int64_t k_block_start = k_slice_id * Kt_blocks;
        const int64_t k_block_end = std::min(k_block_start + Kt_blocks, Kr_blocks);
        const int64_t n_block_start = n_slice_id * Nt_blocks;
        const int64_t n_block_end = std::min(n_block_start + Nt_blocks, Nr_blocks);
        const int64_t m_block_start = std::min(n_group_id * Mt_blocks, Mr_blocks);
        const int64_t m_block_end = std::min(m_block_start + Mt_blocks, Mr_blocks);
        const int64_t num_Mc_blocks_per_thread = (m_block_end - m_block_start + Mc_blocks - 1) / Mc_blocks;
        AMXState amx_state;
        auto _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); auto local_acc_buf = _local_acc_buf.get();
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            const int64_t my_mc_block_id = (mc_block_id + n_slice_id) % num_Mc_blocks_per_thread;
            const int64_t mc = m_block_start + my_mc_block_id * Mc_blocks;
            const int64_t m_start = mc * Mr;
            const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
            const int64_t m_size = m_end - m_start;
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                const int64_t n_start = nc * Nr;
                const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
                const int64_t n_size = n_end - n_start;
                // NB: assume we pad N, nc_block_end won't exceed padded N here.
                const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
                if (_local_acc_buf == nullptr) { _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); local_acc_buf = _local_acc_buf.get(); }
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
                        if (kc == k_block_start) {
                            kernel_micro_gemm<static_cast<bool>(false)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + (16L*m_start))]),
                                &(W[static_cast<int64_t>((32L*k_start) + (512L*nci))]),
                                &(local_acc_buf[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(16L),
                                static_cast<int64_t>(32L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                        } else {
                            kernel_micro_gemm<static_cast<bool>(true)>(
                                amx_state,
                                &(X[static_cast<int64_t>(k_start + (16L*m_start))]),
                                &(W[static_cast<int64_t>((32L*k_start) + (512L*nci))]),
                                &(local_acc_buf[static_cast<int64_t>((Nr*nci) + ((-1L)*Nr*nc))]),
                                static_cast<int64_t>(m_end + ((-1L)*m_start)),
                                static_cast<int64_t>(Nr),
                                static_cast<int64_t>(k_end + ((-1L)*k_start)),
                                static_cast<int64_t>(16L),
                                static_cast<int64_t>(32L),
                                static_cast<int64_t>(Nc_blocks*Nr)
                            );

                        }
                    }
                }
                {
                    {
                        #pragma GCC ivdep
                        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(m_end + ((-1L)*m_start)); x0+=static_cast<int64_t>(1L))
                        {
                            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L)))); x1+=static_cast<int64_t>(16L))
                            {
                                auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(inp + static_cast<int64_t>(n_start + x1), static_cast<int64_t>(16));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<int64_t>(x1 + (Nc_blocks*Nr*x0)), static_cast<int64_t>(16));
                                auto tmp1 = at::vec::convert<float>(tmp0);
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                                auto tmp5 = static_cast<float>(0.0);
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = at::vec::maximum(tmp3, tmp6);
                                auto tmp8 = at::vec::convert<bfloat16>(tmp7);
                                tmp8.store(Y + static_cast<int64_t>(n_start + x1 + (32L*m_start) + (32L*x0)), static_cast<int64_t>(16));
                            }
                            for(int64_t x1=static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L)))); x1<static_cast<int64_t>(n_end + ((-1L)*n_start)); x1+=(static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))) == 0 ? 1 : static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L)))))))
                            {
                                auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(inp + static_cast<int64_t>(n_start + x1), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(local_acc_buf + static_cast<int64_t>(x1 + (Nc_blocks*Nr*x0)), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))));
                                auto tmp1 = at::vec::convert<float>(tmp0);
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp4 = at::vec::convert<bfloat16>(tmp3);
                                auto tmp5 = static_cast<float>(0.0);
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = at::vec::maximum(tmp3, tmp6);
                                auto tmp8 = at::vec::convert<bfloat16>(tmp7);
                                tmp8.store(Y + static_cast<int64_t>(n_start + x1 + (32L*m_start) + (32L*x0)), static_cast<int64_t>(n_end + ((-1L)*n_start) + ((-16L)*(c10::div_floor_integer(static_cast<int64_t>((n_end + ((-1L)*n_start))), static_cast<int64_t>(16L))))));
                            }
                        }
                    }

                }
            }
        }
        amx_state.release([]() { _tile_release(); });
    }
}
''')
```


Conclusion
------------
In this tutorial, we introduced max-autotune support on CPU with GEMM template. We explained the API to activate this feature and demonstrated
the generated code of GEMM template.

This feature is in prototype stage. If you have any feature requests or run into any issues, please file a bug report at `GitHub issues <https://github.com/pytorch/pytorch/issues>`_.