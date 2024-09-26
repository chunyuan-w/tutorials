Max-autotune Support on CPU with GEMM Template Tutorial
==============================================================

**Author**: `Jiong Gong <https://github.com/jgong5>`__, `Leslie Fang <https://github.com/leslie-fang-intel>`__, `Chunyuan Wu <https://github.com/chunyuan-w>`__

Prerequisites:
----------------
-  `torch.compile and TorchInductor concepts in PyTorch <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

Introduction
------------
``max-autotune`` mode for the Inductor CPU backend in ``torch.compile`` profiles multiple implementations of operations at compile time and selects the best-performing one,
trading longer compilation times for improved runtime performance. This enhancement is particularly beneficial for GEMM-related operations.
In the Inductor CPU backend, we’ve introduced a C++ template-based GEMM implementation as an alternative to the ATen-based approach that relies on oneDNN and MKL libraries.
This is similar to the max-autotune mode on CUDA, where implementations from ATen, Triton, and CUTLASS are considered.


How to activate ``max-autotune`` mode
------------
To activate the ``max-autotune`` mode in PyTorch, set the ``mode`` argument to ``max-autotune`` when compiling your model using ``torch.compile``.
If you prefer to bypass the tuning process and always use the CPP template implementations, you can configure this via an environment variable: 
``export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=CPP``.


Example code
------------
The below code is an example of using the ``max-autotune`` mode on a simple neural network with a linear layer followed by a ReLU activation.
You could run the example code by setting this environment variable ``export TORCHINDUCTOR_FREEZING=1``.


.. code:: python

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


When running the above code snippet, you will see the autotuning result (the performance numbers are for demonstration purposes).
In this case, CPP template outperforms ATen kernel so that it will be selected.

.. code:: shell

    AUTOTUNE linear_unary(64x16, 32x16, 32)
    cpp_packed_gemm_0 0.2142 ms 100.0% 
    _linear_pointwise 0.2441 ms 87.7% 


We could check the generated output code by setting ``export TORCH_LOGS="+output_code"``.
When CPP template is selected, we won't have ``torch.ops.mkldnn._linear_pointwise.default`` (for bfloat16) or ``torch.ops.mkl._mkl_linear.default`` (for float32)
in the generated code anymore, instead, we'll find kernel based on CPP GEMM template ``cpp_fused__to_copy_relu_1``
(only part of the code is demonstrated below for simplicity) with the epilogue
ReLU fused inside the CPP GEMM template kernel.

.. code:: python

    cpp_fused__to_copy_relu_1 = async_compile.cpp_pybinding(['const bfloat16*', 'const bfloat16*', 'const bfloat16*', 'bfloat16*'], '''
    
    ...

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
        ...
    }
    
    ...

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
        ...
    }

    extern "C" 
    void kernel(const bfloat16* X, const bfloat16* W, const bfloat16* inp, bfloat16* Y)
    {

        constexpr int64_t num_threads = 40;
        constexpr int64_t N = 32;
        constexpr int64_t K = 16;
        constexpr int64_t Mr = 32;
        constexpr int64_t Nr = 32;
        constexpr int64_t Kr = 32;
        constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
        constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
        constexpr int64_t M = static_cast<int64_t>(64L);
        constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;

        ...

        // make sure all partitions are assigned
        TORCH_CHECK(
            Mt_blocks * Nt_blocks * Kt_blocks * 40 >= Mr_blocks * Nr_blocks * Kr_blocks,
            "Not all partitions are assigned."
        );
        #pragma omp parallel num_threads(40)
        {
            const int tid = omp_get_thread_num();
            
            ...

            AMXState amx_state;
            auto _local_acc_buf = std::make_unique<float[]>(static_cast<int64_t>(Mc_blocks*Mr*Nc_blocks*Nr)); auto local_acc_buf = _local_acc_buf.get();
            for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
                
                ...
                
                for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                    ...

                    for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                        int64_t k_start = kc * Kr;
                        int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
                        for (int64_t nci = nc; nci < nc_block_end; nci++) {
                            if (kc == k_block_start) {
                                kernel_micro_gemm<static_cast<bool>(false)>(
                                    ...
                                );

                            } else {
                                kernel_micro_gemm<static_cast<bool>(true)>(
                                    ...
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
                                
                                ...

                            }
                        }

                    }
                }
            }
            amx_state.release([]() { _tile_release(); });
        }
    }
    ''')

Conclusion
------------
In this tutorial, we introduced max-autotune support on CPU with GEMM template. We explained the API to activate this feature and demonstrated
the generated code of GEMM template.

This feature is in prototype stage. If you have any feature requests or run into any issues, please file a bug report at `GitHub issues <https://github.com/pytorch/pytorch/issues>`_.




TODO: perf numbers?
TODO: output code too long
