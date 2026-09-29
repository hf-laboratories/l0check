using System;
using System.IO;
using LevelZero;
using LevelZero.Kernels;

namespace LevelZeroCheck;

internal static class Program
{
    private static int Main()
    {
        Console.WriteLine("=== Level Zero LLM Kernel Verification Suite ===");
        try
        {
            // 1. Initialize Runtime and Default Device
            using var device = LevelZeroRuntime.GetDefaultDevice();
            Console.WriteLine($"Default Device detected: {device.Name}");

            Console.WriteLine($"Configured Device Target: {KernelCatalog.GetConfiguredDeviceTarget()}");
            Console.WriteLine($"IPU_L0_NATIVE_MODULES env var: '{Environment.GetEnvironmentVariable("IPU_L0_NATIVE_MODULES")}'");

            // Re-implement the embedded resource lookup for logging
            string target = KernelCatalog.GetConfiguredDeviceTarget();
            string kernelName = "llm_kernels";
            string[] extensions = { ".zebin", ".bin", ".native", ".spv" };
            foreach (var ext in extensions)
            {
                string rName = $"LevelZero.Kernels.{target}.{kernelName}{ext}";
                using Stream? stream = typeof(KernelCatalog).Assembly.GetManifestResourceStream(rName);
                if (stream != null)
                {
                    Console.WriteLine($"Resource found: '{rName}', Size: {stream.Length} bytes");
                }
            }


            // 2. Try loading from override SPIR-V path (useful for testing patched versions)
            string? spvOverride = Environment.GetEnvironmentVariable("IPU_L0_LLM_SPV");
            if (!string.IsNullOrWhiteSpace(spvOverride) && File.Exists(spvOverride))
            {
                Console.WriteLine($"[PROBE] Loading override SPIR-V: {spvOverride}");
                byte[] spvBytes = File.ReadAllBytes(spvOverride);
                uint verWord = BitConverter.ToUInt32(spvBytes, 4);
                Console.WriteLine($"[PROBE] Version: v{verWord >> 16}.{(verWord >> 8) & 0xFF}, Size: {spvBytes.Length} bytes");
                try
                {
                    using var probeModule = device.LoadModule(spvBytes, LevelZeroModuleFormat.SpirV);
                    Console.WriteLine("[PROBE] Module loaded successfully — SPIR-V accepted by driver.");
                    Console.WriteLine("[PROBE] This SPIR-V version works. Embed it to fix the suite.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[PROBE] Module load failed: {ex.Message}");
                }
            }

            // 3. Load the LLM Kernel Suite
            using var suite = LlmKernelSuite.Create(device);
            Console.WriteLine("LlmKernelSuite loaded successfully!");

            Console.WriteLine();

            // Run verification tests
            bool allPassed = true;
            allPassed &= VerifyEmbedding(device, suite);
            allPassed &= VerifyRMSNorm(device, suite);
            allPassed &= VerifyGemm(device, suite);
            allPassed &= VerifySwiGLU(device, suite);
            allPassed &= VerifyRoPE(device, suite);
            allPassed &= VerifySDPA(device, suite);
            allPassed &= VerifySampling(device, suite);

            Console.WriteLine();
            if (allPassed)
            {
                Console.WriteLine("OVERALL VERDICT: ALL TESTS PASSED!");
                return 0;
            }
            else
            {
                Console.WriteLine("OVERALL VERDICT: ONE OR MORE TESTS FAILED.");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Critical Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            return -1;
        }
    }

    private static bool VerifyEmbedding(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing Embedding Lookup Kernel... ");
        try
        {
            int hiddenDim = 4;
            int seqLen = 3;

            int[] tokens = { 2, 0, 1 };
            float[] weights = {
                0.1f, 0.2f, 0.3f, 0.4f, // token 0
                0.5f, 0.6f, 0.7f, 0.8f, // token 1
                0.9f, 1.0f, 1.1f, 1.2f  // token 2
            };

            float[] expected = {
                0.9f, 1.0f, 1.1f, 1.2f, // token 2 lookup
                0.1f, 0.2f, 0.3f, 0.4f, // token 0 lookup
                0.5f, 0.6f, 0.7f, 0.8f  // token 1 lookup
            };

            using var tokenBuffer = device.AllocShared(tokens);
            using var weightBuffer = device.AllocShared(weights);
            using var outBuffer = device.AllocShared<float>(seqLen * hiddenDim);

            suite.Embedding.Execute(device, tokenBuffer, weightBuffer, outBuffer, hiddenDim, seqLen * hiddenDim);

            float[] actual = outBuffer.ToArray();

            return AssertArrayClose(expected, actual, "Embedding");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool VerifyRMSNorm(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing RMSNorm Kernel... ");
        try
        {
            int hiddenDim = 4;
            int batchSize = 1;
            float epsilon = 1e-5f;

            float[] x = { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] residual = { 0.5f, 1.0f, 1.5f, 2.0f };
            float[] weight = { 1.0f, 1.0f, 1.0f, 1.0f };

            // x + residual = [1.5, 3.0, 4.5, 6.0]
            // Mean of squares = (2.25 + 9.0 + 20.25 + 36.0) / 4 = 67.5 / 4 = 16.875
            // RMS = sqrt(16.875 + 1e-5) = 4.107931
            // Normalized = [1.5 / RMS, 3.0 / RMS, 4.5 / RMS, 6.0 / RMS]
            //            = [0.365147, 0.730294, 1.095441, 1.460589]
            float[] expected = { 0.365147f, 0.730294f, 1.095441f, 1.460589f };

            using var xBuf = device.AllocShared(x);
            using var resBuf = device.AllocShared(residual);
            using var wBuf = device.AllocShared(weight);
            using var outBuf = device.AllocShared<float>(hiddenDim);

            suite.RMSNorm.Execute(device, xBuf, resBuf, wBuf, outBuf, epsilon, hiddenDim, batchSize, addResidual: true);

            float[] actual = outBuf.ToArray();

            return AssertArrayClose(expected, actual, "RMSNorm");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool VerifyGemm(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing GEMM Kernel... ");
        try
        {
            int m = 2;
            int n = 2;
            int k = 2;

            float[] a = { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] b = { 5.0f, 6.0f, 7.0f, 8.0f };
            float[] expected = {
                19.0f, 22.0f, // 1*5 + 2*7, 1*6 + 2*8
                43.0f, 50.0f  // 3*5 + 4*7, 3*6 + 4*8
            };

            using var aBuf = device.AllocShared(a);
            using var bBuf = device.AllocShared(b);
            using var cBuf = device.AllocShared<float>(m * n);

            suite.Gemm.Execute(device, m, n, k, aBuf, bBuf, cBuf);

            float[] actual = cBuf.ToArray();

            return AssertArrayClose(expected, actual, "GEMM");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool VerifySwiGLU(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing SwiGLU Kernel... ");
        try
        {
            float[] gate = { 1.0f, -1.0f };
            float[] up = { 2.0f, 3.0f };

            // SiLU(1) = 1 / (1 + exp(-1)) = 0.7310586
            // Expected[0] = 0.7310586 * 2 = 1.462117
            // SiLU(-1) = -1 / (1 + exp(1)) = -0.2689414
            // Expected[1] = -0.2689414 * 3 = -0.806824
            float[] expected = { 1.462117f, -0.806824f };

            using var gateBuf = device.AllocShared(gate);
            using var upBuf = device.AllocShared(up);

            suite.SwiGLU.Execute(device, gateBuf, upBuf, gate.Length);

            float[] actual = gateBuf.ToArray();

            return AssertArrayClose(expected, actual, "SwiGLU");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool VerifyRoPE(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing RoPE Kernel... ");
        try
        {
            int headDim = 4;
            int numHeads = 1;
            int numKvHeads = 1;
            int numTokens = 1;
            float ropeBase = 10000.0f;

            float[] q = { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] k = { 1.0f, 2.0f, 3.0f, 4.0f };
            int[] pos = { 1 };

            // dim_idx = 0 -> theta = 10000^0 = 1 -> angle = 1 -> cos = 0.540302, sin = 0.841471
            // q_new[0] = q[0]*cos - q[2]*sin = 1 * cos - 3 * sin = -1.98411
            // q_new[2] = q[2]*cos + q[0]*sin = 3 * cos + 1 * sin = 2.46237
            // dim_idx = 1 -> theta = 10000^-0.5 = 0.01 -> angle = 0.01 -> cos = 0.99995, sin = 0.0099998
            // q_new[1] = q[1]*cos - q[3]*sin = 2 * cos - 4 * sin = 1.9599
            // q_new[3] = q[3]*cos + q[1]*sin = 4 * cos + 2 * sin = 4.0198
            float[] expectedQ = { -1.98411f, 1.9599f, 2.46237f, 4.0198f };

            using var qBuf = device.AllocShared(q);
            using var kBuf = device.AllocShared(k);
            using var posBuf = device.AllocShared(pos);

            suite.RoPE.Execute(device, qBuf, kBuf, posBuf, headDim, numHeads, numKvHeads, ropeBase, numTokens);

            float[] actualQ = qBuf.ToArray();

            return AssertArrayClose(expectedQ, actualQ, "RoPE");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool VerifySDPA(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing SDPA Kernel... ");
        try
        {
            int headDim = 2;
            int numHeads = 1;
            int numKvHeads = 1;
            int seqLen = 2;
            int maxSeqLen = 2;

            float[] q = { 1.0f, 2.0f };
            // kCache shape: [maxSeqLen, numKvHeads, headDim] -> [2, 1, 2]
            float[] kCache = {
                1.0f, 0.0f, // t = 0
                0.0f, 1.0f  // t = 1
            };
            // vCache shape: [2, 1, 2]
            float[] vCache = {
                3.0f, 4.0f, // t = 0
                5.0f, 6.0f  // t = 1
            };

            // Scale = 1 / sqrt(2) = 0.7071067
            // Score_0 = (1*1 + 2*0) * scale = 0.7071067
            // Score_1 = (1*0 + 2*1) * scale = 1.4142135
            // Max = 1.4142135
            // Exp_0 = exp(0.7071067 - 1.4142135) = exp(-0.7071067) = 0.4930687
            // Exp_1 = exp(0.0) = 1.0
            // Sum = 1.4930687
            // Prob_0 = 0.4930687 / 1.4930687 = 0.330238
            // Prob_1 = 1.0 / 1.4930687 = 0.669761
            // Out[0] = Prob_0 * 3.0 + Prob_1 * 5.0 = 0.990715 + 3.348809 = 4.339524
            // Out[1] = Prob_0 * 4.0 + Prob_1 * 6.0 = 1.320953 + 4.018571 = 5.339524
            float[] expected = { 4.339524f, 5.339524f };

            using var qBuf = device.AllocShared(q);
            using var kBuf = device.AllocShared(kCache);
            using var vBuf = device.AllocShared(vCache);
            using var outBuf = device.AllocShared<float>(headDim);

            suite.Attention.Execute(device, qBuf, kBuf, vBuf, outBuf, headDim, numHeads, numKvHeads, seqLen, maxSeqLen);

            float[] actual = outBuf.ToArray();

            return AssertArrayClose(expected, actual, "Attention (SDPA)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool VerifySampling(ComputeDevice device, LlmKernelSuite suite)
    {
        Console.Write("Testing Sampling Kernel... ");
        try
        {
            int vocabSize = 3;
            float[] logits = { 1.0f, 5.0f, 2.0f };

            // 1. Greedy (ArgMax) - Temperature = 0
            using var logitsBuf = device.AllocShared(logits);
            using var outTokenBuf = device.AllocShared<int>(1);

            suite.Sampling.Execute(device, logitsBuf, outTokenBuf, temperature: 0.0f, randomVal: 0.0f, vocabSize: vocabSize);
            int selectedTokenIdx = outTokenBuf.ToArray()[0];

            if (selectedTokenIdx != 1)
            {
                Console.WriteLine($"FAIL Greedy: Expected token 1 (largest logit 5.0), but got {selectedTokenIdx}");
                return false;
            }

            // 2. Random Sampling with Temperature
            // logits scaled by temp 1.0: [1.0, 5.0, 2.0]
            // group_max = 5.0
            // exp(logits - max) = [exp(-4), exp(0), exp(-3)] = [0.0183156, 1.0, 0.049787]
            // sum = 1.0681026
            // probs = [0.0171478, 0.936239, 0.0466127]
            // cumulative = [0.0171478, 0.9533868, 1.0]
            // randomVal = 0.5 -> should pick index 1 (since 0.0171478 < 0.5 <= 0.9533868)
            suite.Sampling.Execute(device, logitsBuf, outTokenBuf, temperature: 1.0f, randomVal: 0.5f, vocabSize: vocabSize);
            int randomSelected = outTokenBuf.ToArray()[0];

            if (randomSelected != 1)
            {
                Console.WriteLine($"FAIL Random: Expected token 1 (cumulative range 0.017 to 0.953 covers random value 0.5), but got {randomSelected}");
                return false;
            }

            Console.WriteLine("PASS");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FAIL with exception: {ex.Message}");
            return false;
        }
    }

    private static bool AssertArrayClose(float[] expected, float[] actual, string testName)
    {
        if (expected.Length != actual.Length)
        {
            Console.WriteLine($"FAIL: Length mismatch. Expected {expected.Length}, got {actual.Length}.");
            return false;
        }

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            if (diff > 1e-4f)
            {
                Console.WriteLine($"FAIL at index {i}. Expected {expected[i]}, got {actual[i]} (diff = {diff}).");
                return false;
            }
        }

        Console.WriteLine("PASS");
        return true;
    }
}
