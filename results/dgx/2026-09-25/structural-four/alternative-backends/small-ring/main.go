// Experimental two-ring encrypted compute/refresh boundary. No model speed claim.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"math"
	"math/cmplx"
	"os"
	"runtime"
	"time"
)

func must(err error) {
	if err != nil {
		panic(err)
	}
}
func main() {
	logN := flag.Int("logn", 15, "ordinary ring log degree; refresh uses 16")
	output := flag.String("output", "small-ring.json", "report path")
	flag.Parse()
	runtime.GOMAXPROCS(4)
	started := time.Now()
	report := map[string]any{"passed": false, "backend": "lattigo-v6.2.0-cpu", "ordinary_log_n": *logN, "refresh_log_n": 16, "scale_bits": 59, "bootstrap_passes": 2, "requested_residual_multiplier": 4096, "intermediate_evaluator_decryptions": 0, "security": "not-set", "tolerance": 1e-6}
	defer func() {
		if failure := recover(); failure != nil {
			report["error"] = fmt.Sprint(failure)
		}
		report["process_seconds"] = time.Since(started).Seconds()
		data, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			panic(err)
		}
		must(os.WriteFile(*output, append(data, '\n'), 0644))
		fmt.Println(string(data))
		if !report["passed"].(bool) {
			os.Exit(2)
		}
	}()
	logQ := []int{60}
	for i := 0; i < 17; i++ {
		logQ = append(logQ, 59)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: *logN, LogNthRoot: 17, LogQ: logQ, LogP: []int{61, 61, 61}, LogDefaultScale: 59, Xs: ring.Ternary{P: 2.0 / 3.0}})
	must(err)
	bp, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(16), LogP: []int{61, 61, 61, 61}, Xs: params.Xs(), EphemeralSecretWeight: utils.Pointy(0),
		// Dense ternary secrets need a wider modular-reduction interval than
		// the default K=16 intended for sparse ephemeral secrets.
		Mod1Type: mod1.CosContinuous, K: utils.Pointy(512), Mod1Degree: utils.Pointy(127), DoubleAngle: utils.Pointy(6),
		IterationsParameters: &bootstrapping.IterationsParameters{BootstrappingPrecision: []float64{12}, ReservedPrimeBitSize: 28},
	})
	must(err)
	report["modular_reduction"] = map[string]any{"type": "cos-continuous", "k": 512, "degree": 127, "double_angles": 6, "ephemeral_secret_weight": 0}
	report["ordinary_log_qp"] = params.LogQP()
	report["refresh_log_qp"] = bp.BootstrappingParameters.LogQP()
	report["ordinary_q"] = params.Q()
	report["refresh_q"] = bp.BootstrappingParameters.Q()
	keygen := rlwe.NewKeyGenerator(params)
	sk, pk := keygen.GenKeyPairNew()
	bootKeys, _, err := bp.GenEvaluationKeys(sk)
	must(err)
	ordinary := ckks.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(keygen.GenRelinearizationKeyNew(sk)))
	refresh, err := bootstrapping.NewEvaluator(bp, bootKeys)
	must(err)
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	report["setup_seconds"] = time.Since(started).Seconds()
	worst := 0.0
	samples := []map[string]any{}
	for _, width := range []int{1, 32, 768, 1536} {
		values := make([]complex128, params.MaxSlots())
		expected := make([]complex128, len(values))
		for i := 0; i < width; i++ {
			values[i] = complex(math.Sin(float64(i)*.7+.1)*.8, 0)
			expected[i] = values[i] * values[i]
		}
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		must(encoder.Encode(values, pt))
		ct, err := encryptor.EncryptNew(pt)
		must(err)
		before := time.Now()
		product, err := ordinary.MulRelinNew(ct, ct)
		must(err)
		must(ordinary.Rescale(product, product))
		ordinarySeconds := time.Since(before).Seconds()
		// Includes encrypted switching into the larger ring, both refresh passes,
		// and switching back. Only final client validation decrypts.
		ordinary.DropLevel(product, product.Level()-1)
		before = time.Now()
		refreshed, err := refresh.Bootstrap(product)
		must(err)
		refreshSeconds := time.Since(before).Seconds()
		before = time.Now()
		result, err := ordinary.MulRelinNew(refreshed, refreshed)
		must(err)
		must(ordinary.Rescale(result, result))
		ordinarySeconds += time.Since(before).Seconds()
		decoded := make([]complex128, len(values))
		must(encoder.Decode(decryptor.DecryptNew(result), decoded))
		local := 0.0
		for i := range expected {
			err := cmplx.Abs(decoded[i] - expected[i]*expected[i])
			if math.IsNaN(err) || math.IsInf(err, 0) {
				panic("non-finite output")
			}
			local = math.Max(local, err)
		}
		worst = math.Max(worst, local)
		samples = append(samples, map[string]any{"width": width, "ordinary_seconds": ordinarySeconds, "refresh_boundary_seconds": refreshSeconds, "max_abs_error": local, "output_level": result.Level()})
		report["samples"] = samples
		fmt.Printf("width=%d ordinary=%f refresh=%f error=%g\n", width, ordinarySeconds, refreshSeconds, local)
	}
	report["max_abs_error"] = worst
	report["passed"] = worst <= 1e-6
}
