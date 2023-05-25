package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"

	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

var (
	modelID  = "model"
	clientID = "client"

	numTrain   = 11982
	numTest    = 1984
	numFeature = 196

	numIter   = 5
	blockSize = 1024
	gamma     = 1.0 / 1024.0                                                                            // learningRate / blockSize
	eta       = [6]float64{0.98990102, -0.00006179, -0.28178332, -0.43406071, -0.53107588, -0.00039170} // weight

	// slot: 2^15, batch: 1024*197 => (1024*32) x 6 + (1024*5)
	numCtPerBlock      = 7  // 13
	numFeaturePerBatch = 32 // 16
	numTestBlock       = 2  // 1024 * 197 + 960 * 197

	c3 = 0.0002
	c1 = -0.0843
	c0 = 0.5 // sigmoid(-x) = c3*x^3 + c1*x + c0

	// Variables for HE setting
	numSlot = int(math.Pow(2, 15))

	PN16QP1761 = ckks.ParametersLiteral{
		LogN:     16,
		LogSlots: 15,
		Q: []uint64{0x80000000080001, 0x2000000a0001, 0x2000000e0001, 0x1fffffc20001, // 55 + 33 x 45
			0x200000440001, 0x200000500001, 0x200000620001, 0x1fffff980001,
			0x2000006a0001, 0x1fffff7e0001, 0x200000860001, 0x200000a60001,
			0x200000aa0001, 0x200000b20001, 0x200000c80001, 0x1fffff360001,
			0x200000e20001, 0x1fffff060001, 0x200000fe0001, 0x1ffffede0001,
			0x1ffffeca0001, 0x1ffffeb40001, 0x200001520001, 0x1ffffe760001,
			0x2000019a0001, 0x1ffffe640001, 0x200001a00001, 0x1ffffe520001,
			0x200001e80001, 0x1ffffe0c0001, 0x1ffffdee0001, 0x200002480001,
			0x1ffffdb60001, 0x200002560001},
		P: []uint64{0x80000000440001, 0x7fffffffba0001}, // 0x80000000500001,

		// 4 x 55
		Scale: 1 << 45,
		Sigma: rlwe.DefaultSigma,
	}
)

type testParams struct {
	params mkckks.Parameters
	ringQ  *ring.Ring
	ringP  *ring.Ring
	prng   utils.PRNG
	kgen   *mkrlwe.KeyGenerator
	skSet  *mkrlwe.SecretKeySet
	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet

	encryptor *mkckks.Encryptor
	decryptor *mkckks.Decryptor
	evaluator *mkckks.Evaluator
	idset     *mkrlwe.IDSet
}

func main() {
	fmt.Println()
	fmt.Println("Reading and Preprocessing Data...")
	const trainFile = "./data/MNIST_train.csv"
	const testFile = "./data/MNIST_test.csv"
	var trainData [][]complex128 // first column is label
	var testData [][]complex128  // first column is label
	trainData = readTrainData(trainFile, numTrain)
	testData = readTestData(testFile, numTest)
	trainData = normalizeData(trainData)
	testData = normalizeData(testData)

	// Setting for HE
	fmt.Println()
	fmt.Println("Setting Parameters...")
	ckks_params, err := ckks.NewParametersFromLiteral(PN16QP1761)
	if err != nil {
		panic(err)
	}
	params := mkckks.NewParameters(ckks_params)

	if err != nil {
		panic(err)
	}

	idset := mkrlwe.NewIDSet()
	idset.Add(modelID)
	idset.Add(clientID)

	var testContext *testParams
	if testContext, err = genTestParams(params, idset); err != nil {
		panic(err)
	}

	// Encrypt Data and Parameters
	fmt.Println()
	fmt.Println("Encrypting Data and Parameters...")

	// Encrypt initial V and W
	zero_msg := mkckks.NewMessage(testContext.params)
	zero_vector := make([]complex128, numSlot)
	for i := 0; i < int(numSlot); i++ {
		zero_vector[i] = complex(0, 0)
	}
	zero_msg.Value = zero_vector
	W := make([]*mkckks.Ciphertext, numCtPerBlock)
	V := make([]*mkckks.Ciphertext, numCtPerBlock)
	for i := 0; i < numCtPerBlock; i++ {
		W[i] = testContext.encryptor.EncryptMsgNew(zero_msg, testContext.pkSet.GetPublicKey(modelID))
		V[i] = testContext.encryptor.EncryptMsgNew(zero_msg, testContext.pkSet.GetPublicKey(modelID))
	}

	// Encrypt Train Data
	Z := make([][]*mkckks.Ciphertext, numIter)
	msg := mkckks.NewMessage(testContext.params)
	msg_vector := make([]complex128, numSlot)

	for a := 0; a < numIter; a++ {
		trainData = shuffleData(trainData)
		Z[a] = make([]*mkckks.Ciphertext, numCtPerBlock)

		for i := 0; i < numCtPerBlock; i++ {
			for j := 0; j < blockSize; j++ {
				for k := 0; k < numFeaturePerBatch; k++ {
					if i*numFeaturePerBatch+k > numFeature {
						msg_vector[j*numFeaturePerBatch+k] = 0
					} else {
						msg_vector[j*numFeaturePerBatch+k] = trainData[j][i*numFeaturePerBatch+k]
					}
				}
			}
			msg.Value = msg_vector
			Z[a][i] = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(modelID))
		}
	}

	// Prepare Constants
	// c1/c3
	for i := 0; i < numSlot; i++ {
		msg_vector[i] = complex(c1/c3, 0)
	}
	msg.Value = msg_vector
	constCt := testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(modelID))

	// c0
	for i := 0; i < numSlot; i++ {
		msg_vector[i] = complex(c0, 0)
	}
	msg.Value = msg_vector
	constCtc0 := testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(modelID))

	// mask
	for i := 0; i < numSlot; i++ {
		if i%numFeaturePerBatch == 0 {
			msg_vector[i] = complex(1, 0)
		} else {
			msg_vector[i] = complex(0, 0)
		}
	}
	msg.Value = msg_vector
	mask := testContext.encryptor.EncodeMsgNew(msg)

	fmt.Println()
	fmt.Println("Training...")
	for a := 0; a < numIter; a++ {
		fmt.Println()
		fmt.Println(a, "-th Iteration")

		start := time.Now()

		M := testContext.evaluator.MulRelinNew(Z[a][0], V[0], testContext.rlkSet)
		M = SumColVec(testContext.evaluator, testContext.rlkSet, testContext.rtkSet, M, mask, blockSize, numFeaturePerBatch)

		for i := 1; i < numCtPerBlock; i++ {
			// M_i = Z_i * V_i
			tmpM := testContext.evaluator.MulRelinNew(Z[a][i], V[i], testContext.rlkSet)

			// M_i = SumColVec(M_i)
			tmpM = SumColVec(testContext.evaluator, testContext.rlkSet, testContext.rtkSet, tmpM, mask, blockSize, numFeaturePerBatch)

			// M = Sum(M_i)
			M = testContext.evaluator.AddNew(M, tmpM)
		}

		// M2 = M * M + c1/c3
		M2 := testContext.evaluator.MulRelinNew(M, M, testContext.rlkSet)
		M2 = testContext.evaluator.AddNew(M2, constCt)

		for i := 0; i < numCtPerBlock; i++ {
			// Z1 = gamma * c0 * Z
			Z1 := testContext.evaluator.MultByConstNew(Z[a][i], gamma*c0)

			// Z3 = gamma * c3 * Z
			Z3 := testContext.evaluator.MultByConstNew(Z[a][i], gamma*c3)

			// M1 = M * Z3
			M1 := testContext.evaluator.MulRelinNew(M, Z3, testContext.rlkSet)

			// G = M1 * M2 + Z1
			G := testContext.evaluator.MulRelinNew(M1, M2, testContext.rlkSet)
			G = testContext.evaluator.AddNew(G, Z1)

			// newW = V + SumRowVec(G)
			G = SumRowVec(testContext.evaluator, testContext.rtkSet, G, blockSize, numFeaturePerBatch)

			newW := testContext.evaluator.AddNew(V[i], G)

			// newV = (1 - eta) * newW + eta * W
			W[i] = testContext.evaluator.MultByConstNew(W[i], eta[a])
			newV := testContext.evaluator.MultByConstNew(newW, 1-eta[a])
			newV = testContext.evaluator.AddNew(newV, W[i])

			///////////////////////////// Update //////////////////////////////////
			W[i] = newW.CopyNew()
			V[i] = newV.CopyNew()
		}
		end := time.Now()
		elapsed := end.Sub(start)
		fmt.Println("Elapsed time(ms): ", elapsed)
	}

	// Inference in ciphertext
	fmt.Println()
	fmt.Println("Encrypting Test Data...")
	testCt := make([][]*mkckks.Ciphertext, numTestBlock)
	for n := 0; n < numTestBlock; n++ {
		testCt[n] = make([]*mkckks.Ciphertext, numCtPerBlock)
	}
	outCt := make([]*mkckks.Ciphertext, numTestBlock)

	for n := 0; n < numTestBlock; n++ {
		for i := 0; i < numCtPerBlock; i++ {
			for j := 0; j < blockSize; j++ {
				for k := 0; k < numFeaturePerBatch; k++ {
					sampleId := n*blockSize + j
					featureId := i*numFeaturePerBatch + k
					msgId := j*numFeaturePerBatch + k
					if featureId == 0 {
						msg_vector[msgId] = complex(1, 0)
					} else if featureId > numFeature || sampleId >= numTest {
						msg_vector[msgId] = 0
					} else {
						msg_vector[msgId] = testData[sampleId][featureId]
					}
				}
			}
			msg.Value = msg_vector
			testCt[n][i] = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(clientID))
		}
	}

	fmt.Println()
	fmt.Println("Testing...")
	start := time.Now()
	for n := 0; n < numTestBlock; n++ {
		M := testContext.evaluator.MulRelinNew(testCt[n][0], W[0], testContext.rlkSet)
		M = SumColVec(testContext.evaluator, testContext.rlkSet, testContext.rtkSet, M, mask, blockSize, numFeaturePerBatch)

		for i := 1; i < numCtPerBlock; i++ {
			// M_i = Z_i * W_i
			tmpM := testContext.evaluator.MulRelinNew(testCt[n][i], W[i], testContext.rlkSet)

			// M_i = SumColVec(M_i)
			tmpM = SumColVec(testContext.evaluator, testContext.rlkSet, testContext.rtkSet, tmpM, mask, blockSize, numFeaturePerBatch)

			// M = Sum(M_i)
			M = testContext.evaluator.AddNew(M, tmpM)
		}

		// M1 = -c3 * M
		M1 := testContext.evaluator.MultByConstNew(M, -c3)

		// M2 = M * M + c1/c3
		M2 := testContext.evaluator.MulRelinNew(M, M, testContext.rlkSet)
		M2 = testContext.evaluator.AddNew(M2, constCt)

		// sigmoid = M1 * M2 + c0
		outCt[n] = testContext.evaluator.MulRelinNew(M1, M2, testContext.rlkSet)
		outCt[n] = testContext.evaluator.AddNew(outCt[n], constCtc0)
	}
	end := time.Now()
	elapsed := end.Sub(start)
	fmt.Println("Elapsed time(ms): ", elapsed)

	fmt.Println()
	fmt.Println("Decrypting Result...")
	outMsg := make([][]complex128, numTestBlock)
	for n := 0; n < numTestBlock; n++ {
		outMsg[n] = testContext.decryptor.Decrypt(outCt[n], testContext.skSet).Value
	}

	correct := 0
	for s := 0; s < numTest; s++ {
		sigmoid := real(outMsg[s/blockSize][(s%blockSize)*numFeaturePerBatch])
		if sigmoid >= 0.5 && real(testData[s][0]) == 1 {
			correct += 1
		} else if sigmoid < 0.5 && real(testData[s][0]) == 0 {
			correct += 1
		}
	}
	fmt.Println()
	fmt.Println("Correct: ", correct)

	/*
		// Inference in plaintext
		fmt.Println()
		fmt.Println("Inference...")
		correct := 0

		Wdec := make([][]complex128, numCtPerBlock)
		for i := 0; i < numCtPerBlock; i++ {
			Wdec[i] = make([]complex128, numFeaturePerBatch)
			Wdec[i] = testContext.decryptor.Decrypt(W[i], testContext.skSet).Value[:numFeaturePerBatch]
		}

		w := make([]float64, numFeature+1)
		for i := 0; i < numFeature+1; i++ {
			w[i] = real(Wdec[i/numFeaturePerBatch][i%numFeaturePerBatch])
		}

		for i := 0; i < numTest; i++ {
			inner_prod := w[0]
			for j := 1; j <= numFeature; j++ {
				inner_prod += w[j] * real(testData[i][j])
			}
			sigmoid := -c3*inner_prod*inner_prod*inner_prod - c1*inner_prod + c0
			if i < 10 {
				fmt.Println(real(testData[i][0]))
				fmt.Println(sigmoid)
			}
			if sigmoid >= 0.5 && real(testData[i][0]) == 1 {
				correct += 1
			} else if sigmoid < 0.5 && real(testData[i][0]) == 0 {
				correct += 1
			}

		}
		fmt.Println("Correct: ", correct)
	*/
	return
}

func SumRowVec(eval *mkckks.Evaluator, rtkSet *mkrlwe.RotationKeySet, ctIn *mkckks.Ciphertext, numRow, numCol int) (ctOut *mkckks.Ciphertext) {
	ctOut = ctIn
	for i := 1; i < numRow; i *= 2 {
		ctOut_rot := eval.RotateNew(ctOut, numCol*i, rtkSet)
		ctOut = eval.AddNew(ctOut, ctOut_rot)
	}
	return
}

func SumColVec(eval *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet, rtkSet *mkrlwe.RotationKeySet, ctIn *mkckks.Ciphertext, mask *ckks.Plaintext, numRow, numCol int) (ctOut *mkckks.Ciphertext) {
	ctOut = ctIn
	for i := 1; i < numCol; i *= 2 {
		ctOut_rot := eval.RotateNew(ctOut, i, rtkSet)
		ctOut = eval.AddNew(ctOut, ctOut_rot)
	}
	ctOut = eval.MulPtxtNew(ctOut, mask)

	for i := 1; i < numCol; i *= 2 {
		ctOut_rot := eval.RotateNew(ctOut, -i, rtkSet)
		ctOut = eval.AddNew(ctOut, ctOut_rot)
	}
	return
}

func genTestParams(defaultParam mkckks.Parameters, idset *mkrlwe.IDSet) (testContext *testParams, err error) {
	testContext = new(testParams)

	testContext.params = defaultParam

	rots := []int{14, 15, 384, 512, 640, 768, 896, 8191, 8190, 8188, 8184}

	for _, rot := range rots {
		testContext.params.AddCRS(rot)
	}

	testContext.kgen = mkckks.NewKeyGenerator(testContext.params)

	testContext.skSet = mkrlwe.NewSecretKeySet()
	testContext.pkSet = mkrlwe.NewPublicKeyKeySet()
	testContext.rlkSet = mkrlwe.NewRelinearizationKeySet(defaultParam.Parameters)
	testContext.rtkSet = mkrlwe.NewRotationKeySet()

	for i := 0; i < testContext.params.LogN()-1; i++ {
		rots = append(rots, 1<<i)
	}

	for id := range idset.Value {
		sk, pk := testContext.kgen.GenKeyPair(id)
		rlk := testContext.kgen.GenRelinearizationKey(sk)

		for _, rot := range rots {
			rk := testContext.kgen.GenRotationKey(rot, sk)
			testContext.rtkSet.AddRotationKey(rk)
		}

		testContext.skSet.AddSecretKey(sk)
		testContext.pkSet.AddPublicKey(pk)
		testContext.rlkSet.AddRelinearizationKey(rlk)
	}

	testContext.ringQ = defaultParam.RingQ()

	if testContext.prng, err = utils.NewPRNG(); err != nil {
		return nil, err
	}

	testContext.encryptor = mkckks.NewEncryptor(testContext.params)
	testContext.decryptor = mkckks.NewDecryptor(testContext.params)
	testContext.evaluator = mkckks.NewEvaluator(testContext.params)

	return testContext, nil
}

func readTrainData(filename string, numData int) [][]complex128 {
	data := make([][]complex128, numData)

	f, err1 := os.Open(filename)
	if err1 != nil {
		panic(err1)
	}

	reader := csv.NewReader(bufio.NewReader(f))
	rows, err2 := reader.ReadAll()
	if err2 != nil {
		panic(err2)
	}

	for i, row := range rows {
		if i == 0 {
			continue
		}
		data[i-1] = make([]complex128, numFeature+1)
		label := float64(0) // label y => 2y - 1

		for j := 0; j < len(row); j++ {
			real_part, err3 := strconv.ParseFloat(row[j], 64)
			if err3 != nil {
				panic(err3)
			}

			if j == 0 {
				label = 2*real_part - 1
				data[i-1][j] = complex(label, 0)
			} else {
				data[i-1][j] = complex(label*real_part, 0)
			}
		}
	}
	return data
}

func readTestData(filename string, numData int) [][]complex128 {
	data := make([][]complex128, numData)

	f, err1 := os.Open(filename)
	if err1 != nil {
		panic(err1)
	}

	reader := csv.NewReader(bufio.NewReader(f))
	rows, err2 := reader.ReadAll()
	if err2 != nil {
		panic(err2)
	}

	for i, row := range rows {
		if i == 0 {
			continue
		}
		data[i-1] = make([]complex128, numFeature+1)

		for j := 0; j < len(row); j++ {
			real_part, err3 := strconv.ParseFloat(row[j], 64)
			if err3 != nil {
				panic(err3)
			}
			data[i-1][j] = complex(real_part, 0)
		}
	}
	return data
}

func normalizeData(data [][]complex128) [][]complex128 {
	for i, d := range data {
		max := float64(-1)
		for j := 1; j <= numFeature; j++ {
			if max < math.Abs(real(d[j])) {
				max = math.Abs(real(d[j]))
			}
		}
		for j := 1; j <= numFeature; j++ {
			data[i][j] = complex(real(data[i][j])/max, 0)
		}
	}

	return data
}

func shuffleData(data [][]complex128) [][]complex128 {
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < len(data); i++ {
		i_rand := rand.Intn(i + 1)
		data[i], data[i_rand] = data[i_rand], data[i]
	}
	return data
}
