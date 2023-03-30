package mkbfv

import "mk-lattigo/mkrlwe"
import "github.com/ldsec/lattigo/v2/ring"

//output is in InvNTTForm
func (ks *KeySwitcher) ExternalProductBFVHoisted(levelQ int, aHoisted1, aHoisted2, bg1, bg2 *mkrlwe.SwitchingKey, c *ring.Poly) {
	params := ks.params
	ringQ := params.RingQ()
	ringP := params.RingP()
	ringQP := params.RingQP()

	levelP := params.PCount() - 1
	beta := params.Beta(levelQ)

	c1QP := ks.Pool[1]

	//ks.DecomposeBFV(levelQ, aR, ks.swkPool1, ks.swkPool2)

	// Key switching with CRT decomposition for the Qi
	for i := 0; i < beta; i++ {
		if i == 0 {
			ringQP.MulCoeffsMontgomeryLvl(levelQ, levelP, bg1.Value[i], aHoisted1.Value[i], c1QP)
			ringQP.MulCoeffsMontgomeryAndAddLvl(levelQ, levelP, bg2.Value[i], aHoisted2.Value[i], c1QP)
		} else {
			ringQP.MulCoeffsMontgomeryAndAddLvl(levelQ, levelP, bg1.Value[i], aHoisted1.Value[i], c1QP)
			ringQP.MulCoeffsMontgomeryAndAddLvl(levelQ, levelP, bg2.Value[i], aHoisted2.Value[i], c1QP)
		}
	}

	ringQ.InvNTTLazyLvl(levelQ, c1QP.Q, c1QP.Q)
	ringP.InvNTTLazyLvl(levelP, c1QP.P, c1QP.P)

	ks.Baseconverter.ModDownQPtoQ(levelQ, levelP, c1QP.Q, c1QP.P, c)
}

// MulRelin multiplies op0 with op1 with relinearization and returns the result in ctOut.
// Input ciphertext should be in NTT form
func (ks *KeySwitcher) MulAndRelinBFVHoisted(op0, op1 *mkrlwe.Ciphertext,
	op0Hoisted1, op0Hoisted2 *mkrlwe.HoistedCiphertext,
	op1Hoisted1, op1Hoisted2 *mkrlwe.HoistedCiphertext,
	rlkSet *RelinearizationKeySet, ctOut *mkrlwe.Ciphertext) {

	level := ctOut.Level()

	if op0.Level() < ctOut.Level() {
		panic("Cannot MulAndRelin: op0 and op1 have different levels")
	}

	if ctOut.Level() < level {
		panic("Cannot MulAndRelin: op0 and ctOut have different levels")
	}

	params := ks.params
	conv := ks.conv
	ringQP := params.RingQP()
	ringQ := params.RingQ()
	ringR := params.RingR()

	idset0 := op0.IDSet()
	idset1 := op1.IDSet()

	levelP := params.PCount() - 1
	beta := params.Beta(level)

	x1 := ks.swkPool3
	x2 := ks.swkPool4
	y1 := ks.swkPool5
	y2 := ks.swkPool6

	//initialize x1, x2, y1, y2
	for i := 0; i < beta; i++ {
		x1.Value[i].Q.Zero()
		x1.Value[i].P.Zero()

		y1.Value[i].Q.Zero()
		y1.Value[i].P.Zero()

		x2.Value[i].Q.Zero()
		x2.Value[i].P.Zero()

		y2.Value[i].Q.Zero()
		y2.Value[i].P.Zero()
	}

	//gen x vector
	for id := range idset0.Value {
		if op0Hoisted1 == nil {
			ks.DecomposeBFV(level, op0.Value[id], ks.swkPool1, ks.swkPool2)
			d1 := rlkSet.Value[id].Value[0].Value[1]
			d2 := rlkSet.Value[id].Value[1].Value[1]
			for i := 0; i < beta; i++ {
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, d1.Value[i], ks.swkPool1.Value[i], x1.Value[i])
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, d2.Value[i], ks.swkPool2.Value[i], x2.Value[i])
			}
		} else {
			d1 := rlkSet.Value[id].Value[0].Value[1]
			d2 := rlkSet.Value[id].Value[1].Value[1]
			for i := 0; i < beta; i++ {
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, d1.Value[i], op0Hoisted1.Value[id].Value[i], x1.Value[i])
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, d2.Value[i], op0Hoisted2.Value[id].Value[i], x2.Value[i])
			}
		}
	}

	for i := 0; i < beta; i++ {
		ringQP.MFormLvl(level, levelP, x1.Value[i], x1.Value[i])
		ringQP.MFormLvl(level, levelP, x2.Value[i], x2.Value[i])
	}

	//gen y vector
	for id := range idset1.Value {
		if op1Hoisted1 == nil {
			ks.DecomposeBFV(level, op1.Value[id], ks.swkPool1, ks.swkPool2)
			b1 := rlkSet.Value[id].Value[0].Value[0]
			b2 := rlkSet.Value[id].Value[1].Value[0]
			for i := 0; i < beta; i++ {
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, b1.Value[i], ks.swkPool1.Value[i], y1.Value[i])
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, b2.Value[i], ks.swkPool2.Value[i], y2.Value[i])
			}
		} else {
			b1 := rlkSet.Value[id].Value[0].Value[0]
			b2 := rlkSet.Value[id].Value[1].Value[0]
			for i := 0; i < beta; i++ {
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, b1.Value[i], op1Hoisted1.Value[id].Value[i], y1.Value[i])
				ringQP.MulCoeffsMontgomeryAndAddLvl(level, levelP, b2.Value[i], op1Hoisted2.Value[id].Value[i], y2.Value[i])
			}
		}
	}

	for i := 0; i < beta; i++ {
		ringQP.MFormLvl(level, levelP, y1.Value[i], y1.Value[i])
		ringQP.MFormLvl(level, levelP, y2.Value[i], y2.Value[i])
	}

	//ctOut_0 <- op0_0 * op1_0
	ringR.NTT(op0.Value["0"], ks.polyRPool1)
	ringR.NTT(op1.Value["0"], ks.polyRPool2)

	ringR.MForm(ks.polyRPool1, ks.polyRPool1)
	ringR.MulCoeffsMontgomery(ks.polyRPool1, ks.polyRPool2, ks.polyRPool3)
	conv.Quantize(ks.polyRPool3, ctOut.Value["0"], params.T())

	//ctOut_j <- op0_0 * op1_j + op0_j * op1_0
	ringR.MForm(ks.polyRPool2, ks.polyRPool2)

	for id := range idset0.Value {
		if !idset1.Has(id) {
			ringR.NTT(op0.Value[id], ks.polyRPool3)
			ringR.MulCoeffsMontgomery(ks.polyRPool2, ks.polyRPool3, ks.polyRPool3)
			conv.Quantize(ks.polyRPool3, ctOut.Value[id], params.T())
		}
	}

	for id := range idset1.Value {
		if !idset0.Has(id) {
			ringR.NTT(op1.Value[id], ks.polyRPool3)
			ringR.MulCoeffsMontgomery(ks.polyRPool1, ks.polyRPool3, ks.polyRPool3)
			conv.Quantize(ks.polyRPool3, ctOut.Value[id], params.T())
		}
	}

	for id := range idset1.Value {
		if idset0.Has(id) {
			ringR.NTT(op1.Value[id], ks.polyRPool3)
			ringR.MulCoeffsMontgomery(ks.polyRPool1, ks.polyRPool3, ks.polyRPool3)

			ringR.NTT(op0.Value[id], ks.polyRPool4)
			ringR.MulCoeffsMontgomeryAndAdd(ks.polyRPool2, ks.polyRPool4, ks.polyRPool3)

			conv.Quantize(ks.polyRPool3, ctOut.Value[id], params.T())
		}
	}

	//ctOut_j <- ctOut_j +  Inter(op1_j, x)
	for id := range idset1.Value {
		if op1Hoisted1 == nil {
			ks.ExternalProductBFV(level, op1.Value[id], x1, x2, ks.polyQPool1)
		} else {
			ks.ExternalProductBFVHoisted(level, op1Hoisted1.Value[id], op1Hoisted2.Value[id], x1, x2, ks.polyQPool1)
		}
		ringQ.AddLvl(level, ctOut.Value[id], ks.polyQPool1, ctOut.Value[id])
	}

	//ctOut_0 <- ctOut_0 + Inter(Inter(op0_i, y), v_i)
	//ctOut_i <- ctOut_i + Inter(Inter(op0_i, y), u)

	u := params.CRS[-1]

	for id := range idset0.Value {
		v := rlkSet.Value[id].Value[0].Value[2]

		if op0Hoisted1 == nil {
			ks.ExternalProductBFV(level, op0.Value[id], y1, y2, ks.polyQPool1)
		} else {
			ks.ExternalProductBFVHoisted(level, op0Hoisted1.Value[id], op0Hoisted2.Value[id], y1, y2, ks.polyQPool1)
		}

		ks.Decompose(level, ks.polyQPool1, ks.swkPool3)

		ks.ExternalProductHoisted(level, ks.swkPool3, v, ks.polyQPool2)
		ringQ.AddLvl(level, ctOut.Value["0"], ks.polyQPool2, ctOut.Value["0"])

		ks.ExternalProductHoisted(level, ks.swkPool3, u, ks.polyQPool2)
		ringQ.AddLvl(level, ctOut.Value[id], ks.polyQPool2, ctOut.Value[id])
	}
}
