package newrelictneuro

import (
	"math"
)

type Percentron struct {
	Input   float64   `json:"input"`
	Output  float64   `json:"output"`
	Target  float64   `json:"target"`
	Error   float64   `json:"error"`
	Weights []float64 `json:"weights"`
	In      bool      `json:"in"`
	Out     bool      `json:"out"`
}

func (p *Percentron) sigmoid() {
	p.Output = 1 / (1 + math.Exp(-p.Input))
	p.Input = 0
}

func SigmoidPrime(x float64) float64 {
	return x * (1 - x)
}
