package newrelictneuro

import (
	"log"
)

func example() {

	dataTeach := []*TeachData{
		{Inputs: []float64{0, 0, 0}, Outputs: []float64{0}},
		{Inputs: []float64{1, 0, 0}, Outputs: []float64{1}},
		{Inputs: []float64{0, 1, 0}, Outputs: []float64{0}},
		{Inputs: []float64{0, 0, 1}, Outputs: []float64{1}},
	}

	net := CreateNetwork(2, 30).Init(dataTeach)
	actNet := net.Train(1, 0.000001)

	for _, dt := range dataTeach {
		log.Println("inputs: ", dt.Inputs, " | result: ", actNet.Predict(dt.Inputs))
	}

	//log.Println("inputs: ", []float64{1, 0, 1}, " | result: ", actNet.Predict([]float64{1, 0, 1}))

}
