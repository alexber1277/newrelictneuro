### NewRelictNeuro (FAST Genetic Algorithm Network)

```
go get -u github.com/alexber1277/newrelictneuro
```


```go
package main

import (
	"log"

	nrelic "github.com/alexber1277/newrelictneuro"
)

func main() {

	// teach data
	dataTeach := []*nrelic.TeachData{
		{Inputs: []float64{0, 0, 0}, Outputs: []float64{0}},
		{Inputs: []float64{1, 0, 0}, Outputs: []float64{1}},
		{Inputs: []float64{0, 1, 0}, Outputs: []float64{0}},
		{Inputs: []float64{0, 0, 1}, Outputs: []float64{1}},
	}

	// create net with 2 layer by 30 neurons
	net := nrelic.CreateNetwork(2, 30).Init(dataTeach)

	// get best net. each 1 iterate log, train max error 0.000001
	actNet := net.Train(1, 0.000001)

	// check net
	for _, dt := range dataTeach {
		log.Println("inputs: ", dt.Inputs, " | result: ", actNet.Predict(dt.Inputs))
	}

}

```
`!!! Train perceptron without backpropogation`