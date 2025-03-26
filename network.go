package newrelictneuro

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

type Network struct {
	Layers    int             `json:"layers"`
	Neurons   int             `json:"neurons"`
	Iterates  int             `json:"iterates"`
	Net       [][]*Percentron `json:"net"`
	Data      []*TeachData    `json:"teachdata"`
	Score     float64         `json:"score"`
	MpData    interface{}     `json:"mp_data"`
	Error     float64         `json:"error"`
	ErrorList []float64       `json:"error_list"`
	Paralell  bool            `json:"paralell"`
}

func init() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())

}

func (n *Network) AddData(inps, outs []float64) *Network {
	n.Data = append(n.Data, &TeachData{
		Inputs:  inps,
		Outputs: outs,
	})
	return n
}

func CreateNetwork(layer, neurons int) *Network {
	network := &Network{
		Layers:  layer,
		Neurons: neurons,
	}
	return network
}

func (n *Network) Init(data []*TeachData) *Network {
	n.Data = data
	var inPerc []*Percentron
	for i := 0; i < len(n.Data[0].Inputs); i++ {
		inPerc = append(inPerc, &Percentron{In: true})
	}
	n.Net = append(n.Net, inPerc)
	for l := 0; l < n.Layers; l++ {
		var perc []*Percentron
		for s := 0; s < n.Neurons; s++ {
			perc = append(perc, &Percentron{})
		}
		n.Net = append(n.Net, perc)
	}
	var outPerc []*Percentron
	for i := 0; i < len(n.Data[0].Outputs); i++ {
		outPerc = append(outPerc, &Percentron{Out: true})
	}
	n.Net = append(n.Net, outPerc)
	return n.SetWeights()
}

func (n *Network) SetWeights() *Network {
	for i := 0; i < len(n.Net); i++ {
		for s := 0; s < len(n.Net[i]); s++ {
			if !n.Net[i][s].Out {
				for w := 0; w < len(n.Net[i+1]); w++ {
					n.Net[i][s].Weights = append(n.Net[i][s].Weights, randFloat())
				}
			}
		}
	}
	return n
}

func (n *Network) log(i int) {
	log.Println("iterate: ", i, " | error: ", n.Error)
}

func (n *Network) ForwardPredict() []float64 {
	var result []float64
	for i := 0; i < len(n.Net); i++ {
		for s := 0; s < len(n.Net[i]); s++ {
			if n.Net[i][s].In {
				for g := 0; g < len(n.Net[i][s].Weights); g++ {
					n.Net[i+1][g].Input += n.Net[i][s].Output * n.Net[i][s].Weights[g]
				}
			} else {
				if !n.Net[i][s].Out {
					for g := 0; g < len(n.Net[i][s].Weights); g++ {
						n.Net[i+1][g].Input += n.Net[i][s].Output * n.Net[i][s].Weights[g]
					}
				}
			}
		}
		if i < len(n.Net)-1 {
			for s := 0; s < len(n.Net[i+1]); s++ {
				n.Net[i+1][s].sigmoid()
			}
		} else {
			for s := 0; s < len(n.Net[i]); s++ {
				result = append(result, n.Net[i][s].Output)
			}
		}
	}
	return result
}

func (n *Network) ForwardItem(out []float64) float64 {
	var er float64
	for i := 0; i < len(n.Net); i++ {
		for s := 0; s < len(n.Net[i]); s++ {
			if n.Net[i][s].In {
				for g := 0; g < len(n.Net[i][s].Weights); g++ {
					n.Net[i+1][g].Input += n.Net[i][s].Output * n.Net[i][s].Weights[g]
				}
			} else {
				if !n.Net[i][s].Out {
					for g := 0; g < len(n.Net[i][s].Weights); g++ {
						n.Net[i+1][g].Input += n.Net[i][s].Output * n.Net[i][s].Weights[g]
					}
				}
			}
		}
		if i < len(n.Net)-1 {
			for s := 0; s < len(n.Net[i+1]); s++ {
				n.Net[i+1][s].sigmoid()
			}
		} else {
			for s := 0; s < len(n.Net[i]); s++ {
				n.Net[i][s].Error = out[s] - n.Net[i][s].Output
				er += math.Pow(n.Net[i][s].Error, 2)
			}
		}
	}
	return er
}

func (n *Network) Forward() *Network {
	var errors []float64
	for _, tr := range n.Data {
		for i, inp := range tr.Inputs {
			n.Net[0][i].Output = inp
		}
		errComp := n.ForwardItem(tr.Outputs)
		errors = append(errors, errComp)
	}
	n.Error = 0
	for _, er := range errors {
		n.Error += er
	}
	n.Error = n.Error / float64(len(errors))
	return n
}

func (n *Network) Predict(inps []float64) []float64 {
	for i, inp := range inps {
		n.Net[0][i].Output = inp
	}
	return n.ForwardPredict()
}

func (n *Network) Copy() *Network {
	var newNetwork *Network
	n.Data = nil
	bts, _ := json.Marshal(n)
	json.Unmarshal(bts, &newNetwork)
	return newNetwork
}

func (n *Network) Train(logIter int, minPredel float64) *Network {
	n.Forward()
	dataCopy := n.Data
	actNet := n.Copy()
	actNet.Data = dataCopy
	for i := 0; true; i++ {
		var wg sync.WaitGroup
		var mtx sync.Mutex
		var listNew []*Network
		wg.Add(runtime.NumCPU())
		for s := 0; s < runtime.NumCPU(); s++ {
			func() {
				defer wg.Done()
				nNew := actNet.Copy().Mutate()
				nNew.Data = dataCopy
				nNew.Forward()
				mtx.Lock()
				listNew = append(listNew, nNew)
				mtx.Unlock()
			}()
		}
		wg.Wait()
		sort.Slice(listNew, func(i, j int) bool {
			return listNew[i].Error < listNew[j].Error
		})
		if listNew[0].Error < actNet.Error {
			actNet = listNew[0]
		}
		if i%logIter == 0 && i != 1 {
			actNet.log(i)
		}
		if minPredel > actNet.Error {
			return actNet
		}
	}
	return actNet
}

func (n *Network) Mutate() *Network {
	lRand := randInt(len(n.Net) - 1)
	nRand := randInt(len(n.Net[lRand]))
	wRand := randInt(len(n.Net[lRand][nRand].Weights))
	n.Net[lRand][nRand].Weights[wRand] = randFloat()
	return n
}

func randFloat() float64 {
	return -1.0 + rand.Float64()*(1.0-(-1.0))
}

func (n *Network) Debug() {
	bts, err := json.Marshal(n)
	if err != nil {
		log.Fatal("error debug: ", err)
	}
	println(string(bts))
	os.Exit(0)
}

func randInt(max int) int {
	if max == 0 {
		return 0
	}
	return rand.Intn(max)
}
