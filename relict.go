package newrelictneuro

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
)

type RelictNode struct {
	Inps []float64 `json:"inps"`
	Out  float64   `json:"out"`
	Net  *Network  `json:"net"`
}

type RelictResult struct {
	Score float64 `json:"score"`
	Best  int     `json:"best"`
	Bad   int     `json:"bad"`
}

type Relict struct {
	Layers    int             `json:"layers"`
	Nodes     int             `json:"nodes"`
	Error     float64         `json:"error"`
	Break     bool            `json:"break"`
	Result    RelictResult    `json:"result"`
	Nets      [][]*RelictNode `json:"nets"`
	Data      []*TeachData    `json:"data"`
	BestCount int             `json:"best_count"`
}

type NeuroConf struct {
	NetLayer    int    `json:"net_layer"`
	NetNeurons  int    `json:"net_neurons"`
	NetPops     int    `json:"net_pops"`
	NetLastBest int    `json:"net_last_best"`
	FileDump    string `json:"file_dump"`
}

var (
	PopulationRelict = 8
	LastBestRelict   = 1
	NetLayerRelict   = 1
	NetNeuronsRelict = 5
	ListRelict       []*Relict
	FileDump         string
)

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UnixNano())
}

func InitRelict(layers int, nodes int, data []*TeachData, conf ...NeuroConf) *Relict {
	r := Relict{
		Layers: layers,
		Nodes:  nodes,
		Data:   data,
		Nets:   [][]*RelictNode{},
	}
	if len(conf) > 0 {
		NetLayerRelict = conf[0].NetLayer
		NetNeuronsRelict = conf[0].NetNeurons
		PopulationRelict = conf[0].NetPops
		LastBestRelict = conf[0].NetLastBest
		FileDump = conf[0].FileDump
	}
	rel, ok := LoadDump()
	if !ok {
		return r.InitNets()
	}
	return rel
}

func (r *Relict) InitNets() *Relict {
	for i := 0; i < r.Layers; i++ {
		var nets []*RelictNode
		for s := 0; s < r.Nodes; s++ {
			if i == 0 {
				var nn RelictNode
				nn.Inps = make([]float64, len(r.Data[0].Inputs))
				nn.Out = 0
				dts := &TeachData{
					Inputs:  make([]float64, len(r.Data[0].Inputs)),
					Outputs: []float64{0},
				}
				nn.Net = CreateNetwork(NetLayerRelict, NetNeuronsRelict).Init([]*TeachData{dts})
				nets = append(nets, &nn)
				continue
			}
			var nn RelictNode
			nn.Inps = make([]float64, len(r.Nets[i-1]))
			nn.Out = 0
			dts := &TeachData{
				Inputs:  make([]float64, len(r.Nets[i-1])),
				Outputs: []float64{0},
			}
			nn.Net = CreateNetwork(NetLayerRelict, NetNeuronsRelict).Init([]*TeachData{dts})
			nets = append(nets, &nn)
		}
		r.Nets = append(r.Nets, nets)
	}

	var nn RelictNode
	nn.Inps = make([]float64, len(r.Nets[len(r.Nets)-1]))
	nn.Out = 0
	dts := &TeachData{
		Inputs:  make([]float64, len(r.Nets[len(r.Nets)-1])),
		Outputs: []float64{0},
	}
	nn.Net = CreateNetwork(NetLayerRelict, NetNeuronsRelict).Init([]*TeachData{dts})
	nnList := []*RelictNode{&nn}
	r.Nets = append(r.Nets, nnList)
	return r
}

func (r *Relict) Predict(inps []float64) []float64 {
	var results []float64
	for i := 0; i < len(r.Nets); i++ {
		if i == 0 {
			for j := 0; j < len(r.Nets[i]); j++ {
				result := r.Nets[i][j].Net.Predict(inps)
				r.Nets[i][j].Out = result[0]
			}
			continue
		}
		var baseInps []float64
		for j := 0; j < len(r.Nets[i-1]); j++ {
			baseInps = append(baseInps, r.Nets[i-1][j].Out)
		}
		for j := 0; j < len(r.Nets[i]); j++ {
			result := r.Nets[i][j].Net.Predict(baseInps)
			r.Nets[i][j].Out = result[0]
		}
		if i >= len(r.Nets)-1 {
			for j := 0; j < len(r.Nets[i]); j++ {
				results = append(results, r.Nets[i][j].Out)
			}
		}
	}
	return results
}

func (r *Relict) Mutate() *Relict {
	lrRand := randIntMinMax(0, len(r.Nets)-1)
	nrRand := randIntMinMax(0, len(r.Nets[lrRand])-1)
	r.Nets[lrRand][nrRand].Net.Mutate()
	return r
}

func (r *Relict) Copy() *Relict {
	var newRelict Relict
	copyData := r.Data
	r.Data = nil
	bts, err := json.Marshal(r)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(bts, &newRelict); err != nil {
		log.Fatal(err)
	}
	r.Data = copyData
	newRelict.Data = copyData
	return &newRelict
}

func randIntMinMax(min, max int) int {
	return rand.Intn(max-min+1) + min
}

func (res *RelictResult) Clear() {
	res.Score = 0
	res.Best = 0
	res.Bad = 0
}

func (r *Relict) Train(fn func(net *Relict)) *Relict {
	var bestCount int
	for i := 0; i < len(r.Data); i++ {
		if r.Data[i].Outputs[0] == 1 {
			bestCount += 1
		}
	}
	newRelict := r.Copy()
	for i := 0; i < PopulationRelict; i++ {
		cp := newRelict.Copy().Mutate()
		cp.BestCount = bestCount
		ListRelict = append(ListRelict, cp)
	}
	for iter := 0; true; iter++ {
		var wg sync.WaitGroup
		for _, nr := range ListRelict {
			wg.Add(1)
			go func(nnr *Relict) {
				defer wg.Done()
				nnr.Result.Clear()
				fn(nnr)
			}(nr)
		}
		wg.Wait()
		sort.Slice(ListRelict, func(i, j int) bool {
			return ListRelict[i].Result.Score > ListRelict[j].Result.Score
		})
		ListRelict[0].LogIteration(iter)
		if iter%100 == 0 && iter >= 100 {
			ListRelict[0].SaveDump()
			log.Println("save dump")
		}
		if ListRelict[0].Break {
			break
		}
		ListRelict = ListRelict[:LastBestRelict]
		r.AddPopsAndMutate()
	}
	return ListRelict[0]
}

func LoadDump() (*Relict, bool) {
	var rel *Relict
	bts, err := ioutil.ReadFile(FileDump)
	if err != nil {
		return rel, false
	}
	if err := json.Unmarshal(bts, &rel); err != nil {
		log.Println(err)
		return rel, false
	}
	return rel, true
}

func (r *Relict) SaveDump() {
	bts, err := json.Marshal(r)
	if err != nil {
		log.Println(err)
		return
	}
	if err := ioutil.WriteFile(FileDump, bts, 0644); err != nil {
		log.Println(err)
	}
}

func (r *Relict) LogIteration(iter int) {
	log.Println("iteration:", iter, "; ",
		"length:", len(ListRelict), "; ",
		"best_count:", r.BestCount, "; ",
		"best:", r.Result.Best, "; ",
		"bad:", r.Result.Bad, "; ",
		"score:", r.Result.Score, "; ",
	)
}

func (r *Relict) AddPopsAndMutate() {
	for {
		var newList []*Relict
		for _, nr := range ListRelict {
			newList = append(newList, nr.Copy().Mutate())
		}
		ListRelict = append(ListRelict, newList...)
		if len(ListRelict) >= PopulationRelict {
			return
		}
	}
}

func toStr(in interface{}) string {
	bts, err := json.Marshal(in)
	if err != nil {
		log.Fatal(err)
	}
	return string(bts)
}

func RandFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func roundFloat(val float64, precision int) float64 {
	factor := math.Pow(10, float64(precision))
	return math.Round(val*factor) / factor
}
