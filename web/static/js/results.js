'use strict';

class Image extends React.Component {
  render () {
    return (
      <div className="col-2">
        <img 
          className="w-100" 
          src={this.props.source}
          onClick ={(event) => this.props.clickHandler(this.props)}>
        </img>
        <p>Distance = {this.props.distance}</p>
      </div>
    )
  }
}

class LineGraph extends React.Component {

  chartRef = React.createRef();

  componentDidMount() {
    const myChartRef = this.chartRef.current.getContext("2d"); 

    const data = {
      labels: Array.from({length: this.props.data.length}, (v, k) => k+1),
      datasets: [{
        label: this.props.label,
        data: this.props.data,
        backgroundColor: "rgba(255,99,132,0.2)",
        borderColor: "rgba(255,99,132,1)",
      }]
    }

    const options = {
      maintainAspectRatio: false,
      scales: {
        yAxes: [{
          stacked: true,
          gridLines: {
            display: true,
            color: "rgba(255,99,132,0.2)"
          }
        }],
        xAxes: [{
          gridLines: {
            display: false
          }
        }]
      }
    }

    new Chart(myChartRef, {
        type: "line",
        data: data,
        options: options
    });
  }

  render() {
    return <canvas id="myChart" ref={this.chartRef}/>
  }
}

class Modal extends React.Component {
  constructor(props) {
    super(props);
    this.createMetadata = this.createMetadata.bind(this);
    this.createActions = this.createActions.bind(this);
  }

  createMetadata(data) {
    return (
      <div id="modalMetadata">
        <h2>Metadata</h2>
        <h5>Distance: {data.distance}</h5>
        <h5>Dataset: {data.dataset}</h5>
        <h5>Binarized: {data.binarized.toString()}</h5>
        <h5>Index: {data.idx}</h5>
        <h5>Model: {data.model}</h5>
        <h5>Source: <a href={data.source}>{data.source}</a></h5>
      </div>
    )
  }

  createActions(data) {
    return (
      <div id="modalActions">
        <h2>Actions</h2>
        <form action="/query/dataset" method="POST">
          <div className="form-group d-none">
            <input name="query_input" type="text" value={data.source} readOnly/>
            <input name="dataset"  type="text" value={data.dataset} readOnly/>
            <input name="target"  type="text" value={data.idx} readOnly/>
            <input name="model" type="text" value={data.model} readOnly/>
            <input name="binarized" type="text" value={data.model ? 1 : 0} readOnly/>
            <input name="num_results" type="text" value={this.props.num_results} readOnly/>
          </div>
          <button type="submit" className="btn btn-primary">Use As Query</button>
        </form> 
        <button className="btn btn-warning" 
          onClick={(event)=>this.props.clickHandler()}>
          Close Modal</button>
      </div>
    )
  }

  render() {
    if (this.props.show){
      var data = this.props.data;
      return (
        <div id="modal" className="result-modal d-block">
          <div className="result-modal-body">
            <div className="row w-100 my-5 justify-content-center">
              <div className="col-8 text-center">
                <img src={data.source}></img>
              </div>
            </div>
            <div className="row w-100 my-5 justify-content-center">
              <div className="col-3">
                {this.createActions(data)}
              </div>
              <div className="col-5">
                {this.createMetadata(data)}
              </div>
            </div>
          </div>
        </div>
      )
    } else {
      return(<div id="modal" className="result-modal d-none"></div>)
    }
  }
}

class Header extends React.Component {
  constructor(props) {
    super(props);
    this.create_query_input = this.create_query_input.bind(this);
    this.create_tags = this.create_tags.bind(this);
  }

  create_query_input() {
    switch(this.props.data.input_modality) {
      case "text":
        return this.props.data.query_input;
      case "image":
        return <img height='100' src={this.props.data.query_input}/>;
    }  
  }

  create_tags() {
    var tags = "";
    for (var i = 0; i < this.props.data.num_datasets; i++) {
      var result = this.props.data.results[i];
      if (result.modality == "text") {
        for (var j = 0; j < result.num_results; j++) {
          tags = tags + result.data[j] + ", ";
        }
      }
    }
    return tags;
  }

  render() {
    return (
      <div id="header">
        <div className="row justify-content-center">
          <div className="col-8 text-center">
            <h1 className="pt-5">Query:</h1>
          </div>
        </div>
        <div className="row justify-content-center">
          <div className="col-1 text-center">
            <h4>Modality:</h4> {this.props.data.input_modality}<br/>
            <a className="btn btn-primary pt-3" role="button" href="/">Search Another</a>
          </div>
          <div className="col-4 text-center">
            <h4>Input:</h4>
            {this.create_query_input()}
          </div>
          <div className="col-4 text-center">
            <h4>Relevant Tags:</h4>
            {this.create_tags()}
          </div>
        </div>
      </div>
    )
  }
}

class ResultsHeader extends React.Component {
  constructor(props) {
    super(props);
    this.createTabs = this.createTabs.bind(this);
    this.createGraph = this.createGraph.bind(this);
  }

  createGraph(dataset) {
    return <LineGraph label={dataset.dataset} data={dataset.dis}/>
  }

  createTabs() {
    let tabs = [];
    for (var i = 0; i < this.props.datasets.length; i++){
      let dataset = this.props.datasets[i];
      if (dataset.modality != "text"){
        if (dataset.dataset === this.props.currentDataset.dataset) {
          var button_class = "btn btn-primary";
        } else {
          var button_class = "btn btn-secondary";
        }
        tabs.push(
          <button 
            key = {i}
            type="button" 
            className={button_class} 
            onClick ={(event)=>{this.props.changeDatasetHandler(dataset)}}>
            {dataset.dataset} with {dataset.model}
          </button>);
      }
    }
    return tabs
  }

  render() {
    return (
      <div id="resultsHeader">
        <div className="row justify-content-center">
        <div className="col-4 pt-5 text-center">
          <h1>Results:</h1>
          <div className="btn-group" role="group" aria-label="Basic example">
            {this.createTabs()}
          </div>
        </div>
        <div className="col-6 pt-5">
          {this.createGraph(this.props.currentDataset)}
        </div>
      </div>
    </div>
    )
  }

}

class ResultsBody extends React.Component {
  constructor(props) {
    super(props);
    this.createResults = this.createResults.bind(this);
  }

  createResults(){
    var dataset = this.props.dataset;
    let results = [];
    for (var i = 0; i < dataset.num_results; i++){
      results.push(
        <Image 
          key={i}
          source={this.props.url + "/" + dataset.data[i]}
          distance={dataset.dis[i]}
          idx={dataset.idx[i]}
          dataset={dataset.dataset}
          model={dataset.model}
          binarized={dataset.is_binarized}
          clickHandler={this.props.clickHandler}>
        </Image>
      );        
    }
    return results
  }

  render() {
    return (
      <div id="resultsBody">
        <div className="row px-5 pt-5">
          {this.createResults()}
        </div>
      </div>
    )
  }
}
class Results extends React.Component{
  constructor(props) {
    super(props);
    this.changeDatasetHandler = this.changeDatasetHandler.bind(this);
    for (var i = 0; i < this.props.data.num_datasets; i++) {
      var dataset = this.props.data.results[i];
      if (dataset.modality == "image") {
        this.state = {"dataset": dataset};
        break;
      }
    }
  }

  changeDatasetHandler (dataset) {
    this.setState({
      "dataset" : dataset
    });
  }

  render() {
    return (
      <div id="results">
        <ResultsHeader 
          datasets = {this.props.data.results} 
          currentDataset = {this.state.dataset}
          changeDatasetHandler = {this.changeDatasetHandler}/>
        <ResultsBody 
          dataset = {this.state.dataset}
          url = {this.props.data.engine_url}
          clickHandler = {this.props.clickHandler}/>
      </div>
    )
  }
}
class Content extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: JSON.parse(this.props.data),
      modal: null,
      show: false
    };
    this.showModal = this.showModal.bind(this);
    this.hideModal = this.hideModal.bind(this);
  }

  showModal(data){
    this.setState({
      show: true,
      modal: data
    });
  }

  hideModal(){
    this.setState({show: false});
  }

  render() {
    return (
      <div>
        <Header 
          data={this.state.data}/>
        <Results 
          data={this.state.data} 
          clickHandler={this.showModal}/>
        <Modal 
          show={this.state.show} 
          data={this.state.modal} 
          num_results={this.state.data.num_results}
          clickHandler={this.hideModal}/>
      </div>
    )
  }
}
