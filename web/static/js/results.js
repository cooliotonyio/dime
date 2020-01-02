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
        {/* <p>Distance = {this.props.distance}</p> */}
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
        <p>
          Distance: {data.distance}<br/>
          Dataset: {data.dataset}<br/>
          Binarized: {data.binarized.toString()}<br/>
          Index: {data.idx}<br/>
          Model: {data.model}<br/>
          Model Desc: {data.model_info.desc}<br/>
          Model Output Dim: {data.model_info.output_dimension}<br/>
          Source: <a href={data.source}>{data.source}</a></p>
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
    this.set_tags = this.set_tags.bind(this);
    this.state = {
      tags: ["Loading tags..."]
    }
  }

  set_tags() {
    let form_data = new FormData();
    form_data.append("modality", this.props.modality)
    form_data.append("target", this.props.target)
    form_data.append("num_results", "30")
    fetch(this.props.server_url + "/query", {
      method: "POST",
      body: form_data})
      .then(r => r.json())
      .then((response) => {
        this.setState({
          tags: response.results.results
        });
      })
  }

  componentDidMount() {
    this.set_tags();
  }

  create_query_input() {
    switch(this.props.modality) {
      case "text":
        return this.props.target;
      case "image":
        return <img height='100' src={this.props.server_url+"/"+this.props.target}/>;
    }  
  }

  create_tags() {
    console.log(this.state)
    var tags = "";
    for (var i = 0; i < this.state.tags.length; i++) {
      tags = tags + this.state.tags[i] + ", ";
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
            <h4>Modality:</h4> {this.props.modality}<br/>
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
      if (dataset[3] != "text"){
        if (dataset[0] === this.props.currentDataset.dataset && dataset[1] === this.props.currentDataset.model) {
          var button_class = "btn btn-primary";
        } else {
          var button_class = "btn btn-secondary";
        } 
        tabs.push(
          <form action="/query/prev_query" method="POST" key = {i}>
            <div className="form-group d-none">
              <input name="query_input" type="text" value={this.props.query_input} readOnly/>
              <input name="dataset"  type="text" value={dataset[0]} readOnly/>
              <input name="model"  type="text" value={dataset[1]} readOnly/>
              <input name="binarized" type="text" value={dataset[2]} readOnly/>
              <input name="modality" type="text" value={this.props.input_modality} readOnly/>
              <input name="num_results" type="text" value={this.props.num_results} readOnly/>
            </div>
            <button 
              type="submit" 
              className={button_class}>
              {dataset[0]} with {dataset[1]}
            </button>
          </form>);
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
          model_info={dataset.model_info}
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
    for (var i = 0; i < this.props.data.results.length; i++) {
      var dataset = this.props.data.results[i];
      if (dataset.modality == "image") {
        this.state = {"dataset": dataset};
        break;
      }
    }
  }

  render() {
    return (
      <div id="results">
        <ResultsHeader 
          datasets = {this.props.data.valid_indexes} 
          currentDataset = {this.state.dataset}
          query_input = {this.props.data.query_input}
          input_modality = {this.props.data.input_modality}/>
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
    const data = JSON.parse(this.props.data);
    this.state = {
      modal: null,
      show: false,
      server_url: data.server_url,
      target: data.target,
      modality: data.modality,
    };
    this.showModal = this.showModal.bind(this);
    this.hideModal = this.hideModal.bind(this);
    this.get_results = this.get_results.bind(this);
  }

  get_results(data) {
    let form_data = new FormData();
    form_data.append("modality", data.modality)
    form_data.append("target", data.target)
    form_data.append("index_name", data.index_name)
    form_data.append("num_results", data.num_results)
    fetch(data.server_url + "/query", {
      method: "POST",
      body: form_data})
      .then(r => r.json())
      .then((response) => {
        this.setState({
          target: response.results.target,
          modality: response.results.modality,
          num_results: response.results.results.length
        });
      })
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
          target={this.state.target}
          modality={this.state.modality}
          server_url={this.state.server_url}/>
        {/* <Results 
          data={this.state} 
          clickHandler={this.showModal}/>
        <Modal 
          data={this.state} 
          clickHandler={this.hideModal}/> */}
      </div>
    )
  }
}
