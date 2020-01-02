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
    this.handle_tab_press = this.handle_tab_press.bind(this);

    this.state = {
      available_indexes: null,
    }
  }

  createGraph(results) {
    return <LineGraph label={results.dataset_name} data={results.dis}/>
  }

  componentDidMount(){
    const data = {available_indexes: this.props.results.modality};
    const url = this.props.server_url + "/info?" + $.param(data);
    fetch(url)
      .then(result => result.json())
      .then(
        (result) => {
          console.log(result)
          this.setState({available_indexes: result.available_indexes})},
        (error) => {
          console.log(error);
          this.setState({available_indexes: []});
        }
      );
  }

  handle_tab_press(index_name) {
    const data = {
      target: this.props.results.target,
      modality: this.props.results.modality,
      num_results: this.props.results.num_results,
      index_name: index_name,
    }
    window.location.href = window.location.origin + "/query?" + $.param(data);
  }

  createTabs() {
    console.log(this.state)
    var tabs = []
    if (this.state.available_indexes !== null) {
      for (let [i, index_name] of this.state.available_indexes.entries()) {
        if (index_name === this.props.results.index_name){
          tabs.push(
            <button 
              key={i}
              type="button"
              className="btn btn-primary disabled">
              {index_name}
            </button>)
        } else {
          tabs.push(
            <button 
              key={i}
              type="button"
              className="btn btn-secondary"
              onClick={(e) => this.handle_tab_press(index_name)}>
              {index_name}
            </button>)
        }
      }
    } else {
      tabs = (<div>Loading indexes...</div>)
    }
    return tabs
  }

  render() {
    return (
      <div id="resultsHeader">
        <div className="row pt-5">
          <div className="col text-center">
            <h1>Results:</h1>
          </div>
        </div>
        <div className="row justify-content-center">
        <div className="col-5 text-center">
          <div className="btn-group" role="group" aria-label="Basic example">
            {this.createTabs()}
          </div>
        </div>
        <div className="col-5">
          {this.createGraph(this.props.results)}
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
    var results = []
    if ("image" === this.props.results.index_modality) {
      for (let [i, result] of this.props.results.results.entries()){
        results.push(
          <Image
            key={i}
            source={this.props.server_url + "/" + result}
            clickHandler={this.props.clickHandler}/>
          )
      }
    } else if ("text" === this.props.results.index_modality) {
      for (let [i, result] of this.props.results.results.entries()){
        results.push(
          <div className="col-2" key={i}>
            <h4>{i+1}. {result}</h4>
          </div>)
      }
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
  }

  render() {
    return (
      <div id="results">
        <ResultsHeader 
          results = {this.props.results}
          server_url = {this.props.server_url}/>
        <ResultsBody 
          results = {this.props.results}
          server_url = {this.props.server_url}
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
      results: null,
    };
    this.showModal = this.showModal.bind(this);
    this.hideModal = this.hideModal.bind(this);
    this.get_results = this.get_results.bind(this);
    this.render_results = this.render_results.bind(this);
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

  componentDidMount() {
    this.get_results(JSON.parse(this.props.data));
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
          num_results: response.results.results.length,
          results: response.results
        });
      })
  }

  render_results() {
    if (this.state.results !== null) {
      return (
        <Results 
          results={this.state.results}
          server_url={this.state.server_url} 
          clickHandler={this.showModal}/>);
    } else {
      return (
        <div className="row w-100 py-5">
          <div className="col text-center">
            <h1>Loading results...</h1>
          </div>
        </div>
      );
    }
  }

  render() {
    return (
      <div>
        <Header 
          target={this.state.target}
          modality={this.state.modality}
          server_url={this.state.server_url}/>
        {this.render_results()}
        {/* <Modal 
          data={this.state} 
          clickHandler={this.hideModal}/> */}
      </div>
    )
  }
}
